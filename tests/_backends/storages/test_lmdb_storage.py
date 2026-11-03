"""LMDB-specific behaviour beyond the shared StorageProtocol compliance suite.

Covers the in-process writer slot: nested writers, threads and coroutines
queueing on it, bounded waits, and snapshots never waiting behind a writer.
"""

from __future__ import annotations

import asyncio
import gc
import json
import multiprocessing as mp
import os
import threading
import time
from pathlib import Path

import pytest

from tests._backends.storages import _write_slot_cases as cases
from virtuals._backends.storages.lmdb import LMDBStorage, LMDBWriteSlot
from virtuals.codecs import BinaryCodec
from virtuals.tkv.storage import (
    StorageClosedError,
    StorageError,
    StorageLockTimeoutError,
    StorageNestedTransactionError,
    StorageScanOptions,
)
from virtuals.tkv.types import EMPTY


MAP = 64 * 1024 * 1024


@pytest.fixture
def storage(tmp_path: Path):
    s = LMDBStorage(
        path=tmp_path / "db",
        codec=BinaryCodec(),
        map_size=64 * 1024 * 1024,
        busy_timeout=5.0,
    )
    s.open()
    yield s
    s.close()


def _inc(tx, key: tuple = ("n",)) -> None:
    cur = tx.get(key)
    tx.put(key, str(int(cur) + 1 if cur is not EMPTY else 1).encode())


def _count(storage: LMDBStorage, key: tuple = ("n",)) -> int:
    with storage.snapshot() as snap:
        return int(snap.get(key))


# =========================================================================
# Nested writers
# =========================================================================


@pytest.mark.timeout(10)
def test_same_thread_second_writer_raises_nested(storage: LMDBStorage) -> None:
    tx = storage.begin_transaction()
    t0 = time.monotonic()
    with pytest.raises(StorageNestedTransactionError, match="wait on itself"):
        storage.begin_transaction()
    with pytest.raises(StorageNestedTransactionError):
        storage.begin_write_batch()
    assert time.monotonic() - t0 < 0.5
    tx.abort()
    storage.begin_transaction().commit()  # slot released by the abort


@pytest.mark.timeout(10)
def test_same_task_second_writer_raises_nested(storage: LMDBStorage) -> None:
    async def main() -> None:
        slot = await storage.areserve_write_slot()
        assert isinstance(slot, LMDBWriteSlot)
        tx = storage.begin_transaction(write_slot=slot)
        with pytest.raises(StorageNestedTransactionError):
            await storage.areserve_write_slot()
        with pytest.raises(StorageNestedTransactionError):
            storage.begin_transaction()
        tx.commit()
        slot.release()  # no-op: the transaction owned the slot
        storage.begin_transaction().commit()

    asyncio.run(main())


# =========================================================================
# Writers: coroutines and threads
# =========================================================================


@pytest.mark.timeout(10)
def test_coroutines_queue_on_slot(storage: LMDBStorage) -> None:
    async def inc() -> None:
        slot = await storage.areserve_write_slot()
        try:
            tx = storage.begin_transaction(write_slot=slot)
            try:
                cur = tx.get(("n",))
                await asyncio.sleep(0.005)
                tx.put(("n",), str(int(cur) + 1 if cur is not EMPTY else 1).encode())
            except BaseException:
                tx.abort()
                raise
            tx.commit()
        finally:
            slot.release()

    async def main() -> None:
        await asyncio.gather(*(inc() for _ in range(10)))

    t0 = time.monotonic()
    asyncio.run(main())
    assert time.monotonic() - t0 < 2.0
    assert _count(storage) == 10


@pytest.mark.timeout(20)
def test_threads_serialize_writers(storage: LMDBStorage) -> None:
    n_threads, rounds = 4, 50
    errors: list[BaseException] = []

    def work() -> None:
        try:
            for _ in range(rounds):
                with storage.transaction() as tx:
                    _inc(tx)
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=work) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(15)
    assert not errors
    assert _count(storage) == n_threads * rounds


@pytest.mark.timeout(10)
def test_sync_begin_behind_loop_peer_fails_fast(storage: LMDBStorage) -> None:
    async def main() -> None:
        slot = await storage.areserve_write_slot()
        tx = storage.begin_transaction(write_slot=slot)

        async def peer() -> None:
            with pytest.raises(StorageLockTimeoutError, match="areserve_write_slot"):
                storage.begin_transaction()

        await asyncio.create_task(peer())
        tx.commit()

    asyncio.run(main())


@pytest.mark.timeout(10)
def test_lock_timeout_between_threads(tmp_path: Path) -> None:
    with LMDBStorage(
        path=tmp_path / "db", codec=BinaryCodec(), map_size=64 * 1024 * 1024, busy_timeout=0.1
    ) as s:
        tx = s.begin_transaction()
        err: list[BaseException] = []

        def other() -> None:
            try:
                s.begin_transaction()
            except BaseException as e:
                err.append(e)

        t = threading.Thread(target=other)
        t.start()
        t.join(5)
        assert err and isinstance(err[0], StorageLockTimeoutError)
        tx.abort()


@pytest.mark.timeout(10)
def test_unused_slot_is_released(storage: LMDBStorage) -> None:
    async def main() -> None:
        (await storage.areserve_write_slot()).release()
        storage.begin_transaction().commit()

    asyncio.run(main())


# =========================================================================
# Snapshots never wait on writers
# =========================================================================


@pytest.mark.timeout(10)
def test_snapshot_does_not_wait_for_writer(storage: LMDBStorage) -> None:
    """A held writer, and a writer queued behind it, never delay snapshots."""
    tx = storage.begin_transaction()
    tx.put(("k",), b"pending")

    queued_done = threading.Event()

    def queued() -> None:
        with storage.transaction() as t:
            t.put(("q",), b"1")
        queued_done.set()

    th = threading.Thread(target=queued)
    th.start()
    time.sleep(0.05)  # queued writer is now waiting for the slot
    assert not queued_done.is_set()

    result: list[object] = []

    def read() -> None:
        t0 = time.monotonic()
        with storage.snapshot() as snap:
            result.append(snap.get(("k",)))
        result.append(time.monotonic() - t0)

    reader = threading.Thread(target=read)
    reader.start()
    reader.join(2)
    assert not reader.is_alive()
    assert result[0] is EMPTY
    assert result[1] < 0.25  # timing: a snapshot never queues behind the writers

    tx.commit()
    th.join(5)
    assert queued_done.is_set()


# =========================================================================
# Writer slot: cancellation, handoff, fork, racing close
# =========================================================================


@pytest.mark.timeout(10)
@pytest.mark.parametrize("ctx", ["transaction", "batch_write"])
def test_cancel_inside_block_releases_slot(storage: LMDBStorage, ctx: str) -> None:
    cases.cancel_inside_block(storage, ctx)


@pytest.mark.timeout(10)
@pytest.mark.parametrize("ctx", ["transaction", "batch_write"])
def test_keyboard_interrupt_inside_block_releases_slot(storage: LMDBStorage, ctx: str) -> None:
    cases.keyboard_interrupt_inside_block(storage, ctx)


@pytest.mark.timeout(10)
def test_cancel_while_reserving_holds_nothing(storage: LMDBStorage) -> None:
    cases.cancel_while_reserving(storage)


@pytest.mark.timeout(10)
def test_cancel_between_reserve_and_begin_releases_slot(storage: LMDBStorage) -> None:
    cases.cancel_between_reserve_and_begin(storage)


@pytest.mark.timeout(10)
def test_to_thread_nested_raises_fast(storage: LMDBStorage) -> None:
    cases.to_thread_nested_raises_fast(storage)


@pytest.mark.timeout(10)
def test_asyncio_run_inside_sync_txn_is_nested(storage: LMDBStorage) -> None:
    cases.asyncio_run_inside_sync_txn_is_nested(storage)


@pytest.mark.timeout(10)
def test_racing_ends_release_slot_once(storage: LMDBStorage) -> None:
    tx = storage.begin_transaction()
    # End LMDB's side once, on the thread that began it; the race is in giving back the slot.
    storage._writer_thread.run(tx._lmdb_txn.abort)
    cases.racing_ends_release_once(storage, tx, lambda t: t._mark_closed(), "_holds_write_slot")


@pytest.mark.timeout(10)
@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
@pytest.mark.parametrize("child_thread", [False, True], ids=["forking-thread", "new-thread"])
def test_fork_during_write_child_write_fails_clearly(
    storage: LMDBStorage, child_thread: bool
) -> None:
    """A child forked while a parent thread holds a write gets a clear error.

    From a new child thread, LMDB's inherited writer state would refuse the
    write with a cryptic error, from the forking thread it would wait on
    the dead holder; the storage refuses every write and says why instead.
    """
    held, go = threading.Event(), threading.Event()

    def holder() -> None:  # another parent thread holds the slot at fork
        with storage.transaction() as tx:
            tx.put(("parent",), b"1")
            held.set()
            go.wait()

    t = threading.Thread(target=holder)
    t.start()
    held.wait()
    pid = os.fork()
    if pid == 0:  # child
        code = 1

        def clear(begin) -> bool:
            try:
                begin()
            except StorageError as e:
                return type(e) is StorageError and "fork" in str(e)
            return False

        def child() -> None:
            nonlocal code
            go.wait(0.2)  # never set here: the parent's commit cannot reach the child
            if not (
                clear(storage.begin_transaction)
                and clear(storage.begin_write_batch)
                and clear(lambda: asyncio.run(storage.areserve_write_slot()))
            ):
                return
            storage.close()  # returns, does not hang on the inherited env
            code = 0

        try:
            if child_thread:
                x = threading.Thread(target=child)
                x.start()
                x.join()
            else:
                child()
        finally:
            os._exit(code)
    time.sleep(0.05)
    go.set()
    t.join()
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    with storage.snapshot() as snap:
        assert snap.get(("parent",)) == b"1"
    storage.begin_transaction().commit()


@pytest.mark.timeout(10)
@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
@pytest.mark.parametrize("child_thread", [False, True], ids=["forking-thread", "new-thread"])
def test_fork_without_write_child_writes(storage: LMDBStorage, child_thread: bool) -> None:
    """With no write in progress at fork, the child writes from any thread."""
    with storage.transaction() as tx:
        tx.put(("parent",), b"1")
    pid = os.fork()
    if pid == 0:  # child
        code = 1

        def child() -> None:
            nonlocal code
            for value in (b"1", b"2"):
                with storage.transaction() as ctx:
                    ctx.put(("child",), value)
            code = 0

        try:
            if child_thread:
                x = threading.Thread(target=child)
                x.start()
                x.join()
            else:
                child()
        finally:
            os._exit(code)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    with storage.snapshot() as snap:
        assert snap.get(("child",)) == b"2"
    storage.begin_transaction().commit()


@pytest.mark.timeout(10)
def test_handoff_worker_is_not_nested(storage: LMDBStorage) -> None:
    cases.handoff_worker_is_not_nested(storage)


@pytest.mark.timeout(10)
def test_handoff_then_writers_on_same_worker(storage: LMDBStorage) -> None:
    cases.handoff_then_writers_on_same_worker(storage)


@pytest.mark.timeout(10)
def test_spawned_task_waits(storage: LMDBStorage) -> None:
    cases.spawned_task_waits(storage)


class _SignalInsideCommit:
    """Wraps an LMDB write txn so ``inside`` is set while its commit runs."""

    def __init__(self, txn, inside: threading.Event) -> None:
        self._txn, self._inside = txn, inside

    def commit(self) -> None:
        self._inside.set()
        time.sleep(0.1)
        self._txn.commit()

    def __getattr__(self, name: str):
        return getattr(self._txn, name)


@pytest.mark.timeout(10)
@pytest.mark.parametrize("ctx", ["transaction", "write_batch"])
def test_abort_racing_commit_keeps_one_outcome(storage: LMDBStorage, ctx: str) -> None:
    """An abort from another thread while the owner commits waits, then is a no-op."""
    tx = storage.begin_transaction() if ctx == "transaction" else storage.begin_write_batch()
    tx.put(("k",), b"committed")
    inside = threading.Event()
    raw, tx._lmdb_txn = tx._lmdb_txn, _SignalInsideCommit(tx._lmdb_txn, inside)
    with storage._env_lock:  # the live set tracks handles by identity: track the wrapper
        storage._live_writes.discard(raw)
        storage._live_writes.add(tx._lmdb_txn)
    errors: list[BaseException] = []

    def abort_from_elsewhere() -> None:
        inside.wait(5)
        try:
            tx.abort()
        except BaseException as e:
            errors.append(e)

    t = threading.Thread(target=abort_from_elsewhere)
    t.start()
    if ctx == "transaction":
        tx.commit()
        assert (tx._committed, tx._aborted) == (True, False)
    else:
        tx.write()
        assert (tx._written, tx._aborted) == (True, False)
    t.join(5)
    assert not errors, errors
    with storage.snapshot() as snap:
        assert snap.get(("k",)) == b"committed"
    storage.begin_transaction().commit()


# =========================================================================
# Cross-process: bounded waits, thread-safe ends, close, readers, map growth
# =========================================================================


def _open(root: str, **kw) -> LMDBStorage:
    kw.setdefault("map_size", MAP)
    st = LMDBStorage(path=Path(root) / "db", codec=BinaryCodec(), **kw)
    st.open()
    return st


# Spawn targets: module level so the child can import them.
def _hold_write(root: str, hold: float, started, **kw) -> None:
    st = _open(root, **kw)
    tx = st.begin_transaction()
    tx.put(("held",), b"1")
    started.set()
    time.sleep(hold)
    tx.commit()
    st.close()


def _peer_write(root: str, busy_timeout: float, ready, go, out: str) -> None:
    """Open now, write when told, report how it went."""
    st = _open(root, busy_timeout=busy_timeout)
    ready.set()
    go.wait(30)
    t0 = time.monotonic()
    try:
        with st.transaction() as tx:
            tx.put(("peer",), b"1")
        result = "ok"
    except Exception as e:
        result = type(e).__name__
    took = time.monotonic() - t0
    st.close()
    Path(out).write_text(json.dumps({"result": result, "took": took}))


def _die_holding_readers(root: str, n: int, max_readers: int) -> None:
    st = _open(root, max_readers=max_readers)
    snaps = [st.begin_snapshot() for _ in range(n)]
    assert len(snaps) == n
    os._exit(0)  # reader slots stay registered


def _grow_map(root: str, map_size: int, nbytes: int) -> None:
    st = _open(root, map_size=map_size)
    with st.transaction() as tx:
        for i in range(nbytes // 65536):
            tx.put(("blob", i), os.urandom(65536))
    st.close()


def _busy_owner(st: LMDBStorage, go: threading.Event, result: list[str]) -> None:
    ctx = st.begin_transaction()
    go.set()
    try:
        for r in range(20000):
            ctx.put(("w", r), b"x")
            ctx.get(("k", r % 300))
            if r % 20 == 0:
                list(ctx.scan(StorageScanOptions(limit=40)).keys())
        result.append("finished")
    except StorageClosedError:
        result.append("closed")
    except Exception as e:
        result.append(type(e).__name__)


def _close_while_scanning(root: str, rounds: int) -> None:
    """close() from the main thread while a worker is inside cursor calls on a write txn."""
    outcomes: dict[str, int] = {}
    for i in range(rounds):
        st = _open(f"{root}/{i}", busy_timeout=2.0)
        with st.transaction() as tx:
            for k in range(300):
                tx.put(("k", k), b"v" * 50)
        go = threading.Event()
        result: list[str] = []
        th = threading.Thread(target=_busy_owner, args=(st, go, result), daemon=True)
        th.start()
        go.wait()
        time.sleep(0.0005 * (i % 7))
        st.close()
        th.join(10)
        key = result[0] if result else "hung"
        outcomes[key] = outcomes.get(key, 0) + 1
        st = _open(f"{root}/{i}")  # the env is usable again
        st.begin_transaction().commit()
        st.close()
    assert set(outcomes) <= {"closed", "finished"}, outcomes
    assert "closed" in outcomes, outcomes


def _wait_pid(pid: int) -> int:
    _, status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(status)


@pytest.fixture
def ctx():
    return mp.get_context("spawn")


@pytest.fixture
def holder(ctx, tmp_path: Path):
    """A spawned process holding a write transaction on ``tmp_path`` for 1.5s."""
    started = ctx.Event()
    p = ctx.Process(target=_hold_write, args=(str(tmp_path), 1.5, started))
    p.start()
    assert started.wait(20)
    yield p
    p.join(10)
    if p.is_alive():
        p.kill()


@pytest.mark.timeout(20)
def test_cross_process_wait_is_bounded(holder, tmp_path: Path) -> None:
    """A sync writer behind another process gives up at busy_timeout with the retryable error."""
    st = _open(str(tmp_path), busy_timeout=0.1)
    try:
        t0 = time.monotonic()
        with pytest.raises(StorageLockTimeoutError, match="another process"):
            with st.transaction() as tx:
                tx.put(("mine",), b"1")
        took = time.monotonic() - t0
        assert 0.1 <= took < 1.0, took  # timing: busy_timeout, plus slack for a loaded box
        assert not st._writer.lock.locked()
        holder.join(10)
        with st.transaction() as tx:  # the gate was given back
            tx.put(("mine",), b"1")
        with st.snapshot() as snap:
            assert snap.get(("held",)) == b"1"
            assert snap.get(("mine",)) == b"1"
    finally:
        st.close()


async def _ticking(body, tick: float = 0.005):
    """Run ``body`` beside a ticker; return (result, largest gap between ticks)."""
    gaps: list[float] = []
    last = time.perf_counter()

    async def ticker() -> None:
        nonlocal last
        while True:
            await asyncio.sleep(tick)
            now = time.perf_counter()
            gaps.append(now - last)
            last = now

    t = asyncio.create_task(ticker())
    try:
        return await body(), max(gaps, default=0.0)
    finally:
        t.cancel()


@pytest.mark.timeout(20)
def test_cross_process_wait_does_not_stall_loop(holder, tmp_path: Path) -> None:
    """The async path waits for another process off the loop: bounded, loop keeps ticking."""
    st = _open(str(tmp_path), busy_timeout=0.3)

    async def give_up():
        t0 = time.monotonic()
        with pytest.raises(StorageLockTimeoutError):
            await st.areserve_write_slot()
        return time.monotonic() - t0

    async def wait_through():
        slot = await st.areserve_write_slot()
        tx = st.begin_transaction(write_slot=slot)
        tx.put(("mine",), b"1")
        tx.commit()

    async def main() -> None:
        took, gap = await _ticking(give_up)
        assert 0.3 <= took < 1.5, took  # timing
        assert gap < 0.3, gap  # timing: the loop never froze for the wait
        st._busy_timeout = 5.0
        _, gap = await _ticking(wait_through)  # the holder commits meanwhile
        assert gap < 0.3, gap  # timing

    try:
        asyncio.run(main())
        with st.snapshot() as snap:
            assert snap.get(("mine",)) == b"1"
    finally:
        st.close()


@pytest.mark.timeout(30)
@pytest.mark.parametrize("end", ["commit", "abort"])
def test_end_on_another_thread_does_not_wedge(ctx, tmp_path: Path, end: str) -> None:
    """Begin in to_thread, end on the loop: the next writer in this process and a peer get through."""
    st = _open(str(tmp_path), busy_timeout=2.0)
    ready, go = ctx.Event(), ctx.Event()
    out = tmp_path / "peer.json"
    peer = ctx.Process(target=_peer_write, args=(str(tmp_path), 2.0, ready, go, str(out)))
    peer.start()
    assert ready.wait(20)  # the peer has the env open before we touch the writer lock
    try:

        async def main() -> None:
            tx = await asyncio.to_thread(st.begin_transaction)
            tx.put(("k",), b"1")
            getattr(tx, end)()

        asyncio.run(main())
        t0 = time.monotonic()
        with st.transaction() as tx:  # a fresh thread, in effect: the pool worker is gone
            tx.put(("after",), b"1")
        assert time.monotonic() - t0 < 1.0
        go.set()
        peer.join(10)
        assert peer.exitcode == 0
        report = json.loads(out.read_text())
        assert report["result"] == "ok", report
        assert report["took"] < 1.0, report
        with st.snapshot() as snap:
            assert snap.get(("peer",)) == b"1"
            assert snap.get(("k",)) == (b"1" if end == "commit" else EMPTY)
    finally:
        if peer.is_alive():
            peer.kill()
        st.close()


@pytest.mark.timeout(30)
def test_close_with_live_worker_txn_does_not_wedge_peer(ctx, tmp_path: Path) -> None:
    """close() from the main thread while a worker thread holds a txn: a peer still writes."""
    st = _open(str(tmp_path), busy_timeout=2.0)
    ready, go = ctx.Event(), ctx.Event()
    out = tmp_path / "peer.json"
    peer = ctx.Process(target=_peer_write, args=(str(tmp_path), 2.0, ready, go, str(out)))
    peer.start()
    assert ready.wait(20)
    keep = threading.Event()
    box: dict = {}

    def worker() -> None:
        box["tx"] = st.begin_transaction()
        box["tx"].put(("k",), b"1")
        box["begun"].set()
        keep.wait(10)

    box["begun"] = threading.Event()
    th = threading.Thread(target=worker, daemon=True)
    th.start()
    assert box["begun"].wait(5)
    try:
        t0 = time.monotonic()
        st.close()
        assert time.monotonic() - t0 < 1.0
        assert box["tx"].is_closed
        go.set()
        peer.join(10)
        assert peer.exitcode == 0
        report = json.loads(out.read_text())
        assert report["result"] == "ok", report
        assert report["took"] < 1.0, report
    finally:
        keep.set()
        if peer.is_alive():
            peer.kill()
    st = _open(str(tmp_path))
    with st.snapshot() as snap:
        assert snap.get(("peer",)) == b"1"
        assert snap.get(("k",)) is EMPTY
    st.close()


@pytest.mark.timeout(60)
def test_close_while_owner_is_inside_cursor_calls(ctx, tmp_path: Path) -> None:
    """close() racing a worker's puts, gets and scans on a write txn: no crash, clean error."""
    p = ctx.Process(target=_close_while_scanning, args=(str(tmp_path), 30))
    p.start()
    p.join(50)
    if p.is_alive():
        p.kill()
        pytest.fail("close() racing an owner's calls hung")
    assert p.exitcode == 0, f"exit code {p.exitcode} (negative: a signal, e.g. segfault)"


@pytest.mark.timeout(20)
def test_close_while_a_thread_waits_behind_a_peer(holder, tmp_path: Path) -> None:
    """close() returns while a thread waits for the writer in another process.

    The waiter gets StorageClosedError, snapshots from a third thread were
    never blocked by the waiter, and the storage reopens.
    """
    st = _open(str(tmp_path), busy_timeout=5.0)
    outcome: list = []

    def waiter() -> None:
        try:
            st.begin_transaction()
            outcome.append("begun")
        except Exception as e:
            outcome.append(type(e).__name__)

    th = threading.Thread(target=waiter, daemon=True)
    th.start()
    time.sleep(0.2)
    assert st._writer.lock.locked()  # the waiter holds the slot and polls the gate

    t0 = time.monotonic()
    with st.snapshot() as snap:
        assert snap.get(("held",)) is EMPTY
    assert time.monotonic() - t0 < 0.5  # timing: never behind the gate waiter

    t0 = time.monotonic()
    st.close()
    assert time.monotonic() - t0 < 1.0
    th.join(5)
    assert outcome == ["StorageClosedError"]
    assert not st._writer.lock.locked()
    holder.join(10)
    st.open()
    with st.transaction() as tx:
        tx.put(("after",), b"1")
    st.close()


@pytest.mark.timeout(20)
def test_dropped_snapshots_free_reader_slots(tmp_path: Path) -> None:
    """Snapshots nobody closed are collected, so they do not pin reader slots."""
    st = _open(str(tmp_path), max_readers=8)
    try:
        for _ in range(3 * 8):
            snap = st.begin_snapshot()
            snap.get(("k",))
            del snap
        gc.collect()
        assert len(st._active_snapshots) == 0
        snaps = [st.begin_snapshot() for _ in range(8)]  # a full table's worth still opens
        for s in snaps:
            s.close()
    finally:
        st.close()


@pytest.mark.timeout(30)
def test_stale_readers_of_a_dead_process_are_reclaimed(ctx, tmp_path: Path) -> None:
    """Reader slots of a process that died mid-snapshot are swept at open and when the table fills."""
    max_readers = 4
    st = _open(str(tmp_path), max_readers=max_readers)
    p = ctx.Process(target=_die_holding_readers, args=(str(tmp_path), max_readers - 1, max_readers))
    p.start()
    p.join(20)
    assert p.exitcode == 0
    assert st._env.readers().count("\n") >= max_readers - 1  # dead pid still listed
    snaps = [st.begin_snapshot() for _ in range(max_readers)]  # full table: swept and retried
    for s in snaps:
        s.close()
    st.close()

    p = ctx.Process(target=_die_holding_readers, args=(str(tmp_path), max_readers - 1, max_readers))
    p.start()
    p.join(20)
    st = _open(str(tmp_path), max_readers=max_readers)  # swept at open
    try:
        assert st._env.readers().count("\n") <= 1, st._env.readers()
        snaps = [st.begin_snapshot() for _ in range(max_readers)]
        for s in snaps:
            s.close()
    finally:
        st.close()


@pytest.mark.timeout(30)
def test_peer_grows_map_past_our_mapping(ctx, tmp_path: Path) -> None:
    """A peer opened with a larger map writes past ours: the next begin here remaps and works."""
    st = _open(str(tmp_path), map_size=1 << 20)
    with st.transaction() as tx:
        tx.put(("k",), b"1")
    p = ctx.Process(target=_grow_map, args=(str(tmp_path), 16 << 20, 3 << 20))
    p.start()
    p.join(20)
    assert p.exitcode == 0
    try:
        with st.snapshot() as snap:
            assert snap.get(("blob", 0)) is not EMPTY
        with st.transaction() as tx:
            tx.put(("after",), b"1")
        with st.snapshot() as snap:
            assert snap.get(("after",)) == b"1"
    finally:
        st.close()


@pytest.mark.timeout(20)
@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_fork_while_env_lock_is_held_elsewhere(storage: LMDBStorage) -> None:
    """A fork while another parent thread holds the storage lock: the child is not deadlocked."""
    with storage.transaction() as tx:
        tx.put(("parent",), b"1")
    held, go = threading.Event(), threading.Event()

    def hold_lock() -> None:
        with storage._env_lock:
            held.set()
            go.wait(5)

    t = threading.Thread(target=hold_lock)
    t.start()
    held.wait()
    pid = os.fork()
    if pid == 0:
        code = 1
        try:
            with storage.snapshot() as snap:
                assert snap.get(("parent",)) == b"1"
            with storage.transaction() as tx:
                tx.put(("child",), b"1")
            storage.close()
            code = 0
        finally:
            os._exit(code)
    try:
        assert _wait_pid(pid) == 0
    finally:
        go.set()
        t.join()
    with storage.snapshot() as snap:
        assert snap.get(("child",)) == b"1"


@pytest.mark.timeout(30)
@pytest.mark.parametrize("what", ["transaction", "write_batch"])
def test_dropped_write_context_is_aborted_and_frees_writers(ctx, tmp_path: Path, what: str) -> None:
    """A transaction or batch dropped unended is aborted on collection, in process and for peers."""
    st = _open(str(tmp_path), busy_timeout=2.0)
    ready, go = ctx.Event(), ctx.Event()
    out = tmp_path / "peer.json"
    peer = ctx.Process(target=_peer_write, args=(str(tmp_path), 2.0, ready, go, str(out)))
    peer.start()
    assert ready.wait(20)

    def drop() -> None:
        t = st.begin_transaction() if what == "transaction" else st.begin_write_batch()
        t.put(("dropped",), b"1")  # goes out of scope: neither ended nor released

    try:
        th = threading.Thread(target=drop)
        th.start()
        th.join(5)
        gc.collect()
        assert not st._writer.lock.locked()
        assert not st._gate_held
        t0 = time.monotonic()
        with st.transaction() as tx:  # another thread's drop freed the slot and the gate
            tx.put(("after",), b"1")
        assert time.monotonic() - t0 < 1.0  # timing
        drop()  # same thread: the finished hold must not read as nested either
        gc.collect()
        go.set()
        peer.join(10)
        assert peer.exitcode == 0
        report = json.loads(out.read_text())
        assert report["result"] == "ok", report
        assert report["took"] < 1.0, report  # timing
        with st.snapshot() as snap:
            assert snap.get(("dropped",)) is EMPTY
            assert snap.get(("after",)) == b"1"
            assert snap.get(("peer",)) == b"1"
    finally:
        if peer.is_alive():
            peer.kill()
        st.close()


# =========================================================================
# Review regressions: cancel after begin lands, exit with an open txn, close racing begins
# =========================================================================


@pytest.mark.timeout(20)
def test_cancel_after_begin_lands_aborts_on_writer_thread(tmp_path: Path) -> None:
    """A reserve cancelled after its begin landed is undone on the writer thread, not the loop.

    Hung before the fix: the undo aborted the LMDB txn on the loop thread,
    which left the writer mutex owned by the writer thread for good, and
    every later write timed out.
    """
    st = _open(str(tmp_path), busy_timeout=1.0)
    seen: list[str] = []
    orig = st._abort_abandoned

    def spy(txn) -> None:
        seen.append(threading.current_thread().name)
        orig(txn)

    st._abort_abandoned = spy

    async def main() -> None:
        task = asyncio.create_task(st.areserve_write_slot())
        for _ in range(3):
            await asyncio.sleep(0)  # the task passes the slot and gate and queues the begin
        time.sleep(0.2)  # the loop is busy while the begin lands on the writer thread
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    try:
        asyncio.run(main())
        deadline = time.monotonic() + 5
        while not seen and time.monotonic() < deadline:
            time.sleep(0.01)
        assert seen and all(name.startswith("lmdb-writer") for name in seen), seen
        t0 = time.monotonic()
        with st.transaction() as tx:
            tx.put(("x",), b"1")
        assert time.monotonic() - t0 < 0.5
    finally:
        st.close()


_EXIT_WITH_OPEN_TXN = """
import sys
from pathlib import Path
from virtuals.codecs import BinaryCodec
from virtuals._backends.storages.lmdb import LMDBStorage
st = LMDBStorage(path=Path(sys.argv[1]) / "db", codec=BinaryCodec(), map_size=64 << 20)
st.open()
mode = sys.argv[2]
def work():
    tx = st.begin_transaction()
    tx.put(("x",), b"1")
    if mode == "exc":
        raise RuntimeError("boom")
    return tx
tx = work()  # "plain": a module global holds the open txn at exit
"""


@pytest.mark.timeout(30)
@pytest.mark.parametrize("mode", ["plain", "exc"])
def test_interpreter_exit_with_open_write_txn_does_not_hang(tmp_path: Path, mode: str) -> None:
    """A write txn still open at exit (dropped, or in an uncaught exception's frame) exits fast.

    Hung before the fix: the finalizer waited on the writer thread, which
    cannot run Python during finalization.
    """
    import subprocess
    import sys

    t0 = time.monotonic()
    proc = subprocess.run(  # noqa: S603 (our own script)
        [sys.executable, "-c", _EXIT_WITH_OPEN_TXN, str(tmp_path), mode],
        capture_output=True,
        text=True,
        timeout=20,
    )
    took = time.monotonic() - t0
    assert took < 5.0, f"exit took {took:.1f}s"
    assert proc.returncode == (1 if mode == "exc" else 0), proc.stderr
    st = _open(str(tmp_path))  # the env is usable afterwards
    try:
        with st.transaction() as tx:
            tx.put(("after",), b"1")
    finally:
        st.close()


def _close_racing_writers(root: str, rounds: int) -> None:
    """close() from the main thread while four threads loop single-put transactions."""
    for r in range(rounds):
        st = _open(f"{root}/{r}", busy_timeout=2.0, sync=False, metasync=False)
        done: list[str] = []

        def write_until_closed(i: int, st: LMDBStorage = st, done: list[str] = done) -> None:
            try:
                while True:
                    with st.transaction() as t:
                        t.put(("k", i), b"1")
            except StorageError as e:
                done.append(type(e).__name__)
            except Exception as e:
                done.append(f"other:{type(e).__name__}")

        threads = [
            threading.Thread(target=write_until_closed, args=(i,), daemon=True) for i in range(4)
        ]
        for t in threads:
            t.start()
        time.sleep(0.002 + (r % 7) * 0.001)
        st.close()
        for t in threads:
            t.join(5)
        assert not any(t.is_alive() for t in threads), f"round {r}: writer stuck after close"
        assert not [d for d in done if d.startswith("other")], done
        st = _open(f"{root}/{r}", busy_timeout=1.0)  # the env is sane: reopen and write
        with st.transaction() as t:
            t.put(("z",), b"1")
        st.close()


@pytest.mark.timeout(120)
def test_close_racing_writer_threads_never_hangs(ctx, tmp_path: Path) -> None:
    """close() mid-flight against threads looping transactions, 30 rounds, no hang.

    Hung before the fix: a begin that landed on the writer thread after
    close() swept the tracked contexts was never ended, and env.close()
    then blocked on it forever.
    """
    p = ctx.Process(target=_close_racing_writers, args=(str(tmp_path), 30))
    p.start()
    p.join(100)
    if p.is_alive():
        p.kill()
        pytest.fail("close() racing writer threads hung")
    assert p.exitcode == 0, f"exit code {p.exitcode}"


@pytest.mark.timeout(20)
def test_write_txn_data_calls_run_on_any_thread(ctx, tmp_path: Path) -> None:
    """Pins the thread model: begin on A, put/get/scan on B, commit on C; a peer writes after.

    Begin, commit and abort run on the writer thread. The data calls run on
    the caller's thread, which liblmdb 0.9.36 allows for a write txn (its
    only thread-bound state is the writer mutex, see ``context.py``).
    """
    st = _open(str(tmp_path), busy_timeout=2.0)
    ready, go = ctx.Event(), ctx.Event()
    out = tmp_path / "peer.json"
    peer = ctx.Process(target=_peer_write, args=(str(tmp_path), 2.0, ready, go, str(out)))
    peer.start()
    assert ready.wait(20)
    box: dict = {}

    def on(name: str, fn) -> None:
        t = threading.Thread(target=lambda: box.__setitem__(name, fn()), name=name)
        t.start()
        t.join(10)
        assert name in box, f"{name} did not finish"

    try:
        on("A", st.begin_transaction)
        tx = box["A"]

        def data() -> list:
            tx.put(("a",), b"1")
            tx.put(("b",), b"2")
            tx.delete(("a",))
            got = tx.get(("b",))
            return [got, list(tx.scan(StorageScanOptions()).keys())]

        on("B", data)
        assert box["B"] == [b"2", [("b",)]]
        on("C", tx.commit)
        with st.snapshot() as snap:
            assert snap.get(("b",)) == b"2"
            assert snap.get(("a",)) is EMPTY
        go.set()
        peer.join(10)
        assert json.loads(out.read_text())["result"] == "ok"
    finally:
        if peer.is_alive():
            peer.kill()
        st.close()


# =========================================================================
# Re-review regressions: fork with the gate lock held, ends racing close, gate fd reuse
# =========================================================================


@pytest.mark.timeout(20)
@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_fork_while_gate_lock_is_held_elsewhere(storage: LMDBStorage) -> None:
    """A fork while another parent thread holds the gate's lock: the child's first write works.

    Hung before the fix: the child reopened the gate under the inherited lock.
    """
    with storage.transaction() as tx:
        tx.put(("parent",), b"1")
    held, go = threading.Event(), threading.Event()

    def hold_lock() -> None:
        with storage._gate._mu:
            held.set()
            go.wait(5)

    t = threading.Thread(target=hold_lock)
    t.start()
    held.wait()
    pid = os.fork()
    if pid == 0:
        code = 1
        try:
            with storage.transaction() as tx:
                tx.put(("child",), b"1")
            storage.close()
            code = 0
        finally:
            os._exit(code)
    try:
        assert _wait_pid(pid) == 0
    finally:
        go.set()
        t.join()
    with storage.snapshot() as snap:
        assert snap.get(("child",)) == b"1"


def _peer_can_write_now(ctx, tmp_path: Path):
    """A spawned peer with the env open; call the returned function to make it write promptly."""
    ready, go = ctx.Event(), ctx.Event()
    out = tmp_path / "peer.json"
    peer = ctx.Process(target=_peer_write, args=(str(tmp_path), 2.0, ready, go, str(out)))
    peer.start()
    assert ready.wait(20)

    def check() -> None:
        go.set()
        peer.join(10)
        if peer.is_alive():
            peer.kill()
        assert peer.exitcode == 0
        report = json.loads(out.read_text())
        assert report["result"] == "ok", report
        assert report["took"] < 1.0, report  # timing

    return check


@pytest.mark.timeout(30)
def test_finalizer_racing_close_leaves_no_live_write(ctx, tmp_path: Path, monkeypatch) -> None:
    """A txn finalized on one thread while close() runs on another: close() ends it itself.

    Before the fix the finalizer marked the context closed, close() skipped
    it and stopped the writer thread, and the abort was never queued.
    """
    from virtuals._backends.storages.lmdb._writer import WriterThread

    st = _open(str(tmp_path), busy_timeout=2.0)
    peer = _peer_can_write_now(ctx, tmp_path)
    original = WriterThread.try_run
    outcomes: list[bool] = []

    def slow_try_run(self, fn, timeout):
        time.sleep(0.3)  # the finalizer is descheduled right after it marked the context closed
        result = original(self, fn, timeout)
        outcomes.append(result)
        return result

    monkeypatch.setattr(WriterThread, "try_run", slow_try_run)

    def drop() -> None:
        tx = st.begin_transaction()
        tx.put(("x",), b"1")
        del tx

    t = threading.Thread(target=drop)
    t.start()
    time.sleep(0.1)
    st.close()
    t.join(5)
    assert outcomes == [False]  # the finalizer found the writer thread gone
    assert not st._live_writes  # and close() had ended the txn itself
    peer()
    st = _open(str(tmp_path))
    try:
        with st.snapshot() as snap:
            assert snap.get(("x",)) is EMPTY
    finally:
        st.close()


@pytest.mark.timeout(30)
def test_slot_release_racing_close_leaves_no_live_write(ctx, tmp_path: Path, monkeypatch) -> None:
    """slot.release() on one thread while close() runs on another: the txn is ended once."""
    from virtuals._backends.storages.lmdb._writer import WriterThread

    st = _open(str(tmp_path), busy_timeout=2.0)
    peer = _peer_can_write_now(ctx, tmp_path)
    slot = asyncio.run(st.areserve_write_slot())
    original = WriterThread.run

    def slow_run(self, fn, **kw):
        if threading.current_thread().name == "releaser":
            time.sleep(0.3)  # between taking the txn from the slot and ending it
        return original(self, fn, **kw)

    monkeypatch.setattr(WriterThread, "run", slow_run)
    t = threading.Thread(target=slot.release, name="releaser")
    t.start()
    time.sleep(0.05)
    st.close()
    t.join(5)
    assert not t.is_alive()
    assert not st._live_writes
    assert not st._writer.lock.locked()
    peer()


@pytest.mark.timeout(30)
def test_gate_helper_never_locks_a_reused_descriptor_number(tmp_path: Path, monkeypatch) -> None:
    """The helper flocks its own duplicate, so an fd number reused after close() is never locked.

    A peer holds the gate; our wait times out; the storage closes (freeing
    the gate's descriptor number) and reopens while the helper is still
    about to call flock. Afterwards a peer must be able to take the gate.
    """
    import fcntl

    from virtuals._backends.storages.lmdb import _writer

    real = _writer.fcntl

    class Shim:
        LOCK_EX, LOCK_NB, LOCK_UN = real.LOCK_EX, real.LOCK_NB, real.LOCK_UN

        @staticmethod
        def flock(fd, op):
            if threading.current_thread().name == "lmdb-gate" and op == real.LOCK_EX:
                time.sleep(0.5)
            return real.flock(fd, op)

    monkeypatch.setattr(_writer, "fcntl", Shim)
    st = _open(str(tmp_path), busy_timeout=0.1)
    gate_file = tmp_path / "db" / "writer.lock"
    peer = os.open(gate_file, os.O_RDWR)
    fcntl.flock(peer, fcntl.LOCK_EX)  # a peer holds the gate
    try:
        with pytest.raises(StorageLockTimeoutError):
            with st.transaction() as tx:
                tx.put(("a",), b"1")
        st.close()
        fcntl.flock(peer, fcntl.LOCK_UN)
        other = os.open(gate_file, os.O_RDWR)  # any open may take the freed number
        st.open()
        time.sleep(1.0)  # the helper's delayed flock runs now
        peer2 = os.open(gate_file, os.O_RDWR)
        try:
            fcntl.flock(peer2, fcntl.LOCK_EX | fcntl.LOCK_NB)  # must be free
            fcntl.flock(peer2, fcntl.LOCK_UN)
        finally:
            os.close(peer2)
        os.close(other)
        with st.transaction() as tx:
            tx.put(("a",), b"1")
    finally:
        os.close(peer)
        st.close()
