"""The kv contract on the RocksDB serial variant.

Sections 1-4 of the contract as point tests: serializable transactions
(lost update, write skew, insert race, phantom on scan), read your own
writes, point-in-time snapshots (repeatable, never torn), the nested
transaction error, retryable error classes, and coroutine writers that
queue on the writer slot without stalling the event loop.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from typing import TYPE_CHECKING, Any

import pytest
import rdbpy

from virtuals._backends.storages.rocksdb import RocksDBSerialStorage, RocksDBStorage
from virtuals._backends.storages.rocksdb.context import classify_rdbpy
from virtuals.codecs import BinaryCodec
from virtuals.tkv import StorageScanOptions
from virtuals.tkv.filter import PrefixFilter
from virtuals.tkv.storage import (
    StorageClosedError,
    StorageError,
    StorageLockTimeoutError,
    StorageNestedTransactionError,
    StorageOperationError,
    StorageTransactionConflictError,
)
from virtuals.tkv.types import EMPTY


if TYPE_CHECKING:
    from pathlib import Path


RETRYABLE = (StorageTransactionConflictError, StorageLockTimeoutError)

K = ("counter",)
X, Y = ("x",), ("y",)

ROUNDS = 20
PAUSE = 0.001  # between a read and its dependent write, so a race can happen
BARRIER = 0.05  # how long a racer waits for its peer after reading


@pytest.fixture
def storage(tmp_path: Path):
    s = RocksDBStorage(path=tmp_path / "db", codec=BinaryCodec(), busy_timeout=5.0)
    s.open()
    yield s
    s.close()


def _retrying(fn, attempts: int = 200):
    for i in range(attempts):
        try:
            return fn()
        except RETRYABLE:
            time.sleep(min(0.0005 * (i + 1), 0.005))
    raise AssertionError("still conflicting")


def _put(storage, items: dict) -> None:
    def body() -> None:
        with storage.transaction() as t:
            for k, v in items.items():
                t.put(k, v)

    _retrying(body)


def _read(storage, key):
    with storage.snapshot() as s:
        return s.get(key)


def _threads(*targets, timeout: float = 10.0) -> list:
    results: list = [None] * len(targets)
    errors: list[BaseException] = []

    def wrap(i, fn):
        try:
            results[i] = fn()
        except BaseException as e:
            errors.append(e)

    threads = [
        threading.Thread(target=wrap, args=(i, fn), daemon=True) for i, fn in enumerate(targets)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout)
    assert not any(t.is_alive() for t in threads), "stalled"
    if errors:
        raise errors[0]
    return results


def _wait(barrier: threading.Barrier) -> None:
    try:
        barrier.wait(timeout=BARRIER)
    except threading.BrokenBarrierError:
        pass


# =============================================================================
# Section 1: serializable transactions
# =============================================================================


def test_variant_naming_and_process_model() -> None:
    assert RocksDBSerialStorage is RocksDBStorage
    assert RocksDBStorage.process_model == "single"


def test_lost_update_threads(storage: RocksDBStorage) -> None:
    _put(storage, {K: 0})
    threads, iters = 4, 25

    def worker() -> None:
        for _ in range(iters):

            def body() -> None:
                with storage.transaction() as t:
                    v = t.get(K)
                    time.sleep(PAUSE)
                    t.put(K, v + 1)

            _retrying(body)

    _threads(*[worker] * threads)
    assert _read(storage, K) == threads * iters


def test_write_skew_disjoint_keys(storage: RocksDBStorage) -> None:
    """At most one of x, y is on: each claimer sets its own only if the other read off."""
    barrier = threading.Barrier(2)
    violations = 0
    for _ in range(ROUNDS):
        _put(storage, {X: False, Y: False})
        barrier.reset()

        def claim(mine, other):
            def body() -> None:
                with storage.transaction() as t:
                    seen = t.get(other)
                    _wait(barrier)
                    if not seen:
                        t.put(mine, True)

            return lambda: _retrying(body)

        _threads(claim(X, Y), claim(Y, X))
        if _read(storage, X) and _read(storage, Y):
            violations += 1
    assert violations == 0


def test_insert_race(storage: RocksDBStorage) -> None:
    """Create-if-absent on one key: exactly one creator wins."""
    barrier = threading.Barrier(2)
    doubles = 0
    for r in range(ROUNDS):
        key = ("slot", r)
        barrier.reset()

        def create(me, key=key):
            def body() -> bool:
                with storage.transaction() as t:
                    present = t.get(key)
                    _wait(barrier)
                    if present is EMPTY:
                        t.put(key, me)
                        return True
                    return False

            return lambda: _retrying(body)

        if sum(_threads(create("a"), create("b"))) != 1:
            doubles += 1
    assert doubles == 0


def test_scan_write_skew_no_phantom(storage: RocksDBStorage) -> None:
    """A decision from a scanned range is not skewed by a concurrent insert into it."""
    barrier = threading.Barrier(2)
    violations = 0
    for r in range(ROUNDS):
        prefix = ("seat", r)
        barrier.reset()
        options = StorageScanOptions(start=prefix, break_filter=PrefixFilter(prefix=prefix))

        def take(me, prefix=prefix, options=options):
            def body() -> None:
                with storage.transaction() as t:
                    taken = list(t.scan(options).keys())
                    _wait(barrier)
                    if not taken:
                        t.put((*prefix, me), True)

            return lambda: _retrying(body)

        _threads(take("a"), take("b"))
        with storage.snapshot() as s:
            if len(list(s.scan(options).keys())) > 1:
                violations += 1
    assert violations == 0


def test_read_your_own_writes(storage: RocksDBStorage) -> None:
    _put(storage, {("r", 1): "old", ("r", 2): "gone"})
    options = StorageScanOptions(start=("r",), break_filter=PrefixFilter(prefix=("r",)))
    with storage.transaction() as t:
        t.put(("r", 1), "new")
        t.put(("r", 3), "added")
        t.delete(("r", 2))
        assert t.get(("r", 1)) == "new"
        assert t.get(("r", 3)) == "added"
        assert t.get(("r", 2)) is EMPTY
        assert not t.exists(("r", 2))
        assert dict(t.scan(options).items()) == {("r", 1): "new", ("r", 3): "added"}
        assert list(t.scan(StorageScanOptions(start=("r", 9), reverse=True)).keys())[:2] == [
            ("r", 3),
            ("r", 1),
        ]
        t.abort()
    with storage.snapshot() as s:
        assert dict(s.scan(options).items()) == {("r", 1): "old", ("r", 2): "gone"}


def test_transaction_reads_are_pinned_while_it_runs(storage: RocksDBStorage) -> None:
    """A read-only transaction is never torn: it holds the slot, writers wait."""
    _put(storage, {X: 1, Y: -1})
    t = storage.begin_transaction()
    try:
        first = t.get(X)
        done = threading.Event()

        def writer() -> None:
            _put(storage, {X: 2, Y: -2})  # blocks on the slot until t ends
            done.set()

        th = threading.Thread(target=writer, daemon=True)
        th.start()
        time.sleep(0.05)
        assert not done.is_set(), "writer committed under an open transaction"
        assert t.get(X) == first
        assert t.get(X) + t.get(Y) == 0
    finally:
        t.abort()
    assert done.wait(5.0)
    assert _read(storage, X) == 2


def test_nested_transaction_raises_distinct_error(storage: RocksDBStorage) -> None:
    assert not issubclass(StorageNestedTransactionError, RETRYABLE)
    outer = storage.begin_transaction()
    try:
        outer.put(K, 1)
        t0 = time.monotonic()
        with pytest.raises(StorageNestedTransactionError):
            storage.begin_transaction()
        with pytest.raises(StorageNestedTransactionError):
            with storage.batch_write() as batch:
                batch.put(K, 2)
        assert time.monotonic() - t0 < 0.5
    finally:
        outer.abort()
    storage.begin_transaction().commit()  # the slot came back with the abort
    assert _read(storage, K) is EMPTY


def test_nested_transaction_in_same_task_raises(storage: RocksDBStorage) -> None:
    async def main() -> None:
        slot = await storage.areserve_write_slot()
        tx = storage.begin_transaction(write_slot=slot)
        with pytest.raises(StorageNestedTransactionError):
            await storage.areserve_write_slot()
        with pytest.raises(StorageNestedTransactionError):
            storage.begin_transaction()
        tx.abort()
        storage.begin_transaction().commit()

    asyncio.run(main())


# =============================================================================
# Section 2: snapshots
# =============================================================================


def test_snapshot_repeatable_read(storage: RocksDBStorage) -> None:
    _put(storage, {X: 0})
    for i in range(1, ROUNDS + 1):
        with storage.snapshot() as s:
            first = s.get(X)
            _threads(lambda i=i: _put(storage, {X: i}))
            assert s.get(X) == first
            assert dict(s.scan(StorageScanOptions()).items())[X] == first
        assert _read(storage, X) == i


def test_snapshot_never_torn(storage: RocksDBStorage) -> None:
    """The writer keeps x + y == 0; a snapshot reads x, a commit lands, it reads y."""
    _put(storage, {X: 0, Y: 0})
    torn = []
    for i in range(1, ROUNDS + 1):
        with storage.snapshot() as s:
            x = s.get(X)
            _threads(lambda i=i: _put(storage, {X: i, Y: -i}))
            y = s.get(Y)
        if x + y != 0:
            torn.append((x, y))
    assert not torn


def test_snapshot_not_torn_under_continuous_writer(storage: RocksDBStorage) -> None:
    _put(storage, {X: 0, Y: 0})
    stop = threading.Event()

    def writer() -> None:
        i = 0
        while not stop.is_set():
            i += 1
            _put(storage, {X: i, Y: -i})

    th = threading.Thread(target=writer, daemon=True)
    th.start()
    reads = torn = 0
    end = time.monotonic() + 0.3
    try:
        while time.monotonic() < end:
            with storage.snapshot() as s:
                a = s.get(X)
                time.sleep(0.0002)
                b = s.get(Y)
            reads += 1
            torn += a + b != 0
    finally:
        stop.set()
        th.join(5.0)
    assert reads > 10
    assert torn == 0


def test_snapshot_never_waits_on_writer(storage: RocksDBStorage) -> None:
    _put(storage, {X: 0})
    wrote, release = threading.Event(), threading.Event()

    def writer() -> None:
        t = storage.begin_transaction()
        try:
            t.put(X, 1)
            wrote.set()
            release.wait(5.0)
        finally:
            t.abort()

    th = threading.Thread(target=writer, daemon=True)
    th.start()
    try:
        assert wrote.wait(5.0)
        t0 = time.monotonic()
        with storage.snapshot() as s:
            seen = s.get(X)
        elapsed = time.monotonic() - t0
    finally:
        release.set()
        th.join(5.0)
    assert seen == 0
    assert elapsed < 0.1


def test_read_only_open_serves_snapshots(tmp_path: Path) -> None:
    with RocksDBStorage(path=tmp_path / "db", codec=BinaryCodec()) as rw:
        _put(rw, {X: 1})
    ro = RocksDBStorage(path=tmp_path / "db", codec=BinaryCodec(), read_only=True)
    ro.open()
    try:
        assert _read(ro, X) == 1
        with pytest.raises(StorageError):
            ro.begin_transaction()
        with pytest.raises(StorageError):
            ro.begin_write_batch()
    finally:
        ro.close()


# =============================================================================
# Section 3: errors and retry
# =============================================================================


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (rdbpy.Busy("busy", 11, 0), StorageTransactionConflictError),
        (rdbpy.Deadlock("deadlock", 11, 5), StorageTransactionConflictError),
        (rdbpy.TryAgain("history", 13, 0), StorageTransactionConflictError),
        (rdbpy.Expired("expired", 12, 0), StorageTransactionConflictError),
        (rdbpy.TimedOut("lock", 9, 2), StorageLockTimeoutError),
        (rdbpy.LockLimit("locks", 11, 3), StorageLockTimeoutError),
        (rdbpy.Corruption("bad", 2, 0), StorageOperationError),
        (RuntimeError("iterator is closed or its parent has ended"), StorageClosedError),
    ],
)
def test_rdbpy_errors_map_by_class(error: Exception, expected: type) -> None:
    mapped = classify_rdbpy(error, "op")
    assert type(mapped) is expected
    assert isinstance(mapped, RETRYABLE) == issubclass(expected, RETRYABLE)


def test_row_lock_held_outside_the_slot_is_a_lock_timeout(tmp_path: Path) -> None:
    """A writer that bypasses the slot (a raw rdbpy txn) surfaces as retryable, fast."""
    s = RocksDBStorage(
        path=tmp_path / "db",
        codec=BinaryCodec(),
        txn_db_options={"transaction_lock_timeout": 20, "default_lock_timeout": 20},
    )
    s.open()
    try:
        raw = s._db.begin()
        raw.put(s.codec.encode_key(K), b"raw")
        t0 = time.monotonic()
        with pytest.raises(StorageLockTimeoutError), s.transaction() as t:
            t.put(K, 1)
        assert time.monotonic() - t0 < 1.0
        raw.rollback()
        _put(s, {K: 1})
    finally:
        s.close()


def test_key_moved_after_snapshot_is_a_conflict(storage: RocksDBStorage) -> None:
    """A write that bypasses the slot after the txn snapshot surfaces as a conflict."""
    _put(storage, {K: 0})
    with pytest.raises(StorageTransactionConflictError), storage.transaction() as t:
        assert t.get(K) == 0
        storage._db.put(storage.codec.encode_key(K), b"\x00")  # behind the slot's back
        t.put(K, 1)
    _put(storage, {K: 2})
    assert _read(storage, K) == 2


def test_lock_timeout_between_threads(tmp_path: Path) -> None:
    s = RocksDBStorage(path=tmp_path / "db", codec=BinaryCodec(), busy_timeout=0.05)
    s.open()
    try:
        holding, release = threading.Event(), threading.Event()

        def holder() -> None:
            with s.transaction() as t:
                t.put(K, 1)
                holding.set()
                release.wait(5.0)

        th = threading.Thread(target=holder, daemon=True)
        th.start()
        assert holding.wait(5.0)
        t0 = time.monotonic()
        with pytest.raises(StorageLockTimeoutError):
            s.begin_transaction()
        with pytest.raises(StorageLockTimeoutError):
            with s.batch_write() as b:
                b.put(K, 2)
        assert 0.05 <= time.monotonic() - t0 < 1.0
        release.set()
        th.join(5.0)
        assert _read(s, K) == 1
    finally:
        s.close()


def test_closed_storage_error_is_not_retryable(tmp_path: Path) -> None:
    s = RocksDBStorage(path=tmp_path / "db", codec=BinaryCodec())
    s.open()
    s.close()
    with pytest.raises(StorageError) as info:
        with s.transaction() as t:
            t.put(K, 1)
    assert not isinstance(info.value, RETRYABLE)


def test_close_refuses_while_a_thread_is_inside_a_call(tmp_path: Path) -> None:
    """close() during another thread's long native call waits busy_timeout, then raises and frees nothing.

    It used to abort that thread's transaction under it (a segfault). The
    call here is a row lock wait longer than ``busy_timeout``, so the
    retried close never lands and the storage stays open and usable.
    """
    s = RocksDBStorage(
        path=tmp_path / "db",
        codec=BinaryCodec(),
        busy_timeout=0.3,
        txn_db_options={"transaction_lock_timeout": 1000, "default_lock_timeout": 1000},
    )
    s.open()
    raw = s._db.begin()
    raw.put(s.codec.encode_key(K), b"raw")  # row lock outside the slot
    entered = threading.Event()
    outcome: list = []

    def worker() -> None:
        t = s.begin_transaction()
        entered.set()
        try:
            t.put(K, 1)  # waits nogil on the row lock
        except RETRYABLE as e:
            outcome.append(e)
        t.abort()

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    entered.wait(5)
    time.sleep(0.2)
    t0 = time.perf_counter()
    with pytest.raises(StorageError, match="in use"):
        s.close()
    assert 0.25 < time.perf_counter() - t0 < 1.0  # waited busy_timeout, not the whole row lock wait
    th.join(5)
    assert outcome and isinstance(outcome[0], StorageLockTimeoutError)
    raw.rollback()
    _put(s, {K: 2})
    assert _read(s, K) == 2
    s.close()


def test_close_wakes_slot_waiter_with_closed_error(tmp_path: Path) -> None:
    """A sync writer queued on the slot sees a closed storage once close() ends."""
    s = RocksDBStorage(path=tmp_path / "db", codec=BinaryCodec(), busy_timeout=5.0)
    s.open()
    held = s.begin_transaction()
    errors: list[BaseException] = []

    def waiter() -> None:
        try:
            s.begin_transaction()
        except BaseException as e:
            errors.append(e)

    th = threading.Thread(target=waiter, daemon=True)
    th.start()
    time.sleep(0.05)
    s.close()
    th.join(5)
    assert not th.is_alive()
    assert held.is_closed
    assert errors and isinstance(errors[0], StorageClosedError)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
def test_close_in_forked_child_frees_nothing(tmp_path: Path) -> None:
    """A forked child's close() forgets the parent's handle instead of tearing it down.

    Tearing it down in the child hung (no-WAL memtable flush waits on a
    background thread only the parent has).
    """
    s = RocksDBStorage(path=tmp_path / "db", codec=BinaryCodec(), disable_wal=True)
    s.open()
    _put(s, {K: 1})
    txn = s.begin_transaction()
    snap = s.begin_snapshot()
    before = sorted(p.name for p in (tmp_path / "db").iterdir())
    pid = os.fork()
    if pid == 0:  # pragma: no cover - child
        code = 1
        try:
            s.close()
            code = 0 if txn.is_closed and snap.is_closed and not s._opened else 2
        finally:
            os._exit(code)
    deadline = time.monotonic() + 10
    while (done := os.waitpid(pid, os.WNOHANG))[0] == 0 and time.monotonic() < deadline:
        time.sleep(0.02)
    if done[0] == 0:
        os.kill(pid, 9)
        os.waitpid(pid, 0)
        pytest.fail("close hung in the forked child")
    assert os.waitstatus_to_exitcode(done[1]) == 0
    assert sorted(p.name for p in (tmp_path / "db").iterdir()) == before
    assert snap.get(K) == 1
    txn.put(K, 2)
    txn.commit()
    snap.close()
    assert _read(s, K) == 2
    s.close()


def test_write_batch_queues_behind_transaction(storage: RocksDBStorage) -> None:
    """A batch never lands inside a transaction: it waits for the slot."""
    _put(storage, {K: 0})
    order: list[str] = []
    holding = threading.Event()

    def holder() -> None:
        with storage.transaction() as t:
            holding.set()
            v = t.get(K)
            time.sleep(0.1)
            t.put(K, v + 1)
            order.append("txn")

    th = threading.Thread(target=holder, daemon=True)
    th.start()
    assert holding.wait(5.0)
    with storage.batch_write() as b:
        b.put(K, 100)
    order.append("batch")
    th.join(5.0)
    assert order == ["txn", "batch"]
    assert _read(storage, K) == 100


# =============================================================================
# Section 4: coroutines and the event loop
# =============================================================================


async def _increment(storage: RocksDBStorage, key: tuple, pause: float) -> None:
    slot = await storage.areserve_write_slot()
    try:
        tx = storage.begin_transaction(write_slot=slot)
        try:
            cur = tx.get(key)
            tx.put(key, (cur if cur is not EMPTY else 0) + 1)
            await asyncio.sleep(pause)
        except BaseException:
            tx.abort()
            raise
        tx.commit()
    finally:
        slot.release()


def test_concurrent_incs_do_not_stall_the_loop(storage: RocksDBStorage) -> None:
    """10 coroutines inc one key with an await inside the txn: all land, fast, loop alive."""
    n, step, tick = 10, 0.005, 0.002

    async def main() -> tuple[float, float]:
        stop = asyncio.Event()
        last = time.perf_counter()
        gap = 0.0

        async def ticker() -> None:
            nonlocal last, gap
            while not stop.is_set():
                await asyncio.sleep(tick)
                now = time.perf_counter()
                gap = max(gap, now - last)
                last = now

        t = asyncio.create_task(ticker())
        t0 = time.perf_counter()
        await asyncio.gather(*(_increment(storage, K, step) for _ in range(n)))
        wall = time.perf_counter() - t0
        stop.set()
        await t
        return wall, gap

    wall, gap = asyncio.run(main())
    assert _read(storage, K) == n
    assert wall < 1.0, f"{n} incs took {wall:.2f}s"
    assert gap < 0.1, f"event loop blocked for {gap * 1000:.0f}ms"


def test_coroutine_holding_slot_blocks_peer_until_commit(storage: RocksDBStorage) -> None:
    order: list[str] = []

    async def a() -> None:
        slot = await storage.areserve_write_slot()
        tx = storage.begin_transaction(write_slot=slot)
        tx.put(("a",), 1)
        order.append("a holds")
        await asyncio.sleep(0.05)
        order.append("a commits")
        tx.commit()
        slot.release()  # no-op: the transaction owned the slot

    async def b() -> None:
        await asyncio.sleep(0.01)
        slot = await storage.areserve_write_slot()
        order.append("b holds")
        with storage.begin_transaction(write_slot=slot) as tx:
            assert tx.get(("a",)) == 1
            tx.put(("b",), 2)

    async def main() -> None:
        await asyncio.gather(a(), b())

    asyncio.run(main())
    assert order == ["a holds", "a commits", "b holds"]
    assert _read(storage, ("b",)) == 2


def test_sync_begin_behind_loop_peer_fails_fast(storage: RocksDBStorage) -> None:
    async def holder(ready: asyncio.Event, done: asyncio.Event) -> None:
        slot = await storage.areserve_write_slot()
        ready.set()
        await done.wait()
        slot.release()

    async def main() -> None:
        ready, done = asyncio.Event(), asyncio.Event()
        task = asyncio.create_task(holder(ready, done))
        await ready.wait()
        t0 = time.monotonic()
        with pytest.raises(StorageLockTimeoutError, match="areserve_write_slot"):
            storage.begin_transaction()
        assert time.monotonic() - t0 < 0.5
        done.set()
        await task

    asyncio.run(main())
    storage.begin_transaction().commit()


def test_unused_slot_is_released(storage: RocksDBStorage) -> None:
    async def main() -> None:
        slot = await storage.areserve_write_slot()
        assert slot.held
        slot.release()
        slot.release()
        assert not slot.held
        with pytest.raises(StorageError):
            storage.begin_transaction(write_slot=slot)

    asyncio.run(main())
    storage.begin_transaction().commit()


def test_failed_begin_returns_slot_to_holder(storage: RocksDBStorage) -> None:
    async def main() -> None:
        slot = await storage.areserve_write_slot()
        db = storage._db

        class Broken:
            def __getattr__(self, name: str) -> Any:
                return getattr(db, name)

            def begin(self, **kwargs: Any) -> Any:
                raise rdbpy.RocksIOError("no begin", 5, 0)

        storage._db = Broken()
        try:
            with pytest.raises(StorageError, match="no begin"):
                storage.begin_transaction(write_slot=slot)
        finally:
            storage._db = db
        assert slot.held
        storage.begin_transaction(write_slot=slot).commit()
        assert not slot.held

    asyncio.run(main())
    storage.begin_transaction().commit()


def test_async_waiter_times_out_without_blocking_loop(tmp_path: Path) -> None:
    with RocksDBStorage(path=tmp_path / "db", codec=BinaryCodec(), busy_timeout=0.2) as s:

        async def main() -> int:
            held = await s.areserve_write_slot()
            ticks = 0
            stop = asyncio.Event()

            async def ticker() -> None:
                nonlocal ticks
                while not stop.is_set():
                    ticks += 1
                    await asyncio.sleep(0.005)

            tick_task = asyncio.create_task(ticker())
            t0 = time.monotonic()

            async def waiter() -> None:
                await s.areserve_write_slot()

            with pytest.raises(StorageLockTimeoutError, match="Timed out"):
                await asyncio.create_task(waiter())
            elapsed = time.monotonic() - t0
            stop.set()
            await tick_task
            held.release()
            assert 0.15 < elapsed < 1.0
            return ticks

        assert asyncio.run(main()) >= 10


# =============================================================================
# Options compatibility
# =============================================================================


def test_option_dicts_translate(tmp_path: Path) -> None:
    s = RocksDBStorage(
        path=tmp_path / "db",
        codec=BinaryCodec(),
        options={"block_cache_mb": 4, "block_table": {"cache_index_and_filter_blocks": True}},
        txn_db_options={"lock_timeout_ms": 10, "max_num_deadlocks": 2},
        txn_options={"deadlock_detect": True, "lock_timeout": 10},
    )
    s.open()
    try:
        _put(s, {K: 1})
        assert _read(s, K) == 1
    finally:
        s.close()


def test_unknown_option_is_a_storage_error(tmp_path: Path) -> None:
    s = RocksDBStorage(path=tmp_path / "db", codec=BinaryCodec(), options={"nope": 1})
    with pytest.raises(StorageError, match="options"):
        s.open()
    s = RocksDBStorage(path=tmp_path / "db2", codec=BinaryCodec(), txn_db_options={"nope": 1})
    with pytest.raises(StorageError, match="options"):
        s.open()
