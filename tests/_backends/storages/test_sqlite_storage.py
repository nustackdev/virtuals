"""SQLite-specific behaviour beyond the shared StorageProtocol compliance suite.

Covers what compliance cannot: byte ordering against LMDB, paged scans past
a page boundary, snapshot isolation, the writer lock across threads,
coroutines and processes (including a writer killed mid-transaction),
read-only opens and reopening after close.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import gc
import os
import random
import signal
import subprocess
import sys
import textwrap
import threading
import time
from typing import TYPE_CHECKING

import pytest

from tests._backends.storages import _write_slot_cases as cases
from virtuals._backends.storages.sqlite import SQLiteStorage, SQLiteWriteSlot
from virtuals.codecs import BinaryCodec
from virtuals.tkv import StorageScanOptions
from virtuals.tkv.filter import LengthFilter, PrefixFilter
from virtuals.tkv.storage import (
    StorageClosedError,
    StorageError,
    StorageLockTimeoutError,
    StorageNestedTransactionError,
)
from virtuals.tkv.types import EMPTY


if TYPE_CHECKING:
    import sqlite3
    from pathlib import Path


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "kv.sqlite"


@pytest.fixture
def storage(db_path: Path):
    s = SQLiteStorage(path=db_path, codec=BinaryCodec(), busy_timeout=5.0)
    s.open()
    yield s
    s.close()


def _keys(n_top: int = 20, n_mid: int = 12, n_leaf: int = 12) -> list[tuple]:
    """A tree of mixed int/str keys, with containers at every level."""
    rng = random.Random(57)  # noqa: S311
    keys: set[tuple] = set()
    for i in range(n_top):
        top = rng.choice([i, -i - 1, f"t{i}", f"t{i}" * 3, "zz" * (i + 1), 2**40 + i])
        keys.add((top,))
        for j in range(n_mid):
            mid = rng.choice([j, f"m{j}", -j * 1000, "a" * (j + 1)])
            keys.add((top, mid))
            for k in range(n_leaf):
                keys.add((top, mid, rng.choice([k, f"l{k}", -(2**33) - k])))
    return sorted(keys, key=str)


def _fill(storage, keys: list[tuple]) -> None:
    with storage.transaction() as tx:
        for i, k in enumerate(keys):
            tx.put(k, f"v{i}".encode())


def _scan(storage, options: StorageScanOptions) -> list:
    with storage.snapshot() as snap:
        return list(snap.scan(options).items())


# =========================================================================
# Ordering and scans
# =========================================================================


def test_schema_and_pragmas(storage: SQLiteStorage, db_path: Path) -> None:
    conn = storage._acquire_connection()
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1  # NORMAL
        sql = conn.execute("SELECT sql FROM sqlite_master WHERE name = 'kv'").fetchone()[0]
        assert "WITHOUT ROWID" in sql
    finally:
        storage._release_connection(conn)

    full = SQLiteStorage(path=db_path, codec=BinaryCodec(), synchronous="FULL")
    with full:
        conn = full._acquire_connection()
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 2
        full._release_connection(conn)


def test_scan_order_is_encoded_byte_order(storage: SQLiteStorage) -> None:
    keys = _keys()
    _fill(storage, keys)
    codec = storage.codec

    got = [k for k, _ in _scan(storage, StorageScanOptions())]
    assert got == sorted(keys, key=codec.encode_key)

    got_rev = [k for k, _ in _scan(storage, StorageScanOptions(reverse=True))]
    assert got_rev == got[::-1]


def _option_grid(keys: list[tuple], codec: BinaryCodec) -> list[StorageScanOptions]:
    rng = random.Random(3)  # noqa: S311
    grid: list[StorageScanOptions] = [StorageScanOptions(), StorageScanOptions(reverse=True)]
    containers = [k for k in keys if len(k) < 3]
    for site in rng.sample(containers, 25):
        prefix = PrefixFilter(prefix=site)
        child = prefix & LengthFilter(length=len(site) + 1)
        grid += [
            StorageScanOptions(start=site, break_filter=prefix),
            StorageScanOptions(start=site, break_filter=prefix, filter=child),
            StorageScanOptions(start=site, break_filter=prefix, filter=child, limit=3),
            StorageScanOptions(start=site, break_filter=prefix, limit=1),
            StorageScanOptions(
                start_encoded=codec.upper_bound_of_prefix(site),
                reverse=True,
                break_filter=prefix,
                filter=child,
            ),
            StorageScanOptions(
                start_encoded=codec.upper_bound_of_prefix(site), reverse=True, break_filter=prefix
            ),
            StorageScanOptions(start=site, reverse=True, limit=40),
            StorageScanOptions(start=site, limit=100),
            StorageScanOptions(break_filter=prefix),
            StorageScanOptions(reverse=True, break_filter=prefix),
            StorageScanOptions(
                start=site, break_filter=prefix, filter=child, max_depth=len(site) + 1
            ),
            StorageScanOptions(
                start_encoded=codec.upper_bound_of_prefix(site),
                reverse=True,
                break_filter=prefix,
                filter=child,
                max_depth=len(site) + 1,
            ),
            StorageScanOptions(start=site, break_filter=prefix, max_depth=2, limit=5),
        ]
    return grid


def test_scans_match_lmdb(storage: SQLiteStorage, tmp_path: Path) -> None:
    """Every scan shape the views use returns exactly what LMDB returns."""
    pytest.importorskip("lmdb")
    from virtuals._backends.storages.lmdb import LMDBStorage

    keys = _keys()
    assert len(keys) > 2 * 1024  # crosses several scan pages
    _fill(storage, keys)
    with LMDBStorage(path=tmp_path / "ref.lmdb", codec=BinaryCodec(), map_size=2**26) as ref:
        _fill(ref, keys)
        for options in _option_grid(keys, storage.codec):
            assert _scan(storage, options) == _scan(ref, options), options


def test_keys_and_values_iterators(storage: SQLiteStorage) -> None:
    _fill(storage, [("a", i) for i in range(100)])
    with storage.snapshot() as snap:
        opts = StorageScanOptions(start=("a",), break_filter=PrefixFilter(prefix=("a",)))
        assert list(snap.scan(opts).keys()) == [("a", i) for i in range(100)]
        assert list(snap.scan(opts).values()) == [f"v{i}".encode() for i in range(100)]


def test_delete_while_iterating(storage: SQLiteStorage) -> None:
    """Paged scans hold no statement open, so writing mid-scan is safe."""
    _fill(storage, [("x", i) for i in range(3000)] + [("y", 0)])
    opts = StorageScanOptions(start=("x",), break_filter=PrefixFilter(prefix=("x",)))
    with storage.transaction() as tx:
        n = 0
        for key in tx.scan(opts).keys():
            tx.delete(key)
            n += 1
        assert n == 3000
        assert list(tx.scan(opts).keys()) == []
    with storage.snapshot() as snap:
        assert snap.get(("y", 0)) == b"v3000"
        assert snap.get(("x", 5)) is EMPTY


def _count_pages(storage: SQLiteStorage, options: StorageScanOptions) -> tuple[int, int]:
    """Run a scan, return (keys yielded, pages queried)."""
    from virtuals._backends.storages.sqlite.scan import SQLiteScan

    fetch = SQLiteScan._fetch_page
    pages = 0

    def counted(self, *args, **kwargs):
        nonlocal pages
        pages += 1
        return fetch(self, *args, **kwargs)

    SQLiteScan._fetch_page = counted  # type: ignore[method-assign]
    try:
        with storage.snapshot() as snap:
            n = sum(1 for _ in snap.scan(options).keys())
    finally:
        SQLiteScan._fetch_page = fetch  # type: ignore[method-assign]
    return n, pages


@pytest.mark.parametrize("reverse", [False, True])
def test_max_depth_skip_stays_in_page(storage: SQLiteStorage, reverse: bool) -> None:
    """A skip over a small subtree moves on within the fetched page, no re-query."""
    children = 1000
    keys = [("d",)]
    for i in range(children):
        keys += [("d", i), ("d", i, "x"), ("d", i, "y")]
    _fill(storage, keys)
    prefix = PrefixFilter(prefix=("d",))
    if reverse:
        options = StorageScanOptions(
            start_encoded=storage.codec.upper_bound_of_prefix(("d",)),
            reverse=True,
            break_filter=prefix,
            filter=prefix & LengthFilter(length=2),
            max_depth=2,
        )
    else:
        options = StorageScanOptions(
            start=("d",),
            break_filter=prefix,
            filter=prefix & LengthFilter(length=2),
            max_depth=2,
        )

    n, pages = _count_pages(storage, options)
    assert n == children
    # 3001 rows in pages that double up to 1024: a handful of queries, not one per child
    assert pages <= 12, pages


def test_max_depth_wide_subtree_reseeks_small(storage: SQLiteStorage) -> None:
    """A subtree wider than a page is left with one small query per child."""
    children, width = 20, 2000
    keys = [("d",)]
    for i in range(children):
        keys += [("d", i)] + [("d", i, j) for j in range(width)]
    _fill(storage, keys)
    prefix = PrefixFilter(prefix=("d",))
    options = StorageScanOptions(
        start=("d",), break_filter=prefix, filter=prefix & LengthFilter(length=2), max_depth=2
    )

    n, pages = _count_pages(storage, options)
    assert n == children
    assert pages <= children + 2, pages


# =========================================================================
# Snapshots
# =========================================================================


def test_snapshot_isolation(storage: SQLiteStorage) -> None:
    with storage.transaction() as tx:
        tx.put(("k",), b"old")

    snap = storage.begin_snapshot()  # pinned now, before any read
    with storage.transaction() as tx:
        tx.put(("k",), b"new")
        tx.put(("k2",), b"added")

    assert snap.get(("k",)) == b"old"
    assert snap.get(("k2",)) is EMPTY
    assert [k for k, _ in snap.scan(StorageScanOptions()).items()] == [("k",)]
    snap.close()

    with storage.snapshot() as fresh:
        assert fresh.get(("k",)) == b"new"


def test_snapshot_does_not_wait_for_writer(storage: SQLiteStorage) -> None:
    """A held write lock, and a writer queued behind it, never delay snapshots."""
    tx = storage.begin_transaction()
    tx.put(("k",), b"pending")

    queued = threading.Thread(target=lambda: storage.begin_transaction().commit())
    queued.start()
    time.sleep(0.05)  # queued writer is now waiting for the lock

    t0 = time.monotonic()
    with storage.snapshot() as snap:
        assert snap.get(("k",)) is EMPTY
    assert time.monotonic() - t0 < 0.5

    tx.commit()
    queued.join(5)
    assert not queued.is_alive()


# =========================================================================
# Writers: threads
# =========================================================================


def test_same_thread_second_writer_fails_fast(storage: SQLiteStorage) -> None:
    tx = storage.begin_transaction()
    t0 = time.monotonic()
    with pytest.raises(StorageNestedTransactionError):
        storage.begin_transaction()
    assert time.monotonic() - t0 < 0.5
    tx.abort()
    storage.begin_transaction().commit()  # slot released by the abort


def test_threads_serialize_writers(storage: SQLiteStorage) -> None:
    n_threads, rounds = 8, 50

    def work() -> None:
        for _ in range(rounds):
            with storage.transaction() as tx:
                cur = tx.get(("n",))
                tx.put(("n",), str(int(cur) + 1 if cur is not EMPTY else 1).encode())

    threads = [threading.Thread(target=work) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    with storage.snapshot() as snap:
        assert int(snap.get(("n",))) == n_threads * rounds


def test_transaction_can_finish_on_another_thread(storage: SQLiteStorage) -> None:
    tx = storage.begin_transaction()
    tx.put(("k",), b"v")
    t = threading.Thread(target=tx.commit)
    t.start()
    t.join(5)
    with storage.snapshot() as snap:
        assert snap.get(("k",)) == b"v"
    storage.begin_transaction().abort()  # slot released across threads


def test_lock_timeout_between_threads(db_path: Path) -> None:
    with SQLiteStorage(path=db_path, codec=BinaryCodec(), busy_timeout=0.1) as s:
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


# =========================================================================
# Writers: coroutines on one event loop
# =========================================================================


async def _increment(storage: SQLiteStorage, key: tuple, pause: float) -> None:
    """Read-modify-write ``key`` under a reserved slot, awaiting mid-transaction."""
    slot = await storage.areserve_write_slot()
    try:
        tx = storage.begin_transaction(write_slot=slot)
        try:
            cur = tx.get(key)
            await asyncio.sleep(pause)
            tx.put(key, str(int(cur) + 1 if cur is not EMPTY else 1).encode())
        except BaseException:
            tx.abort()
            raise
        tx.commit()
    finally:
        slot.release()


def test_coroutines_wait_for_each_other(storage: SQLiteStorage) -> None:
    async def main() -> None:
        await asyncio.gather(*(_increment(storage, ("n",), 0.01) for _ in range(8)))

    asyncio.run(main())
    with storage.snapshot() as snap:
        assert int(snap.get(("n",))) == 8


def test_coroutine_holding_slot_across_await_blocks_peer_until_commit(
    storage: SQLiteStorage,
) -> None:
    order: list[str] = []

    async def a() -> None:
        slot = await storage.areserve_write_slot()
        tx = storage.begin_transaction(write_slot=slot)
        tx.put(("a",), b"1")
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
            assert tx.get(("a",)) == b"1"
            tx.put(("b",), b"2")

    async def main() -> None:
        await asyncio.gather(a(), b())

    asyncio.run(main())
    assert order == ["a holds", "a commits", "b holds"]
    with storage.snapshot() as snap:
        assert snap.get(("a",)) == b"1"
        assert snap.get(("b",)) == b"2"


def test_same_task_second_writer_fails_fast(storage: SQLiteStorage) -> None:
    async def main() -> None:
        slot = await storage.areserve_write_slot()
        tx = storage.begin_transaction(write_slot=slot)
        t0 = time.monotonic()
        with pytest.raises(StorageNestedTransactionError, match="wait on itself"):
            await storage.areserve_write_slot()
        with pytest.raises(StorageNestedTransactionError, match="wait on itself"):
            storage.begin_transaction()
        with pytest.raises(StorageNestedTransactionError, match="wait on itself"):
            with storage.batch_write() as batch:
                batch.put(("x",), b"1")
        assert time.monotonic() - t0 < 0.5
        tx.abort()
        storage.begin_transaction().commit()  # slot released by the abort

    asyncio.run(main())


def test_sync_begin_behind_loop_peer_fails_fast(storage: SQLiteStorage) -> None:
    """A sync begin on the loop thread never blocks behind a coroutine there."""

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


def test_unused_slot_is_released(storage: SQLiteStorage) -> None:
    async def main() -> None:
        slot = await storage.areserve_write_slot()
        assert slot.held
        slot.release()
        slot.release()  # idempotent
        assert not slot.held
        with pytest.raises(StorageError):
            storage.begin_transaction(write_slot=slot)

    asyncio.run(main())
    storage.begin_transaction().commit()


def test_failed_begin_returns_slot_to_holder(storage: SQLiteStorage) -> None:
    """A begin that fails after taking the slot hands it back to the holder.

    The slot normally carries a begun connection, so begin cannot fail on
    one; here the connection is taken away first (what close() does), so
    begin falls back to the pool, which is made to fail.
    """

    async def main() -> None:
        slot = await storage.areserve_write_slot()
        storage._release_connection(slot._detach_conn())
        real = storage._acquire_connection

        def broken() -> sqlite3.Connection:
            raise StorageError("no connection")

        storage._acquire_connection = broken  # type: ignore[method-assign]
        try:
            with pytest.raises(StorageError, match="no connection"):
                storage.begin_transaction(write_slot=slot)
        finally:
            storage._acquire_connection = real  # type: ignore[method-assign]
        assert slot.held
        storage.begin_transaction(write_slot=slot).commit()
        assert not slot.held

    asyncio.run(main())
    storage.begin_transaction().commit()


def test_async_waiter_times_out_without_blocking_loop(db_path: Path) -> None:
    with SQLiteStorage(path=db_path, codec=BinaryCodec(), busy_timeout=0.2) as s:

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

            async def waiter() -> None:
                await s.areserve_write_slot()

            t0 = time.monotonic()
            with pytest.raises(StorageLockTimeoutError, match="Timed out"):
                await asyncio.create_task(waiter())
            elapsed = time.monotonic() - t0
            stop.set()
            await tick_task
            held.release()
            assert 0.15 < elapsed < 1.0
            return ticks

        assert asyncio.run(main()) >= 10


def test_async_waiter_cancelled_holds_nothing(storage: SQLiteStorage) -> None:
    async def main() -> None:
        held = await storage.areserve_write_slot()

        async def waiter() -> None:
            await storage.areserve_write_slot()

        task = asyncio.create_task(waiter())
        await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        held.release()
        (await storage.areserve_write_slot()).release()

    asyncio.run(main())


def test_coroutines_and_threads_share_the_slot(storage: SQLiteStorage) -> None:
    n_threads, n_coros, rounds = 4, 4, 20

    def thread_work() -> None:
        for _ in range(rounds):
            with storage.transaction() as tx:
                cur = tx.get(("n",))
                time.sleep(0.0005)
                tx.put(("n",), str(int(cur) + 1 if cur is not EMPTY else 1).encode())

    async def coro_work() -> None:
        for _ in range(rounds):
            await _increment(storage, ("n",), 0.0005)

    async def main() -> None:
        loop = asyncio.get_running_loop()
        await asyncio.gather(
            *(loop.run_in_executor(None, thread_work) for _ in range(n_threads)),
            *(coro_work() for _ in range(n_coros)),
        )

    asyncio.run(main())
    with storage.snapshot() as snap:
        assert int(snap.get(("n",))) == (n_threads + n_coros) * rounds


def test_slot_taken_on_loop_finished_on_thread(storage: SQLiteStorage) -> None:
    async def main() -> None:
        slot = await storage.areserve_write_slot()

        def work() -> None:
            with storage.begin_transaction(write_slot=slot) as tx:
                tx.put(("k",), b"v")

        await asyncio.to_thread(work)
        slot.release()

    asyncio.run(main())
    with storage.snapshot() as snap:
        assert snap.get(("k",)) == b"v"
    storage.begin_transaction().commit()


# =========================================================================
# Write batches
# =========================================================================


def test_write_batch_holds_no_lock_until_write(storage: SQLiteStorage) -> None:
    with storage.transaction() as tx:
        tx.put(("gone",), b"x")
    batch = storage.begin_write_batch()
    batch.put(("a",), b"1")
    batch.put(("a",), b"2")  # last write wins
    batch.delete(("gone",))

    with storage.transaction() as tx:  # not blocked by the open batch
        tx.put(("b",), b"3")

    batch.write()
    with storage.snapshot() as snap:
        assert snap.get(("a",)) == b"2"
        assert snap.get(("b",)) == b"3"
        assert snap.get(("gone",)) is EMPTY


# =========================================================================
# Lifecycle and read-only
# =========================================================================


def test_close_aborts_open_contexts_and_reopens(db_path: Path) -> None:
    s = SQLiteStorage(path=db_path, codec=BinaryCodec())
    s.open()
    with s.transaction() as tx:
        tx.put(("kept",), b"1")
    tx = s.begin_transaction()
    tx.put(("lost",), b"1")
    snap = s.begin_snapshot()
    s.close()
    assert tx.is_closed and snap.is_closed

    s.open()
    with s.snapshot() as snap2:
        assert snap2.get(("kept",)) == b"1"
        assert snap2.get(("lost",)) is EMPTY
    s.begin_transaction().commit()  # writer slot was released by close
    s.close()


def test_read_only(db_path: Path) -> None:
    with pytest.raises(StorageError):
        SQLiteStorage(path=db_path, codec=BinaryCodec(), read_only=True).open()

    rw = SQLiteStorage(path=db_path, codec=BinaryCodec())
    rw.open()
    with rw.transaction() as tx:
        tx.put(("k",), b"1")

    with SQLiteStorage(path=db_path, codec=BinaryCodec(), read_only=True) as ro:
        with ro.snapshot() as snap:
            assert snap.get(("k",)) == b"1"
        with pytest.raises(StorageError):
            ro.begin_transaction()
        with pytest.raises(StorageError):
            ro.begin_write_batch()

        with rw.transaction() as tx:  # a live writer next to the reader
            tx.put(("k",), b"2")
        with ro.snapshot() as snap:
            assert snap.get(("k",)) == b"2"
    rw.close()

    # Reopening read-only once no writer is left (WAL checkpointed away).
    with SQLiteStorage(path=db_path, codec=BinaryCodec(), read_only=True) as ro:
        with ro.snapshot() as snap:
            assert snap.get(("k",)) == b"2"


# =========================================================================
# Writers: processes
# =========================================================================


_HOLDER = textwrap.dedent(
    """
    import sys, time
    from virtuals._backends.storages.sqlite import SQLiteStorage
    from virtuals.codecs import BinaryCodec

    s = SQLiteStorage(path=sys.argv[1], codec=BinaryCodec())
    s.open()
    tx = s.begin_transaction()
    tx.put(("uncommitted",), b"x")
    print("ready", flush=True)
    time.sleep(60)
    """
)


_HOLD_THEN_RELEASE = textwrap.dedent(
    """
    import sys, time
    from virtuals._backends.storages.sqlite import SQLiteStorage
    from virtuals.codecs import BinaryCodec

    s = SQLiteStorage(path=sys.argv[1], codec=BinaryCodec())
    s.open()
    tx = s.begin_transaction()
    tx.put(("held",), b"x")
    print("held", flush=True)
    time.sleep(float(sys.argv[2]))
    tx.commit()
    s.close()
    print("released", flush=True)
    """
)


def _hold_in_other_process(db_path: Path, hold: float) -> subprocess.Popen:
    """Start a process that holds the file write lock for ``hold`` seconds."""
    proc = subprocess.Popen(  # noqa: S603
        [sys.executable, "-c", _HOLD_THEN_RELEASE, str(db_path), str(hold)],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None
    assert proc.stdout.readline().strip() == "held"
    return proc


class _Ticker:
    """Longest gap between two turns of the event loop, measured by a task."""

    def __init__(self, tick: float = 0.005) -> None:
        self.tick = tick
        self.gap = 0.0
        self.ticks = 0
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None

    async def _run(self) -> None:
        last = time.perf_counter()
        while not self._stop.is_set():
            await asyncio.sleep(self.tick)
            now = time.perf_counter()
            self.gap = max(self.gap, now - last)
            last = now
            self.ticks += 1

    async def __aenter__(self) -> _Ticker:
        self._task = asyncio.create_task(self._run())
        await asyncio.sleep(0)
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._stop.set()
        assert self._task is not None
        await self._task


@pytest.mark.timeout(20)
def test_cross_process_lock_wait_does_not_block_loop(storage: SQLiteStorage, db_path: Path) -> None:
    """Another process holds the write lock: the slot waits for it off the loop."""
    hold = 0.5
    proc = _hold_in_other_process(db_path, hold)

    async def main() -> tuple[float, float, int]:
        try:
            async with _Ticker() as ticker:
                t0 = time.perf_counter()
                slot = await storage.areserve_write_slot()
                waited = time.perf_counter() - t0
                tx = storage.begin_transaction(write_slot=slot)
                tx.put(("loop",), b"1")
                tx.commit()
            return waited, ticker.gap, ticker.ticks
        finally:
            proc.wait(10)

    waited, gap, ticks = asyncio.run(main())
    assert waited >= hold * 0.6, f"got the lock after {waited:.3f}s, holder held it {hold}s"
    assert gap < 0.2, f"loop blocked for {gap * 1000:.0f}ms during a cross-process lock wait"
    assert ticks >= int(waited / 0.005 * 0.5), f"ticker ran {ticks} times in {waited:.3f}s"
    with storage.snapshot() as snap:
        assert snap.get(("loop",)) == b"1"
        assert snap.get(("held",)) == b"x"


@pytest.mark.timeout(20)
def test_cross_process_lock_wait_times_out_without_blocking_loop(db_path: Path) -> None:
    """The file lock wait is bounded by busy_timeout and surfaces as a lock timeout."""
    proc = _hold_in_other_process(db_path, 5.0)
    s = SQLiteStorage(path=db_path, codec=BinaryCodec(), busy_timeout=0.3)
    s.open()
    try:

        async def main() -> tuple[float, float]:
            async with _Ticker() as ticker:
                t0 = time.perf_counter()
                with pytest.raises(StorageLockTimeoutError, match="write lock"):
                    await s.areserve_write_slot()
                return time.perf_counter() - t0, ticker.gap

        elapsed, gap = asyncio.run(main())
        assert 0.25 < elapsed < 2.0
        assert gap < 0.2, f"loop blocked for {gap * 1000:.0f}ms"
        # Nothing is held: the slot is free and the pooled connection is idle.
        assert not s._writer.lock.locked()
        assert all(not c.in_transaction for c in s._idle)
    finally:
        proc.kill()
        proc.wait(10)
        s.close()


@pytest.mark.timeout(20)
def test_cancel_during_file_lock_wait_holds_nothing(storage: SQLiteStorage, db_path: Path) -> None:
    proc = _hold_in_other_process(db_path, 1.0)
    try:

        async def main() -> None:
            task = asyncio.create_task(storage.areserve_write_slot())
            await asyncio.sleep(0.1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not storage._writer.lock.locked()
            assert all(not c.in_transaction for c in storage._idle)

        asyncio.run(main())
    finally:
        proc.wait(10)
    storage.begin_transaction().commit()


_TRY_WRITE = textwrap.dedent(
    """
    import sys
    from virtuals._backends.storages.sqlite import SQLiteStorage
    from virtuals.codecs import BinaryCodec
    from virtuals.tkv.storage import StorageLockTimeoutError

    s = SQLiteStorage(path=sys.argv[1], codec=BinaryCodec(), busy_timeout=0.05)
    s.open()
    try:
        with s.transaction() as tx:
            tx.put(("other",), b"1")
        print("wrote", flush=True)
    except StorageLockTimeoutError:
        print("busy", flush=True)
    s.close()
    """
)


def _other_process_write(db_path: Path) -> str:
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _TRY_WRITE, str(db_path)],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return out.stdout.strip()


@pytest.mark.timeout(30)
def test_reserved_slot_holds_file_lock_until_released(
    storage: SQLiteStorage, db_path: Path
) -> None:
    """A reserved slot owns the file lock (other processes wait) and release gives it back."""

    async def main() -> None:
        slot = await storage.areserve_write_slot()
        assert _other_process_write(db_path) == "busy"
        slot.release()
        assert _other_process_write(db_path) == "wrote"

    asyncio.run(main())


@pytest.mark.timeout(30)
def test_close_takes_file_lock_back_from_reserved_slot(db_path: Path) -> None:
    """close() gives back a reserved slot's connection even while the slot object lives."""
    s = SQLiteStorage(path=db_path, codec=BinaryCodec(), busy_timeout=1.0)
    s.open()

    async def main() -> SQLiteWriteSlot:
        return await s.areserve_write_slot()

    slot = asyncio.run(main())
    s.close()
    assert _other_process_write(db_path) == "wrote"
    slot.release()  # no connection left to give back, no error
    s.open()
    s.begin_transaction().commit()
    s.close()


def test_killed_writer_releases_lock(db_path: Path) -> None:
    """A writer killed mid-transaction leaves no stuck lock and no partial write."""
    with SQLiteStorage(path=db_path, codec=BinaryCodec(), busy_timeout=0.2) as s:
        proc = subprocess.Popen(  # noqa: S603
            [sys.executable, "-c", _HOLDER, str(db_path)],
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            assert proc.stdout is not None
            assert proc.stdout.readline().strip() == "ready"

            with pytest.raises(StorageLockTimeoutError):
                s.begin_transaction()
            with s.snapshot() as snap:  # readers unaffected by the other writer
                assert snap.get(("uncommitted",)) is EMPTY
        finally:
            proc.kill()
            proc.wait(10)

        t0 = time.monotonic()
        with s.transaction() as tx:
            assert tx.get(("uncommitted",)) is EMPTY
            tx.put(("after",), b"1")
        assert time.monotonic() - t0 < 1.0


# =========================================================================
# close() against contexts in use on other threads
# =========================================================================


_CLOSE_BUSY = textwrap.dedent(
    """
    import json, sys, threading, time
    from virtuals._backends.storages.sqlite import SQLiteStorage
    from virtuals.codecs import BinaryCodec
    from virtuals.tkv import StorageScanOptions
    from virtuals.tkv.storage import StorageClosedError

    root, rounds, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    outcomes = {}
    for i in range(rounds):
        s = SQLiteStorage(path=f"{root}/db{i}.sqlite", codec=BinaryCodec(), busy_timeout=2.0)
        s.open()
        with s.transaction() as tx:
            for k in range(300):
                tx.put(("k", k), b"v" * 50)
        go = threading.Event()
        result = []

        def worker():
            ctx = s.begin_snapshot() if mode == "snap" else s.begin_transaction()
            go.set()
            try:
                for r in range(20000):
                    if mode == "txn":
                        ctx.put(("w", r), b"x")
                    ctx.get(("k", r % 300))
                    if r % 50 == 0:
                        list(ctx.scan(StorageScanOptions(limit=40)).keys())
                result.append("finished")
            except StorageClosedError:
                result.append("closed")
            except BaseException as e:
                result.append(type(e).__name__)

        th = threading.Thread(target=worker, daemon=True)
        th.start()
        go.wait()
        time.sleep(0.0005 * (i % 7))
        t0 = time.perf_counter()
        s.close()
        took = time.perf_counter() - t0
        th.join(10)
        key = (result[0] if result else "hung") + (" slow-close" if took > 1 else "")
        outcomes[key] = outcomes.get(key, 0) + 1
        s.open()
        s.begin_transaction().commit()
        s.close()
    print(json.dumps(outcomes))
    """
)


@pytest.mark.timeout(120)
@pytest.mark.parametrize("mode", ["txn", "snap"])
def test_close_during_in_flight_statements_never_crashes(tmp_path: Path, mode: str) -> None:
    """close() while another thread is inside a statement on a context: no crash, ever.

    Runs in a subprocess with faulthandler so a regression is a failed
    child, not a dead test run. The worker hammers get, put and scan on
    its context while the main thread closes the storage at varied
    offsets; close waits for the in-flight statement and the worker's
    next call sees the context closed.
    """
    rounds = 20
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-X", "faulthandler", "-c", _CLOSE_BUSY, str(tmp_path), str(rounds), mode],
        capture_output=True,
        text=True,
        timeout=100,
        check=False,
    )
    assert out.returncode == 0, f"child died (rc={out.returncode}):\n{out.stderr[-2000:]}"
    import json

    outcomes = json.loads(out.stdout.strip().splitlines()[-1])
    assert set(outcomes) <= {"finished", "closed"}, outcomes
    assert sum(outcomes.values()) == rounds


@pytest.mark.timeout(10)
def test_batch_closed_while_waiting_at_begin_writes_nothing(storage: SQLiteStorage) -> None:
    """close() aborts a batch blocked at begin behind a writer: its write raises, nothing lands."""
    holder = storage.begin_transaction()
    holder.put(("held",), b"1")
    batch = storage.begin_write_batch()
    batch.put(("batched",), b"1")
    errors: list[BaseException] = []
    started = threading.Event()

    def write() -> None:
        started.set()
        try:
            batch.write()
        except BaseException as e:
            errors.append(e)

    t = threading.Thread(target=write)
    t.start()
    started.wait()
    time.sleep(0.05)  # the batch is now waiting for the slot
    storage.close()  # aborts holder (frees the slot) and the batch
    t.join(5)
    assert not t.is_alive()
    assert errors and isinstance(errors[0], StorageClosedError), errors
    storage.open()
    with storage.snapshot() as snap:
        assert snap.get(("batched",)) is EMPTY
        assert snap.get(("held",)) is EMPTY
    storage.begin_transaction().commit()  # slot free


@pytest.mark.timeout(10)
def test_scan_yields_nothing_after_close(storage: SQLiteStorage) -> None:
    """A page already fetched is not handed out once the context or storage closed."""
    _fill(storage, [(i,) for i in range(100)])
    snap = storage.begin_snapshot()
    items = snap.scan(StorageScanOptions()).items()
    next(items)
    snap.close()
    with pytest.raises(StorageClosedError):
        next(items)

    tx = storage.begin_transaction()
    keys = tx.scan(StorageScanOptions()).keys()
    next(keys)
    storage.close()
    with pytest.raises(StorageClosedError):
        next(keys)
    storage.open()


@pytest.mark.timeout(10)
def test_dropped_transaction_is_collected_and_frees_slot(
    storage: SQLiteStorage, db_path: Path
) -> None:
    """A transaction dropped without commit or abort frees the write lock once collected.

    Both locks: the in-process slot (the next writer here lands fast) and
    the file lock (another process can write).
    """

    def drop() -> None:
        t = storage.begin_transaction()
        t.put(("k",), b"dropped")  # neither committed nor aborted: goes out of scope

    th = threading.Thread(target=drop)
    th.start()
    th.join()
    gc.collect()
    assert not storage._active_transactions
    t0 = time.monotonic()
    with storage.transaction() as tx:
        tx.put(("k",), b"next")
    assert time.monotonic() - t0 < 0.2
    drop()  # on this thread too: the finished hold must not read as nested
    gc.collect()
    assert _other_process_write(db_path) == "wrote"
    with storage.snapshot() as snap:
        assert snap.get(("k",)) == b"next"


@pytest.mark.timeout(10)
def test_dropped_snapshot_is_collected_and_unpins_wal(storage: SQLiteStorage) -> None:
    """A snapshot dropped without close gives its read transaction up once collected."""
    with storage.transaction() as tx:
        tx.put(("k",), b"1")

    def drop() -> None:
        snap = storage.begin_snapshot()
        assert snap.get(("k",)) == b"1"

    drop()
    gc.collect()
    assert not storage._active_snapshots
    with storage.transaction() as tx:
        tx.put(("k",), b"2")
    conn = storage._acquire_connection()
    try:
        # A pinned reader would make a TRUNCATE checkpoint report busy.
        busy, _, _ = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        assert busy == 0
    finally:
        storage._release_connection(conn)


@pytest.mark.timeout(20)
def test_unreleased_slot_collected_under_state_lock_does_not_deadlock(
    storage: SQLiteStorage, db_path: Path
) -> None:
    """GC of an unreleased slot on a thread holding the storage state lock must not hang.

    The slot sits in a reference cycle (a cancelled task's traceback does
    this). Its finalizer must not return the connection to the pool,
    which takes the state lock the collecting thread may already hold.
    """

    async def reserve() -> None:
        slot = await storage.areserve_write_slot()
        cycle = {"slot": slot}
        cycle["self"] = cycle  # unreachable but uncollected until gc runs
        del slot

    asyncio.run(reserve())
    done = threading.Event()

    def collect_under_lock() -> None:
        with storage._state_lock:
            gc.collect()
        done.set()

    t = threading.Thread(target=collect_under_lock, daemon=True)
    t.start()
    assert done.wait(5), "slot finalizer deadlocked on the storage state lock"
    assert not storage._writer.lock.locked()
    assert _other_process_write(db_path) == "wrote"  # the file lock went with the connection
    storage.begin_transaction().commit()


@pytest.mark.timeout(30)
def test_reservation_landing_after_close_holds_nothing(db_path: Path) -> None:
    """close() while the reservation polls the file lock: the slot never lands on a closed storage."""
    proc = _hold_in_other_process(db_path, 0.4)
    s = SQLiteStorage(path=db_path, codec=BinaryCodec(), busy_timeout=5.0)
    s.open()
    try:

        async def main() -> None:
            task = asyncio.create_task(s.areserve_write_slot())
            await asyncio.sleep(0.1)  # polling now, not yet registered
            s.close()
            with pytest.raises(StorageClosedError):
                await task

        asyncio.run(main())
    finally:
        proc.wait(10)
    assert _other_process_write(db_path) == "wrote"  # no file lock left behind
    s.open()
    s.begin_transaction().commit()
    s.close()


_FORK_EXIT = textwrap.dedent(
    """
    import os, sqlite3, sys
    from virtuals._backends.storages.sqlite import SQLiteStorage
    from virtuals.codecs import BinaryCodec

    root, rounds = sys.argv[1], int(sys.argv[2])
    for i in range(rounds):
        path = f"{root}/db{i}.sqlite"
        s = SQLiteStorage(path=path, codec=BinaryCodec(), busy_timeout=1.0, cache_size=10)
        s.open()
        with s.transaction() as t:
            for k in range(2000):
                t.put(("base", k), b"b" * 200)
        tx = s.begin_transaction()
        for k in range(2000):  # spills pages into the WAL before the fork
            tx.put(("pend", k), b"p" * 200)
        pid = os.fork()
        if pid == 0:
            sys.exit(0)  # a normal exit: module globals are cleared, finalizers run
        os.waitpid(pid, 0)
        tx.commit()
        s.close()
        c = sqlite3.connect(path)
        ok = c.execute("PRAGMA integrity_check").fetchone()[0]
        n = c.execute("SELECT count(*) FROM kv").fetchone()[0]
        c.close()
        print(i, ok, n, flush=True)
        if ok != "ok" or n != 4000:
            sys.exit(3)
    """
)


@pytest.mark.timeout(120)
@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_fork_child_normal_exit_leaves_parent_write_intact(tmp_path: Path) -> None:
    """A child that exits normally while the parent holds a large write never corrupts the file.

    Interpreter shutdown in the child must not close the inherited
    connection (that would roll the parent's transaction back in the
    shared WAL index). 10 rounds, each checked with integrity_check.
    """
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _FORK_EXIT, str(tmp_path), "10"],
        capture_output=True,
        text=True,
        timeout=100,
        check=False,
    )
    assert out.returncode == 0, f"rc={out.returncode}\n{out.stdout[-800:]}\n{out.stderr[-1500:]}"
    assert out.stdout.count(" ok 4000") == 10, out.stdout


# =========================================================================
# Fork
# =========================================================================


def _child_exits_with(code_fn, child_thread: bool) -> int:
    """Run ``code_fn`` in a forked child (on a new thread if asked), return its exit code."""
    pid = os.fork()
    if pid == 0:
        code = 1
        try:
            if child_thread:
                box: list[int] = []
                x = threading.Thread(target=lambda: box.append(code_fn()))
                x.start()
                x.join()
                code = box[0] if box else 1
            else:
                code = code_fn()
        finally:
            os._exit(code)
    _, status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(status)


@pytest.mark.timeout(20)
@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
@pytest.mark.parametrize("child_thread", [False, True], ids=["forking-thread", "new-thread"])
def test_fork_during_write_child_write_fails_clearly(
    storage: SQLiteStorage, db_path: Path, child_thread: bool
) -> None:
    """A child forked while a parent thread holds a write gets a clear error, at once.

    The inherited lock state marks the file's write lock as taken in the
    child for good, so a plain begin would wait out busy_timeout and
    raise a retryable error forever. Every write path refuses instead,
    on the inherited storage and on a fresh one opened on the same file,
    and close() never runs a statement on the parent's connections.
    """
    held, go = threading.Event(), threading.Event()
    traced: list[str] = []

    def holder() -> None:
        with storage.transaction() as tx:
            tx.put(("parent",), b"1")
            assert tx._conn is not None
            tx._conn.set_trace_callback(traced.append)
            held.set()
            go.wait()

    t = threading.Thread(target=holder)
    t.start()
    held.wait()
    parent_conn = next(iter(storage._active_transactions))._conn

    def clear(begin) -> bool:
        t0 = time.monotonic()
        try:
            begin()
        except StorageError as e:
            return type(e) is StorageError and "fork" in str(e) and time.monotonic() - t0 < 0.5
        return False

    def child() -> int:
        go.wait(0.2)  # never set here: the parent's commit cannot reach the child
        with storage.snapshot() as snap:  # reads still work
            if snap.get(("parent",)) is not EMPTY:
                return 2
        if not clear(storage.begin_transaction):
            return 3
        if not clear(storage.begin_write_batch):
            return 4
        if not clear(lambda: asyncio.run(storage.areserve_write_slot())):
            return 5
        fresh = SQLiteStorage(path=db_path, codec=BinaryCodec(), busy_timeout=0.2)
        fresh.open()
        if not clear(fresh.begin_transaction):
            return 6
        fresh.close()
        storage.close()  # returns, touches nothing inherited
        if traced or not parent_conn.in_transaction:
            return 7
        return 0

    try:
        assert _child_exits_with(child, child_thread) == 0
    finally:
        go.set()
        t.join()
    assert traced == ["COMMIT"]
    with storage.snapshot() as snap:
        assert snap.get(("parent",)) == b"1"
    storage.begin_transaction().commit()


@pytest.mark.timeout(20)
@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
@pytest.mark.parametrize("child_thread", [False, True], ids=["forking-thread", "new-thread"])
def test_fork_without_write_child_writes(
    storage: SQLiteStorage, db_path: Path, child_thread: bool
) -> None:
    """With no write in progress at fork, the child writes from any thread, and reopens."""
    with storage.transaction() as tx:
        tx.put(("parent",), b"1")
    snap = storage.begin_snapshot()  # an open reader is fine to inherit

    def child() -> int:
        for value in (b"1", b"2"):
            with storage.transaction() as tx:
                tx.put(("child",), value)
        with storage.snapshot() as s:
            if s.get(("child",)) != b"2" or s.get(("parent",)) != b"1":
                return 2
        if not snap.is_closed:  # inherited contexts are forgotten, not ended
            return 3
        storage.close()
        storage.open()
        with storage.transaction() as tx:
            tx.put(("child",), b"3")
        storage.close()
        return 0

    assert _child_exits_with(child, child_thread) == 0
    assert not snap.is_closed  # the parent's snapshot is untouched by the child
    assert snap.get(("child",)) is EMPTY  # pinned before the child's writes
    snap.close()
    with storage.snapshot() as s:
        assert s.get(("child",)) == b"3"
    with storage.transaction() as tx:
        tx.put(("parent",), b"2")
    conn = storage._acquire_connection()
    try:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        storage._release_connection(conn)


@pytest.mark.timeout(20)
def test_reservation_straddling_close_and_open_holds_nothing(db_path: Path, monkeypatch) -> None:
    """close() and open() land between the slot grant and the file lock take: refused, nothing kept.

    The lmdb shape of round 4 audit C1, on sqlite: the reservation is
    refused instead of coming back as a slot the reopened storage does
    not honor, the file lock it took is given back, and a thread writes
    at once.
    """
    import sqlite3

    st = SQLiteStorage(path=db_path, codec=BinaryCodec(), busy_timeout=2.0)
    st.open()
    fired: list = []
    orig = SQLiteStorage._abegin_immediate

    async def abegin_immediate(self, deadline):
        if not fired:  # the other thread wins the race at this point
            fired.append(1)
            st.close()
            st.open()
        return await orig(self, deadline)

    monkeypatch.setattr(SQLiteStorage, "_abegin_immediate", abegin_immediate)

    async def reserve() -> None:
        with pytest.raises(StorageClosedError):
            await st.areserve_write_slot()

    asyncio.run(reserve())
    assert fired
    try:
        assert not st._writer.lock.locked()
        assert not st._reserved_slots
        raw = sqlite3.connect(db_path, timeout=0)  # the file lock is free: taken at once
        try:
            raw.execute("BEGIN IMMEDIATE")
            raw.rollback()
        finally:
            raw.close()
        t0 = time.monotonic()
        with st.transaction() as tx:
            tx.put(("after",), b"1")
        assert time.monotonic() - t0 < 1.0
    finally:
        st.close()


@pytest.mark.timeout(20)
@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_fork_with_reserved_slot_child_refuses_stale_slot(storage: SQLiteStorage) -> None:
    """A slot reserved before the fork cannot begin a transaction in the child."""

    async def main() -> int:
        slot = await storage.areserve_write_slot()

        def child() -> int:
            try:
                storage.begin_transaction(write_slot=slot)
            except StorageError as e:
                if "fork" not in str(e):
                    return 2
            else:
                return 3
            slot.release()  # no-op on a stale slot, never touches the parent's lock
            storage.close()
            return 0

        code = _child_exits_with(child, False)
        storage.begin_transaction(write_slot=slot).commit()  # the parent's slot still works
        return code

    assert asyncio.run(main()) == 0
    storage.begin_transaction().commit()


# =========================================================================
# Writer slot: cancellation, handoff, racing close
# =========================================================================


@pytest.mark.timeout(10)
@pytest.mark.parametrize("ctx", ["transaction", "batch_write"])
def test_cancel_inside_block_releases_slot(storage: SQLiteStorage, ctx: str) -> None:
    cases.cancel_inside_block(storage, ctx)


@pytest.mark.timeout(10)
@pytest.mark.parametrize("ctx", ["transaction", "batch_write"])
def test_keyboard_interrupt_inside_block_releases_slot(storage: SQLiteStorage, ctx: str) -> None:
    cases.keyboard_interrupt_inside_block(storage, ctx)


@pytest.mark.timeout(10)
def test_cancel_while_reserving_holds_nothing(storage: SQLiteStorage) -> None:
    cases.cancel_while_reserving(storage)


def test_cancel_meeting_a_grant_still_cancels(storage: SQLiteStorage) -> None:
    cases.cancel_meeting_a_grant_still_cancels(storage)


@pytest.mark.timeout(10)
def test_cancel_between_reserve_and_begin_releases_slot(storage: SQLiteStorage) -> None:
    cases.cancel_between_reserve_and_begin(storage)


@pytest.mark.timeout(10)
def test_to_thread_nested_raises_fast(storage: SQLiteStorage) -> None:
    cases.to_thread_nested_raises_fast(storage)


@pytest.mark.timeout(10)
def test_asyncio_run_inside_sync_txn_is_nested(storage: SQLiteStorage) -> None:
    cases.asyncio_run_inside_sync_txn_is_nested(storage)


@pytest.mark.timeout(10)
def test_racing_ends_release_slot_once(storage: SQLiteStorage) -> None:
    tx = storage.begin_transaction()
    cases.racing_ends_release_once(storage, tx, lambda t: t._end("ROLLBACK"), "_holds_write_lock")


@pytest.mark.timeout(10)
def test_handoff_worker_is_not_nested(storage: SQLiteStorage) -> None:
    cases.handoff_worker_is_not_nested(storage)


@pytest.mark.timeout(10)
def test_handoff_worker_unrelated_writer_waits(storage: SQLiteStorage) -> None:
    """The full repro: the handed-off txn commits on the loop, the other writer then lands."""

    async def main() -> float:
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(1) as ex:
            slot = await storage.areserve_write_slot()
            tx = await loop.run_in_executor(ex, lambda: storage.begin_transaction(write_slot=slot))
            tx.put(("k",), b"1")

            def other() -> float:
                t0 = time.monotonic()
                with storage.transaction() as t:
                    t.put(("k",), b"2")
                return time.monotonic() - t0

            waiter = loop.run_in_executor(ex, other)
            await asyncio.sleep(0.2)
            tx.commit()
            return await waiter

    assert asyncio.run(main()) >= 0.15
    with storage.snapshot() as snap:
        assert snap.get(("k",)) == b"2"


@pytest.mark.timeout(10)
def test_handoff_then_writers_on_same_worker(storage: SQLiteStorage) -> None:
    cases.handoff_then_writers_on_same_worker(storage)


@pytest.mark.timeout(10)
def test_spawned_task_waits(storage: SQLiteStorage) -> None:
    cases.spawned_task_waits(storage)


@pytest.mark.timeout(10)
def test_close_during_commit_keeps_one_outcome(storage: SQLiteStorage) -> None:
    """close() aborting while the owner commits: committed, not also aborted.

    The abort runs on another thread, started from inside the COMMIT
    statement (a trace callback). It waits on the context lock until the
    commit is through, then finds nothing left to end.
    """
    tx = storage.begin_transaction()
    tx.put(cases.KEY, b"committed")
    errors: list[BaseException] = []
    threads: list[threading.Thread] = []

    def abort() -> None:
        try:
            tx.abort()
        except BaseException as e:
            errors.append(e)

    def trace(sql: str) -> None:
        if sql == "COMMIT":
            t = threading.Thread(target=abort)
            t.start()
            threads.append(t)
            time.sleep(0.05)  # let the abort reach the context lock mid-commit

    assert tx._conn is not None
    tx._conn.set_trace_callback(trace)
    try:
        tx.commit()
        for t in threads:
            t.join(5)
        assert threads and not any(t.is_alive() for t in threads)
        assert not errors, errors
        assert (tx._committed, tx._aborted) == (True, False)
        with storage.snapshot() as snap:
            assert snap.get(cases.KEY) == b"committed"
        storage.begin_transaction().commit()  # slot released exactly once
    finally:
        for conn in storage._idle:
            conn.set_trace_callback(None)


# =============================================================================
# Round 2: loop-thread cap on the file lock, close under a busy-handler
# writer, stale slot wording, slot finalizer in a forked child
# =============================================================================


@pytest.mark.timeout(20)
def test_sync_begin_on_loop_behind_peer_process_is_capped(db_path: Path) -> None:
    """A sync begin or batch write from a coroutine gives up after the loop cap, not busy_timeout.

    The write lock is held by another process, so SQLite's busy handler
    would sleep for the whole ``busy_timeout`` with the loop stalled behind
    it. On a loop thread the file lock wait is capped like the slot wait
    and surfaced as the retryable error that names ``areserve_write_slot``.
    """
    st = SQLiteStorage(path=db_path, codec=BinaryCodec(), busy_timeout=5.0)
    st.open()
    proc = _hold_in_other_process(db_path, 1.0)

    async def main() -> tuple[float, float, str]:
        async with _Ticker() as ticker:
            t0 = time.perf_counter()
            with pytest.raises(StorageLockTimeoutError) as info:
                st.begin_transaction()
            with pytest.raises(StorageLockTimeoutError):
                b = st.begin_write_batch()
                b.put(("x",), b"1")
                b.write()
            waited = time.perf_counter() - t0
        return waited, ticker.gap, str(info.value)

    try:
        waited, gap, message = asyncio.run(main())
    finally:
        proc.wait(10)
        st.close()
    assert waited < 0.4, f"two sync begins on the loop took {waited:.3f}s"
    assert gap < 0.2, f"loop blocked {gap * 1000:.0f}ms behind a peer process"
    assert "areserve_write_slot" in message and "event loop" in message


@pytest.mark.timeout(20)
def test_close_and_reopen_under_busy_handler_writer_refuses_slotless_transaction(
    db_path: Path,
) -> None:
    """A writer waiting in SQLite's busy handler across close() and open() does not get a transaction.

    close() released its slot on its behalf; a transaction returned after
    the reopen would be tracked by the reopened storage without holding
    its slot, and the next in-process writer would wait out busy_timeout
    in SQLite instead of the in-process handoff. The writer gets
    ``StorageClosedError`` instead, and the reopened storage writes at once.
    """
    st = SQLiteStorage(path=db_path, codec=BinaryCodec(), busy_timeout=3.0)
    st.open()
    proc = _hold_in_other_process(db_path, 0.8)
    box: dict = {}

    def writer() -> None:
        try:
            box["txn"] = st.begin_transaction()
        except BaseException as e:
            box["err"] = e

    th = threading.Thread(target=writer)
    th.start()
    time.sleep(0.2)  # the writer holds the slot and waits in BEGIN IMMEDIATE
    st.close()
    st.open()
    th.join(10)
    assert not th.is_alive()
    proc.wait(10)
    try:
        if "txn" in box:
            box["txn"].abort()
        assert isinstance(box.get("err"), StorageClosedError), box
        assert not st._writer.lock.locked()
        t0 = time.perf_counter()
        with st.transaction() as t:
            t.put(("after",), b"1")
        assert time.perf_counter() - t0 < 0.5
    finally:
        st.close()


def test_stale_slot_after_close_names_the_close(storage: SQLiteStorage) -> None:
    """The stale slot error says the storage was closed, as every other backend does."""

    async def main() -> None:
        slot = await storage.areserve_write_slot()
        storage.close()
        storage.open()
        with pytest.raises(StorageError, match="closed") as info:
            storage.begin_transaction(write_slot=slot)
        assert "stale" in str(info.value)
        slot.release()

    asyncio.run(main())
    with storage.transaction() as t:
        t.put(("k",), b"1")


def test_slot_finalizer_in_forked_child_leaves_the_inherited_lock_held(db_path: Path) -> None:
    """A held slot collected in a child before any storage call does not free the parent's lock.

    The lock is the inherited one until ``_check_fork`` replaces it; a
    release here would make the child read the parent's write lock as
    free and go on to wait out busy_timeout in SQLite instead of learning
    at once that writes are refused. The finalizer parks the begun
    connection and leaves the lock alone. Simulated without a fork: the
    slot is told it belongs to another process.
    """
    from virtuals._backends.storages.sqlite import storage as module

    st = SQLiteStorage(path=db_path, codec=BinaryCodec(), busy_timeout=0.5)
    st.open()
    parked = len(module._INHERITED_CONNECTIONS)

    async def main() -> None:
        slot = await st.areserve_write_slot()
        slot._pid = os.getpid() + 1  # as seen by a forked child
        lock = st._writer.lock
        del slot
        gc.collect()
        assert lock.locked(), "the finalizer released the inherited writer lock"
        assert len(module._INHERITED_CONNECTIONS) == parked + 1

    try:
        asyncio.run(main())
    finally:
        st.close()  # releases the lock on the holder's behalf


def test_commit_ended_by_a_racing_close_raises_closed_error(storage: SQLiteStorage) -> None:
    """A commit whose transaction close() ended first raises StorageClosedError, not a wrapped one."""
    from virtuals._backends.storages.sqlite.transaction import SQLiteTransaction

    tx = storage.begin_transaction()
    tx.put(("k",), b"1")
    real_end = SQLiteTransaction._end

    def ended_by_close(self, sql, deferred=None):
        real_end(self, "ROLLBACK")  # close() on another thread took the connection
        return None

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(SQLiteTransaction, "_end", ended_by_close)
        with pytest.raises(StorageClosedError, match="closed while committing"):
            tx.commit()
    assert tx.is_closed
    with storage.transaction() as t:
        t.put(("k",), b"2")


def test_sync_waiter_interrupted_by_signal_leaves_no_ghost(storage) -> None:
    cases.sync_waiter_interrupted_by_signal_leaves_no_ghost(storage)


def test_areserve_task_destroyed_with_its_loop_leaves_no_ghost(storage) -> None:
    cases.areserve_task_destroyed_with_its_loop_leaves_no_ghost(storage)


def test_areserve_task_pending_on_a_closed_loop_is_skipped_at_handoff(storage) -> None:
    cases.areserve_task_pending_on_a_closed_loop_is_skipped_at_handoff(storage)


def test_grant_to_a_loop_closed_after_the_handoff_is_reclaimed(storage) -> None:
    cases.grant_to_a_loop_closed_after_the_handoff_is_reclaimed(storage)


def test_areserve_timeout_meeting_a_release_is_a_lock_timeout(storage) -> None:
    cases.areserve_timeout_meeting_a_release_is_a_lock_timeout(storage)


# =========================================================================
# Round 3 review: an interrupt never changes a commit's outcome or how it is reported (S3)
# =========================================================================


@pytest.mark.timeout(60)
@pytest.mark.parametrize("what", ["transaction", "batch"])
def test_interrupted_commit_settles_once_it_lands(tmp_path: Path, what: str, monkeypatch) -> None:
    """A signal while the COMMIT is in flight: committed, notified once, then the interrupt as it is.

    The signal is sent from inside the end of the context, so it lands in
    the COMMIT itself (a batch applies its writes first, and an interrupt
    there aborts, as designed). Before, the interrupt surfaced after the
    COMMIT with the transaction neither committed nor notified (and
    wrapped as a failed commit when the handler raised a plain
    ``Exception``).
    """
    from virtuals._backends.storages.sqlite.context import ContextBase

    original = ContextBase._end_locked

    class CommitThenSignal:
        """The connection, with the signal sent right after a statement ran: handled inside the end."""

        def __init__(self, conn) -> None:
            self._conn = conn
            self._fired = False

        def execute(self, sql, *args):
            cursor = self._conn.execute(sql, *args)
            if not self._fired:  # once: the proxy goes back to the pool with the connection
                self._fired = True
                os.kill(os.getpid(), signal.SIGUSR1)  # the handler runs at the next bytecode
            return cursor

        def __getattr__(self, name):
            return getattr(self._conn, name)

    def end_with_signal(self, sql, deferred=None):
        self._conn = CommitThenSignal(self._conn)
        return original(self, sql, deferred)

    monkeypatch.setattr(ContextBase, "_end_locked", end_with_signal)
    seen: list[set] = []

    class Recorder:
        def notify(self, keys) -> None:
            seen.append(set(keys))

    st = SQLiteStorage(
        path=tmp_path / "db.sqlite", codec=BinaryCodec(), publisher=Recorder(), synchronous="FULL"
    )
    st.open()
    keys = {("k", i) for i in range(300)}
    try:
        if what == "batch":
            with pytest.raises(cases.Interrupt), st.batch_write() as ctx:
                for key in keys:
                    ctx.put(key, os.urandom(100_000))
                with cases.interrupt_after(60):
                    ctx.write()
            assert ctx._written and not ctx._aborted
        else:
            with pytest.raises(cases.Interrupt), st.transaction() as ctx:
                for key in keys:
                    ctx.put(key, os.urandom(100_000))
                with cases.interrupt_after(60):
                    ctx.commit()
            assert ctx._committed and not ctx._aborted
        monkeypatch.undo()  # before any other end runs the proxy's signal with no handler installed
        assert ctx.is_closed
        assert seen == [keys]
        assert not st._writer.lock.locked()
        with st.snapshot() as snap:
            assert snap.exists(("k", 7))
        with st.transaction() as tx:
            tx.put(("after",), b"1")
    finally:
        st.close()
