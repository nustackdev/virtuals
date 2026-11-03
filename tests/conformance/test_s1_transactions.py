"""Contract section 1: transactions are serializable per storage."""

from __future__ import annotations

import asyncio
import contextlib
import multiprocessing as mp
import threading
import time

import pytest

from tests.conformance.helpers import (
    ITERS,
    PAUSE,
    PROCS,
    RETRYABLE,
    ROUNDS,
    THREADS,
    proc_increment,
    put_committed,
    retrying,
    run_threads,
    wait_barrier,
)
from virtuals.tkv import StorageScanOptions
from virtuals.tkv.filter import PrefixFilter
from virtuals.tkv.storage import exceptions as storage_exceptions
from virtuals.tkv.types import EMPTY


K = ("counter",)
X, Y = ("x",), ("y",)


def _read(storage, key):
    with storage.snapshot() as s:
        return s.get(key)


def _increment(storage, pause: float) -> None:
    def body() -> None:
        with storage.transaction() as t:
            v = t.get(K)
            time.sleep(pause)
            t.put(K, v + 1)

    retrying(body)


@pytest.mark.timeout(20)
def test_s1_lost_update_threads(storage):
    """Concurrent read-modify-write increments from threads all land."""
    put_committed(storage, {K: 0})

    def worker() -> None:
        for _ in range(ITERS):
            _increment(storage, PAUSE)

    run_threads(*[worker] * THREADS)
    assert _read(storage, K) == THREADS * ITERS


@pytest.mark.timeout(60)
def test_s1_lost_update_processes(multi_process, tmp_path):
    """Concurrent increments from processes that each open the storage all land."""
    st = multi_process.make(tmp_path)
    st.open()
    put_committed(st, {K: 0})
    st.close()

    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(
            target=proc_increment, args=(multi_process.name, str(tmp_path), K, ITERS, PAUSE)
        )
        for _ in range(PROCS)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(50)
    stuck = [p for p in procs if p.is_alive()]
    for p in stuck:
        p.kill()
    assert not stuck, "writer process stalled"
    assert [p.exitcode for p in procs] == [0] * PROCS, "writer process failed"

    st = multi_process.make(tmp_path)
    st.open()
    try:
        assert _read(st, K) == PROCS * ITERS
    finally:
        st.close()


@pytest.mark.timeout(20)
def test_s1_write_skew_disjoint_keys(storage):
    """Two txns that each read the other's key and write their own cannot both commit on a stale read.

    Invariant: at most one of x, y is on. Each claimer turns its own key on
    only if it read the other one off.
    """
    barrier = threading.Barrier(2)
    violations = 0
    for _ in range(ROUNDS):
        put_committed(storage, {X: False, Y: False})
        barrier.reset()

        def claim(mine, other):
            def body() -> None:
                with storage.transaction() as t:
                    seen = t.get(other)
                    wait_barrier(barrier)
                    if not seen:
                        t.put(mine, True)

            return lambda: retrying(body)

        run_threads(claim(X, Y), claim(Y, X))
        if _read(storage, X) and _read(storage, Y):
            violations += 1
    assert violations == 0, f"write skew in {violations}/{ROUNDS} rounds"


@pytest.mark.timeout(20)
def test_s1_insert_race(storage):
    """Two txns that create a key only if absent: exactly one believes it created it."""
    barrier = threading.Barrier(2)
    doubles = 0
    for r in range(ROUNDS):
        key = ("slot", r)
        barrier.reset()

        def create(me, key=key):
            def body() -> bool:
                with storage.transaction() as t:
                    present = t.get(key)
                    wait_barrier(barrier)
                    if present is EMPTY:
                        t.put(key, me)
                        return True
                    return False

            return lambda: retrying(body)

        won = run_threads(create("a"), create("b"))
        if sum(won) != 1:
            doubles += 1
    assert doubles == 0, f"both creators won in {doubles}/{ROUNDS} rounds"


@pytest.mark.timeout(20)
def test_s1_scan_write_skew(storage):
    """A txn that decides from a scanned range is not skewed by a concurrent insert into it.

    Invariant: the range ("seat", r, *) holds at most one key. Each txn scans
    it and inserts its own key only if the scan was empty.
    """
    barrier = threading.Barrier(2)
    violations = 0
    for r in range(ROUNDS):
        prefix = ("seat", r)
        barrier.reset()

        def take(me, prefix=prefix):
            def body() -> None:
                with storage.transaction() as t:
                    taken = list(
                        t.scan(
                            StorageScanOptions(
                                start=prefix, break_filter=PrefixFilter(prefix=prefix)
                            )
                        ).keys()
                    )
                    wait_barrier(barrier)
                    if not taken:
                        t.put((*prefix, me), True)

            return lambda: retrying(body)

        run_threads(take("a"), take("b"))
        with storage.snapshot() as s:
            n = len(
                list(
                    s.scan(
                        StorageScanOptions(start=prefix, break_filter=PrefixFilter(prefix=prefix))
                    ).keys()
                )
            )
        if n > 1:
            violations += 1
    assert violations == 0, f"scan write skew in {violations}/{ROUNDS} rounds"


def _prefix_keys(ctx, prefix):
    return list(
        ctx.scan(StorageScanOptions(start=prefix, break_filter=PrefixFilter(prefix=prefix))).keys()
    )


@pytest.mark.timeout(10)
def test_s1_read_your_own_writes(storage):
    """get and scan inside a txn see its own pending set and delete."""
    put_committed(storage, {("r", 1): "old", ("r", 2): "gone"})
    t = storage.begin_transaction()
    try:
        t.put(("r", 1), "new")
        t.put(("r", 3), "added")
        t.delete(("r", 2))
        assert t.get(("r", 1)) == "new"
        assert t.get(("r", 3)) == "added"
        assert t.get(("r", 2)) is EMPTY
        assert dict(
            t.scan(
                StorageScanOptions(start=("r",), break_filter=PrefixFilter(prefix=("r",)))
            ).items()
        ) == {
            ("r", 1): "new",
            ("r", 3): "added",
        }
    finally:
        t.abort()


ATOMIC_KEYS = [("atom", i) for i in range(8)]


@pytest.mark.timeout(10)
def test_s1_atomic_commit_pending_invisible(storage):
    """A pending multi-key txn is invisible to other readers; after commit all its keys are."""
    t = storage.begin_transaction()
    try:
        for k in ATOMIC_KEYS:
            t.put(k, 1)
        seen: list = []
        run_threads(lambda: seen.extend(_prefix_keys_snapshot(storage)))
        assert seen == [], "pending writes visible before commit"
        t.commit()
    except BaseException:
        t.abort()
        raise
    assert _prefix_keys_snapshot(storage) == ATOMIC_KEYS


def _prefix_keys_snapshot(storage):
    with storage.snapshot() as s:
        return _prefix_keys(s, ("atom",))


@pytest.mark.timeout(10)
def test_s1_atomic_commit_abort_leaves_none(storage):
    """An aborted txn (explicit, or by exception in the context manager) leaves no key behind."""
    t = storage.begin_transaction()
    for k in ATOMIC_KEYS:
        t.put(k, 1)
    t.abort()
    assert _prefix_keys_snapshot(storage) == []

    with pytest.raises(RuntimeError, match="boom"):
        with storage.transaction() as t:
            for k in ATOMIC_KEYS:
                t.put(k, 2)
            raise RuntimeError("boom")
    assert _prefix_keys_snapshot(storage) == []


NestedError = getattr(storage_exceptions, "StorageNestedTransactionError", None)


@pytest.mark.timeout(10)
def test_s1_nested_transaction_raises_distinct_error(storage):
    """A second write txn on the same storage, same thread, raises StorageNestedTransactionError.

    The error is not retryable: retrying a programming error never succeeds.
    """
    if NestedError is None:
        pytest.fail("virtuals.tkv.storage.exceptions has no StorageNestedTransactionError")
    assert not issubclass(NestedError, RETRYABLE)

    outer = storage.begin_transaction()
    try:
        outer.put(K, 1)
        inner = None
        try:
            with pytest.raises(NestedError):
                inner = storage.begin_transaction()
                inner.put(K, 2)
                inner.commit()
        finally:
            _quiet_abort(inner)
    finally:
        _quiet_abort(outer)


def _quiet_abort(txn) -> None:
    if txn is None:
        return
    with contextlib.suppress(Exception):
        txn.abort()


@pytest.mark.timeout(10)
def test_s1_nested_batch_write_raises_distinct_error(storage):
    """A write batch on the same storage inside an open write txn raises StorageNestedTransactionError."""
    if NestedError is None:
        pytest.fail("virtuals.tkv.storage.exceptions has no StorageNestedTransactionError")
    outer = storage.begin_transaction()
    try:
        outer.put(K, 1)
        with pytest.raises(NestedError):
            with storage.batch_write() as b:
                b.put(K, 2)
    finally:
        _quiet_abort(outer)


class Interrupt(BaseException):
    """Not an Exception: stands in for KeyboardInterrupt, SystemExit and the like."""


FREE_BUDGET = 0.2  # seconds the next writer may take once the interrupted one is gone
FREE_LOCK_TIMEOUT = 1.0  # bounds a writer stuck behind a leaked lock


def _interrupt_by_cancel(open_writer) -> None:
    async def main() -> None:
        entered = asyncio.Event()

        async def holder() -> None:
            with open_writer() as w:
                w.put(K, 1)
                entered.set()
                await asyncio.sleep(10)

        task = asyncio.create_task(holder())
        await asyncio.wait_for(entered.wait(), 5)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    asyncio.run(main())


def _interrupt_by_base_exception(open_writer) -> None:
    with pytest.raises(Interrupt):
        with open_writer() as w:
            w.put(K, 1)
            raise Interrupt


def _timed_write(storage) -> float:
    t0 = time.monotonic()
    with storage.transaction() as t:
        t.put(K, 2)
    return time.monotonic() - t0


@pytest.mark.parametrize("kind", ["transaction", "batch_write"])
@pytest.mark.parametrize(
    "interrupt",
    [_interrupt_by_cancel, _interrupt_by_base_exception],
    ids=["cancelled", "base_exception"],
)
@pytest.mark.timeout(15)
def test_s1_interrupted_writer_frees_storage(backend, tmp_path, kind, interrupt):
    """A writer cancelled or hit by a BaseException inside its block leaves the storage writable.

    Its write does not land, and the next writer, on another thread and then
    on this one, begins and commits within FREE_BUDGET.
    """
    storage = backend.make(tmp_path, lock_timeout=FREE_LOCK_TIMEOUT)
    storage.open()
    try:
        put_committed(storage, {K: 0})
        interrupt(getattr(storage, kind))
        assert _read(storage, K) == 0, "interrupted write landed"
        other = run_threads(lambda: _timed_write(storage), timeout=FREE_LOCK_TIMEOUT + 5)[0]
        same = _timed_write(storage)
        assert _read(storage, K) == 2
        slow = {n: f"{t * 1000:.0f}ms" for n, t in (("other thread", other), ("same thread", same))}
        assert other < FREE_BUDGET and same < FREE_BUDGET, f"next writer took {slow}"
    finally:
        with contextlib.suppress(Exception):
            storage.close()
