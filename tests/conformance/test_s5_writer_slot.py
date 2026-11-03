"""Contract sections 1 and 4: the writer slot across drops, loop threads, close and nesting.

Every case runs on every backend. A backend passes when a forgotten
transaction cannot hold its slot, a sync begin on the loop thread is
bounded, a reserved slot never survives close, and nesting through a
worker thread or a nested event loop is the distinct non-retryable error.
"""

from __future__ import annotations

import asyncio
import gc
import threading
import time

import pytest

from tests.conformance.helpers import RETRYABLE
from virtuals.tkv.storage import (
    StorageError,
    StorageLockTimeoutError,
    StorageNestedTransactionError,
)


NestedError = StorageNestedTransactionError


K = ("slot",)
FREE_BUDGET = 0.2  # seconds the next writer may take once the slot should be free
LOCK_TIMEOUT = 1.0  # bounds a writer stuck behind a leaked slot
MAX_GAP = 0.2  # longest the loop may go without running the ticker
LOOP_CAP_BUDGET = 0.3  # a loop-thread sync begin must give up well inside this


@pytest.fixture
def quick(backend, tmp_path):
    """An open storage whose lock waits are short, so a leaked slot fails fast."""
    st = backend.make(tmp_path, lock_timeout=LOCK_TIMEOUT)
    st.open()
    yield st
    st.close()


def _write(storage, value) -> float:
    t0 = time.perf_counter()
    with storage.transaction() as t:
        t.put(K, value)
    return time.perf_counter() - t0


def _in_thread(fn, timeout: float = 5.0):
    """Run ``fn`` on a thread, join it, return its result (or raise its error)."""
    box: dict = {}

    def run() -> None:
        try:
            box["result"] = fn()
        except BaseException as e:
            box["error"] = e

    t = threading.Thread(target=run)
    t.start()
    t.join(timeout)
    assert not t.is_alive(), "thread did not finish"
    if "error" in box:
        raise box["error"]
    return box.get("result")


@pytest.mark.timeout(10)
def test_s1_dropped_transaction_frees_slot(quick):
    """A transaction dropped without commit or abort does not hold the slot once collected."""

    def drop() -> None:
        t = quick.begin_transaction()
        t.put(K, 1)  # goes out of scope: neither committed nor aborted

    th = threading.Thread(target=drop)
    th.start()
    th.join()
    gc.collect()
    assert _write(quick, 2) < FREE_BUDGET
    drop()  # same thread: the finished hold must not read as nested either
    gc.collect()
    assert _write(quick, 3) < FREE_BUDGET
    with quick.snapshot() as s:
        assert s.get(K) == 3


@pytest.mark.timeout(10)
def test_s1_nested_through_worker_or_nested_loop_raises_distinct_error(quick):
    """A write begun below the slot holder, in a worker or a nested loop, is nested, at once."""
    if NestedError is None:
        pytest.fail("virtuals.tkv.storage.exceptions has no StorageNestedTransactionError")
    assert not issubclass(NestedError, RETRYABLE)

    async def through_worker() -> float:
        slot = await quick.areserve_write_slot()

        def work() -> float:
            tx = quick.begin_transaction(write_slot=slot)
            t0 = time.perf_counter()
            try:
                with pytest.raises(NestedError):
                    quick.begin_transaction()
                with pytest.raises(NestedError):
                    with quick.batch_write() as b:
                        b.put(K, 1)
            finally:
                tx.abort()
            return time.perf_counter() - t0

        return await asyncio.to_thread(work)

    assert asyncio.run(through_worker()) < FREE_BUDGET

    outer = quick.begin_transaction()

    async def inner() -> None:
        with pytest.raises(NestedError):
            quick.begin_transaction()
        with pytest.raises(NestedError):
            await quick.areserve_write_slot()

    t0 = time.perf_counter()
    try:
        asyncio.run(inner())
    finally:
        outer.abort()
    assert time.perf_counter() - t0 < FREE_BUDGET
    assert _write(quick, 2) < FREE_BUDGET


@pytest.mark.timeout(10)
@pytest.mark.parametrize("ctx", ["transaction", "batch_write"])
def test_s4_sync_begin_on_loop_thread_is_bounded(backend, tmp_path, ctx):
    """A sync begin or batch write from a coroutine, with a thread holding the slot, gives up fast.

    Contract section 4: a wait on the loop is bounded and surfaced as the
    retryable ``StorageLockTimeoutError``. The storage's own lock timeout
    is long here, so only the loop-thread bound can keep this short.
    """
    st = backend.make(tmp_path, lock_timeout=5.0)
    st.open()
    holding, release = threading.Event(), threading.Event()

    def holder() -> None:
        with st.transaction() as t:
            t.put(K, 1)
            holding.set()
            release.wait(5)

    th = threading.Thread(target=holder)
    th.start()
    assert holding.wait(5)

    async def main() -> tuple[float, float]:
        gap, last = 0.0, time.perf_counter()
        stop = asyncio.Event()

        async def ticker() -> None:
            nonlocal gap, last
            while not stop.is_set():
                await asyncio.sleep(0.005)
                now = time.perf_counter()
                gap, last = max(gap, now - last), now

        tick = asyncio.create_task(ticker())
        await asyncio.sleep(0.01)
        t0 = time.perf_counter()
        with pytest.raises(StorageLockTimeoutError):
            if ctx == "transaction":
                st.begin_transaction()
            else:
                b = st.begin_write_batch()
                b.put(K, 2)
                b.write()
        waited = time.perf_counter() - t0
        stop.set()
        await tick
        return waited, gap

    try:
        waited, gap = asyncio.run(main())
    finally:
        release.set()
        th.join()
        st.close()
    assert waited < LOOP_CAP_BUDGET, f"sync begin on the loop waited {waited * 1000:.0f}ms"
    assert gap < MAX_GAP, f"event loop blocked for {gap * 1000:.0f}ms"


@pytest.mark.timeout(10)
def test_s4_reserved_slot_does_not_survive_close(quick):
    """A slot reserved and never taken is gone after close; the reopened storage writes at once."""

    async def main() -> None:
        slot = await quick.areserve_write_slot()  # kept alive, never released
        quick.close()
        quick.open()
        t0 = time.perf_counter()
        fresh = await quick.areserve_write_slot()
        tx = quick.begin_transaction(write_slot=fresh)
        tx.put(K, 1)
        tx.commit()
        assert time.perf_counter() - t0 < FREE_BUDGET
        with pytest.raises(StorageError) as info:
            quick.begin_transaction(write_slot=slot)
        assert not isinstance(info.value, RETRYABLE), type(info.value).__name__
        slot.release()  # late and stale: touches nothing
        assert _write(quick, 2) < FREE_BUDGET

    asyncio.run(main())
    with quick.snapshot() as s:
        assert s.get(K) == 2


@pytest.mark.timeout(10)
def test_s4_late_release_after_close_and_reopen_frees_nothing(quick):
    """close() and reopen while a commit is in flight: the old holder's late release touches nothing.

    The commit has claimed its end and is about to give the slot back when
    another thread closes the storage, reopens it, and a new writer takes
    the fresh slot. The old release lands afterwards: it must neither free
    that writer's slot nor raise. Holders release by lock identity.
    """
    holder_ready, holder_go = threading.Event(), threading.Event()
    holder_errors: list[BaseException] = []

    def new_holder() -> None:  # a write transaction begun, held and ended on its own thread
        try:
            tx = quick.begin_transaction()
            holder_ready.set()
            holder_go.wait(5)
            tx.abort()
        except BaseException as e:
            holder_errors.append(e)
            holder_ready.set()

    holder = threading.Thread(target=new_holder)
    release = quick._release_write_slot
    fired = []

    def interleaved(lock=None) -> None:
        if not fired:
            fired.append(True)
            _in_thread(lambda: (quick.close(), quick.open()))
            holder.start()
            assert holder_ready.wait(5)
        release(lock)

    quick._release_write_slot = interleaved
    old = quick.begin_transaction()
    old.put(K, 1)
    try:
        old.commit()  # its release lands after the reopen and the new holder's begin
    finally:
        del quick._release_write_slot
        if not holder.is_alive() and not fired:
            holder_go.set()
    try:
        assert fired and not holder_errors, holder_errors
        assert quick._writer.lock.locked()  # the new holder still holds the slot
        busy, quick._busy_timeout = quick._busy_timeout, 0.2
        try:
            with pytest.raises(StorageLockTimeoutError):
                _in_thread(quick.begin_transaction)  # and it is exclusive
        finally:
            quick._busy_timeout = busy
    finally:
        holder_go.set()
        holder.join(5)
    assert not holder_errors, holder_errors
    assert _write(quick, 2) < FREE_BUDGET


SYNC_LOOP = 1.5  # seconds the sync writer threads loop
ASYNC_WINDOW = 1.0  # seconds the async writer tries inside that
MIN_ASYNC = 10  # transactions the async writer must land in the window


@pytest.mark.timeout(20)
def test_s4_async_writer_lands_next_to_sync_writer_threads(backend, tmp_path):
    """An areserve writer on the loop gets the slot while sync threads write in a loop.

    The slot is free for microseconds between two sync transactions, so a
    coroutine that polled for it would never see it free; the slot hands
    off in FIFO order instead. Two sync threads, since one leaves gaps on
    the fastest backend.
    """
    st = backend.make(tmp_path, lock_timeout=0.5)
    st.open()
    stop = threading.Event()
    sync_n = [0]
    errors: list[BaseException] = []

    def writer(i: int) -> None:
        try:
            while not stop.is_set():
                with st.transaction() as t:
                    t.put(("s", i), 1)
                sync_n[0] += 1
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(i,), daemon=True) for i in range(2)]
    for th in threads:
        th.start()

    async def main() -> tuple[int, int, float]:
        ok, failed, worst = 0, 0, 0.0
        await asyncio.sleep(0.05)  # the sync loop is running
        end = time.perf_counter() + ASYNC_WINDOW
        while time.perf_counter() < end:
            t0 = time.perf_counter()
            try:
                slot = await st.areserve_write_slot()
                tx = st.begin_transaction(write_slot=slot)
                tx.put(("a",), 1)
                tx.commit()
                ok += 1
            except StorageLockTimeoutError:
                failed += 1
            worst = max(worst, time.perf_counter() - t0)
        return ok, failed, worst

    try:
        ok, failed, worst = asyncio.run(main())
    finally:
        stop.set()
        for th in threads:
            th.join(5)
        st.close()
    assert not errors, errors[0]
    assert sync_n[0] > 0
    assert failed == 0, f"async writer timed out {failed} times next to {sync_n[0]} sync txns"
    assert ok >= MIN_ASYNC, f"async writer landed {ok} txns next to {sync_n[0]} sync txns"
    assert worst < 0.5, f"worst async wait {worst:.3f}s"
