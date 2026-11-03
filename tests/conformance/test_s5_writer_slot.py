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
from virtuals.tkv.storage import StorageError, StorageLockTimeoutError


try:
    from virtuals.tkv.storage import StorageNestedTransactionError as NestedError
except ImportError:  # pragma: no cover
    NestedError = None  # type: ignore[assignment,misc]


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
