"""Contract section 4: coroutine writers on one event loop queue, and never stall it."""

from __future__ import annotations

import asyncio
import contextlib
import time

import pytest

from tests.conformance.helpers import CORO_N, aretrying, put_committed


K = ("counter",)
STEP = 0.005  # the await inside each transaction
TICK = 0.005
MAX_GAP = 0.2  # longest the loop may go without running the ticker
WALL = 2.0  # bound for CORO_N=10; scales linearly


class StalledError(Exception):
    """The loop was blocked for MAX_GAP. Not retryable, so writers stop."""


class LoopWatch:
    """Tracks the longest stretch the event loop went without running the ticker."""

    def __init__(self, stop_on_stall: bool) -> None:
        self.stop_on_stall = stop_on_stall
        self.last = time.perf_counter()
        self.gap = 0.0

    def mark(self) -> None:
        now = time.perf_counter()
        self.gap = max(self.gap, now - self.last)
        self.last = now

    def check(self) -> None:
        # Writers call this too: after a blocking call several writers may
        # run before the ticker gets a turn, and each could block again.
        self.gap = max(self.gap, time.perf_counter() - self.last)
        if self.stop_on_stall and self.gap >= MAX_GAP:
            raise StalledError


async def _begin(backend, storage, use_slot: bool):
    if use_slot and backend.has_write_slot(storage):
        slot = await storage.areserve_write_slot()
        try:
            return storage.begin_transaction(write_slot=slot)
        except BaseException:
            slot.release()
            raise
    return storage.begin_transaction()


async def _txn(backend, storage, body, watch: LoopWatch, use_slot: bool) -> None:
    async def attempt() -> None:
        watch.check()
        t = await _begin(backend, storage, use_slot)
        try:
            await body(t)
            watch.check()
            t.commit()
        except BaseException:
            with contextlib.suppress(Exception):
                t.abort()
            raise

    await aretrying(attempt)


def _run(backend, storage, body, *, use_slot: bool, stop_on_stall: bool = False):
    """Run CORO_N coroutines of ``body`` next to a ticker. Return (wall, max loop gap).

    With ``stop_on_stall`` the writers stop once the loop has been blocked
    for MAX_GAP: the verdict is in, so the case does not sit through every
    further stall.
    """

    async def main():
        watch = LoopWatch(stop_on_stall)
        stop = asyncio.Event()

        async def ticker() -> None:
            while not stop.is_set():
                await asyncio.sleep(TICK)
                watch.mark()

        tick = asyncio.create_task(ticker())
        watch.mark()
        t0 = time.perf_counter()
        try:
            results = await asyncio.gather(
                *(_txn(backend, storage, body, watch, use_slot) for _ in range(CORO_N)),
                return_exceptions=True,
            )
        finally:
            wall = time.perf_counter() - t0
            stop.set()
            await tick
        errors = [
            r for r in results if isinstance(r, BaseException) and not isinstance(r, StalledError)
        ]
        if errors:
            raise errors[0]
        return wall, watch.gap

    return asyncio.run(main())


# "slot": writers wait in areserve_write_slot where the backend has it.
# "plain": writers call begin_transaction directly and retry RETRYABLE, even
# where a slot exists, so a sync begin that blocks the loop shows up.
USE_SLOT = pytest.mark.parametrize("use_slot", [True, False], ids=["slot", "plain"])


@USE_SLOT
@pytest.mark.timeout(20)
def test_s4_coroutine_writers_serialize(backend, storage, use_slot):
    """N coroutines doing read, await, write each land their increment, in bounded time."""
    put_committed(storage, {K: 0})

    async def body(t) -> None:
        v = t.get(K)
        await asyncio.sleep(STEP)
        t.put(K, v + 1)

    wall, _ = _run(backend, storage, body, use_slot=use_slot)
    with storage.snapshot() as s:
        assert s.get(K) == CORO_N
    assert wall < WALL * CORO_N / 10, f"{CORO_N} writers took {wall:.2f}s"


@USE_SLOT
@pytest.mark.timeout(20)
def test_s4_coroutine_writers_do_not_block_loop(backend, storage, use_slot):
    """While coroutine writers hold a write across an await, the event loop keeps running.

    Each writer writes first and then awaits before commit, so a backend that
    waits for a held lock on the loop thread shows up as a long loop gap.
    """
    put_committed(storage, {K: 0})

    async def body(t) -> None:
        t.put(K, t.get(K) + 1)
        await asyncio.sleep(STEP)

    wall, gap = _run(backend, storage, body, use_slot=use_slot, stop_on_stall=True)
    assert gap < MAX_GAP, f"event loop blocked for {gap * 1000:.0f}ms"
    assert wall < WALL * CORO_N / 10, f"{CORO_N} writers took {wall:.2f}s"


XP_HOLD = 0.5  # how long the other process keeps the write lock


@pytest.mark.timeout(30)
def test_s4_cross_process_lock_wait_does_not_block_loop(multi_process, tmp_path):
    """Another process holds the write lock: a slot writer waits for it off the loop.

    The holder process keeps a write transaction open for XP_HOLD. On the
    loop, a writer reserves the slot and begins with it; a ticker runs
    alongside. The writer must land (so it did wait), and the loop must
    never go MAX_GAP without the ticker running (so it waited off the
    loop). Multi-process backends only (contract section 4).
    """
    import multiprocessing as mp

    from tests.conformance.helpers import proc_hold_write

    storage = multi_process.make(tmp_path)
    storage.open()
    try:
        ctx = mp.get_context("spawn")
        held = ctx.Event()
        proc = ctx.Process(
            target=proc_hold_write, args=(multi_process.name, str(tmp_path), XP_HOLD, held)
        )
        proc.start()
        assert held.wait(30), "holder process never took the lock"

        async def main():
            watch = LoopWatch(stop_on_stall=False)
            stop = asyncio.Event()

            async def ticker() -> None:
                while not stop.is_set():
                    await asyncio.sleep(TICK)
                    watch.mark()

            tick = asyncio.create_task(ticker())
            await asyncio.sleep(TICK)  # the ticker is running before the wait starts
            watch.mark()
            t0 = time.perf_counter()
            try:
                t = await _begin(multi_process, storage, use_slot=True)
                waited = time.perf_counter() - t0
                watch.check()  # a begin that blocked shows here even if the ticker never ran
                try:
                    t.put(K, 1)
                    t.commit()
                except BaseException:
                    with contextlib.suppress(Exception):
                        t.abort()
                    raise
            finally:
                stop.set()
                await tick
            return waited, watch.gap

        waited, gap = asyncio.run(main())
        proc.join(10)
        assert waited >= XP_HOLD * 0.5, f"writer landed after {waited:.3f}s while the lock was held"
        assert gap < MAX_GAP, (
            f"loop blocked {gap * 1000:.0f}ms waiting for a lock held by another process"
        )
        with storage.snapshot() as s:
            assert s.get(K) == 1 and s.get(("held",)) == 1
    finally:
        storage.close()


XP_HOLD_LONG = 1.0
LOOP_CAP_BUDGET = 0.3  # a loop-thread sync begin must give up well inside this


@pytest.mark.timeout(30)
def test_s4_cross_process_sync_begin_on_loop_is_bounded(multi_process, tmp_path):
    """Another process holds the write lock: a sync begin or batch write from a coroutine gives up fast.

    Contract section 4: a wait on the loop is bounded and surfaced. The
    in-process cap must cover the cross-process lock too, or the same
    call stalls the loop for the whole busy_timeout whenever the holder
    is a peer process. Multi-process backends only.
    """
    import multiprocessing as mp

    from tests.conformance.helpers import proc_hold_write
    from virtuals.tkv.storage import StorageLockTimeoutError

    storage = multi_process.make(tmp_path, lock_timeout=5.0)
    storage.open()
    try:
        ctx = mp.get_context("spawn")
        held = ctx.Event()
        proc = ctx.Process(
            target=proc_hold_write, args=(multi_process.name, str(tmp_path), XP_HOLD_LONG, held)
        )
        proc.start()
        assert held.wait(30), "holder process never took the lock"

        async def main():
            watch = LoopWatch(stop_on_stall=False)
            stop = asyncio.Event()

            async def ticker() -> None:
                while not stop.is_set():
                    await asyncio.sleep(TICK)
                    watch.mark()

            tick = asyncio.create_task(ticker())
            await asyncio.sleep(TICK)
            watch.mark()
            t0 = time.perf_counter()
            try:
                with pytest.raises(StorageLockTimeoutError) as info:
                    storage.begin_transaction()
                with pytest.raises(StorageLockTimeoutError):
                    b = storage.begin_write_batch()
                    b.put(K, 1)
                    b.write()
                waited = time.perf_counter() - t0
                watch.check()
            finally:
                stop.set()
                await tick
            return waited, watch.gap, str(info.value)

        waited, gap, message = asyncio.run(main())
        proc.join(10)
        assert waited < 2 * LOOP_CAP_BUDGET, f"sync begins on the loop waited {waited:.3f}s"
        assert gap < MAX_GAP, f"loop blocked {gap * 1000:.0f}ms behind a peer process"
        assert "areserve_write_slot" in message
    finally:
        storage.close()
