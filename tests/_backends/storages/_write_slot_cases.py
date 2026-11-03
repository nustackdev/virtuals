"""Writer slot cases shared by the SQLite, LMDB, RocksDB and in-memory storage tests.

Each case takes an open storage with ``busy_timeout`` of a few seconds, so
a slot left taken shows up as a slow or failed write, not a hang.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import gc
import threading
import time

import pytest

from virtuals._backends.storages._write_slot import LOOP_THREAD_WAIT_CAP
from virtuals.tkv.storage import (
    StorageClosedError,
    StorageError,
    StorageLockTimeoutError,
    StorageNestedTransactionError,
)
from virtuals.tkv.types import EMPTY


KEY = ("slot",)
FAST = 0.5  # seconds; anything slower means a writer waited on a stranded slot


def _write_fast(storage) -> None:
    """A write from this thread and one from another both land at once."""
    t0 = time.monotonic()
    with storage.transaction() as tx:
        tx.put(KEY, b"main")
    errors: list[BaseException] = []

    def other() -> None:
        try:
            with storage.transaction() as tx:
                tx.put(KEY, b"thread")
        except BaseException as e:
            errors.append(e)

    t = threading.Thread(target=other)
    t.start()
    t.join()
    assert not errors, errors
    assert time.monotonic() - t0 < FAST


def _wait_cancelled(task: asyncio.Task) -> None:
    """Swallow the task's CancelledError, keeping no reference to it."""
    try:
        task.result()
    except asyncio.CancelledError:
        pass


def cancel_inside_block(storage, ctx: str) -> None:
    """A coroutine cancelled inside ``with storage.<ctx>()`` aborts and frees the slot."""

    async def body() -> None:
        with getattr(storage, ctx)() as w:
            w.put(KEY, b"cancelled")
            await asyncio.sleep(10)

    async def main() -> None:
        task = asyncio.create_task(body())
        await asyncio.sleep(0.01)
        task.cancel()
        await asyncio.wait([task])
        _wait_cancelled(task)

    asyncio.run(main())
    with storage.snapshot() as snap:
        assert snap.get(KEY) is EMPTY
    _write_fast(storage)


def keyboard_interrupt_inside_block(storage, ctx: str) -> None:
    """KeyboardInterrupt inside ``with storage.<ctx>()`` aborts and frees the slot."""
    with pytest.raises(KeyboardInterrupt), getattr(storage, ctx)() as w:
        w.put(KEY, b"interrupted")
        raise KeyboardInterrupt
    with storage.snapshot() as snap:
        assert snap.get(KEY) is EMPTY
    _write_fast(storage)


def cancel_while_reserving(storage) -> None:
    """Cancelled while waiting in ``areserve_write_slot``, a coroutine holds nothing."""

    async def main() -> None:
        held = await storage.areserve_write_slot()
        task = asyncio.create_task(storage.areserve_write_slot())
        await asyncio.sleep(0.02)
        task.cancel()
        await asyncio.wait([task])
        _wait_cancelled(task)
        held.release()

    asyncio.run(main())
    _write_fast(storage)


def cancel_between_reserve_and_begin(storage) -> None:
    """A reserved slot no transaction took is freed: by ``with``, or once dropped."""

    async def with_block() -> None:
        with await storage.areserve_write_slot():
            await asyncio.sleep(10)

    async def dropped() -> None:
        slot = await storage.areserve_write_slot()  # noqa: F841 (held, never released)
        await asyncio.sleep(10)

    async def main(body) -> None:
        task = asyncio.create_task(body())
        await asyncio.sleep(0.01)
        task.cancel()
        await asyncio.wait([task])
        _wait_cancelled(task)

    asyncio.run(main(with_block))
    _write_fast(storage)
    asyncio.run(main(dropped))
    gc.collect()
    _write_fast(storage)


def to_thread_nested_raises_fast(storage) -> None:
    """A slot reserved on the loop and begun in a worker belongs to that worker."""

    async def main() -> float:
        slot = await storage.areserve_write_slot()

        def work() -> float:
            tx = storage.begin_transaction(write_slot=slot)
            t0 = time.monotonic()
            try:
                with pytest.raises(StorageNestedTransactionError):
                    storage.begin_transaction()
                with pytest.raises(StorageNestedTransactionError):
                    batch = storage.begin_write_batch()  # LMDB takes the slot here
                    batch.put(KEY, b"x")
                    batch.write()  # SQLite takes it here
            finally:
                tx.commit()
            return time.monotonic() - t0

        return await asyncio.to_thread(work)

    assert asyncio.run(main()) < FAST
    _write_fast(storage)


def asyncio_run_inside_sync_txn_is_nested(storage) -> None:
    """``asyncio.run`` inside an open sync transaction cannot write: it is nested."""
    tx = storage.begin_transaction()

    async def inner() -> None:
        with pytest.raises(StorageNestedTransactionError):
            storage.begin_transaction()
        with pytest.raises(StorageNestedTransactionError):
            await storage.areserve_write_slot()

    t0 = time.monotonic()
    try:
        asyncio.run(inner())
    finally:
        tx.commit()
    assert time.monotonic() - t0 < FAST
    _write_fast(storage)


def racing_ends_release_once(storage, tx, end, flag: str) -> None:
    """Two concurrent ends of one context (close aborting while the owner commits).

    The context's slot flag is slowed on read, so without a lock both ends
    see it set and release the slot twice.
    """
    base = next(c.__dict__[flag] for c in type(tx).__mro__ if flag in c.__dict__)

    def get(self):
        value = base.__get__(self)
        time.sleep(0.05)
        return value

    slow = type(
        "Slow" + type(tx).__name__,
        (type(tx),),
        {
            "__slots__": (),
            flag: property(get, lambda self, v: base.__set__(self, v)),
        },
    )
    tx.__class__ = slow

    releases: list[int] = []
    release = storage._release_write_slot

    def counting(lock=None) -> None:
        releases.append(1)
        release(lock)

    storage._release_write_slot = counting
    errors: list[BaseException] = []
    barrier = threading.Barrier(2)

    def run() -> None:
        barrier.wait()
        try:
            end(tx)
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=run) for _ in range(2)]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        del storage._release_write_slot
    assert not errors, errors
    assert len(releases) == 1
    assert not storage._writer.lock.locked()
    _write_fast(storage)


def handoff_worker_is_not_nested(storage) -> None:
    """A pool worker that took a handed-off slot is not the owner for later jobs.

    The slot is reserved on the loop and taken in a ``run_in_executor`` job
    (the worker's own context), then kept held. Another job on that same
    worker must wait for the slot (here: time out), not fail as nested.
    Driven on the writer slot directly, so it holds on LMDB too, where the
    transaction itself could not finish off its thread.
    """
    writer = storage._writer

    async def main() -> tuple[str, float]:
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(1) as ex:
            slot = await storage.areserve_write_slot()
            give_back = await loop.run_in_executor(
                ex, lambda: writer.take(storage, slot, storage._busy_timeout)
            )

            def other() -> tuple[str, float]:
                t0 = time.monotonic()
                try:
                    writer.acquire(0.2)
                except StorageLockTimeoutError:
                    return "timeout", time.monotonic() - t0
                except StorageNestedTransactionError:
                    return "nested", time.monotonic() - t0
                writer.release()
                return "acquired", time.monotonic() - t0

            try:
                return await loop.run_in_executor(ex, other)
            finally:
                give_back()
                slot.release()

    outcome, waited = asyncio.run(main())
    assert outcome == "timeout"
    assert waited >= 0.15
    _write_fast(storage)


def handoff_then_writers_on_same_worker(storage) -> None:
    """After a handed-off transaction, two unrelated writers on its worker both land."""

    async def main() -> None:
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(1) as ex:
            slot = await storage.areserve_write_slot()
            tx = await loop.run_in_executor(ex, lambda: storage.begin_transaction(write_slot=slot))
            await loop.run_in_executor(ex, lambda: (tx.put(KEY, b"handoff"), tx.commit()))

            def write(value: bytes) -> None:
                with storage.transaction() as t:
                    t.put(KEY, value)

            for value in (b"one", b"two"):
                await asyncio.wait_for(loop.run_in_executor(ex, write, value), FAST)

    asyncio.run(main())
    with storage.snapshot() as snap:
        assert snap.get(KEY) == b"two"
    _write_fast(storage)


def spawned_task_waits(storage) -> None:
    """A task spawned by the slot holder carries its context but waits, not nested."""

    async def main() -> None:
        slot = await storage.areserve_write_slot()
        tx = storage.begin_transaction(write_slot=slot)

        async def child() -> None:
            with await storage.areserve_write_slot() as s:
                t = storage.begin_transaction(write_slot=s)
                t.put(KEY, b"child")
                t.commit()

        task = asyncio.create_task(child())
        await asyncio.sleep(0.05)
        assert not task.done()
        tx.put(KEY, b"parent")
        tx.commit()
        await asyncio.wait_for(task, FAST)

    asyncio.run(main())
    with storage.snapshot() as snap:
        assert snap.get(KEY) == b"child"
    _write_fast(storage)


def close_during_commit_keeps_one_outcome(storage, tx, during_commit) -> None:
    """close() aborting while the owner commits: committed, not also aborted.

    ``during_commit`` arranges for ``fn`` to run inside the commit, after
    it has claimed the transaction.
    """
    tx.put(KEY, b"committed")
    errors: list[BaseException] = []

    def abort_from_close() -> None:
        def run() -> None:
            try:
                tx.abort()
            except BaseException as e:
                errors.append(e)

        t = threading.Thread(target=run)
        t.start()
        t.join()

    during_commit(abort_from_close)
    tx.commit()
    assert not errors, errors
    assert (tx._committed, tx._aborted) == (True, False)
    with storage.snapshot() as snap:
        assert snap.get(KEY) == b"committed"
    _write_fast(storage)


def reserved_slot_does_not_survive_close(storage) -> None:
    """A slot reserved and never taken is dropped by close(); the reopened storage writes at once."""

    async def main() -> None:
        slot = await storage.areserve_write_slot()
        storage.close()
        storage.open()
        assert not storage._writer.lock.locked()
        with pytest.raises(StorageError, match="stale"):
            storage.begin_transaction(write_slot=slot)
        assert not slot.held
        slot.release()  # no-op: never touches the fresh slot
        assert not storage._writer.lock.locked()
        t0 = time.monotonic()
        storage.begin_transaction().abort()
        (await storage.areserve_write_slot()).release()
        assert time.monotonic() - t0 < FAST

    asyncio.run(main())
    _write_fast(storage)


def waiter_on_stale_slot_wakes_on_close(storage) -> None:
    """close() frees a slot a reserved-but-untaken WriteSlot holds, so a blocked writer moves on."""
    slot = asyncio.run(storage.areserve_write_slot())  # holds the slot, no transaction takes it
    outcome: dict = {}

    def waiter() -> None:
        t0 = time.monotonic()
        try:
            storage.begin_transaction().abort()
            outcome["began"] = True
        except Exception as e:
            outcome["error"] = type(e)
        outcome["secs"] = time.monotonic() - t0

    t = threading.Thread(target=waiter)
    t.start()
    time.sleep(0.05)
    storage.close()
    t.join(2)
    assert not t.is_alive()
    assert outcome.get("error") is StorageClosedError, outcome
    assert outcome["secs"] < 1.0  # did not wait out busy_timeout
    storage.open()
    slot.release()  # stale: touches nothing
    assert not storage._writer.lock.locked()
    _write_fast(storage)


def writer_waiting_across_close_gets_closed_error(storage) -> None:
    """A sync writer blocked behind a holder when close() runs wakes closed, owning nothing."""
    holder = storage.begin_transaction()
    outcome: dict = {}

    def waiter() -> None:
        try:
            outcome["txn"] = storage.begin_transaction()
        except Exception as e:
            outcome["error"] = type(e)

    t = threading.Thread(target=waiter)
    t.start()
    time.sleep(0.05)
    storage.close()  # aborts the holder, frees the old lock, swaps in a new one
    t.join(2)
    assert not t.is_alive()
    assert outcome == {"error": StorageClosedError}
    assert holder.is_closed
    storage.open()
    x = storage.begin_transaction()
    assert storage._writer._owner is not None and storage._writer._owner[0] == threading.get_ident()
    busy_timeout, storage._busy_timeout = storage._busy_timeout, 0.2
    try:
        with pytest.raises(StorageLockTimeoutError):
            _in_thread(storage.begin_transaction)  # exactly one writer on the reopened storage
    finally:
        storage._busy_timeout = busy_timeout
    x.put(KEY, b"x")
    x.commit()
    with storage.snapshot() as snap:
        assert snap.get(KEY) == b"x"


def dropped_transaction_frees_slot(storage) -> None:
    """A transaction dropped without commit or abort gives the slot back once collected."""

    def drop() -> None:
        t = storage.begin_transaction()
        t.put(KEY, b"dropped")  # neither committed nor aborted: goes out of scope

    th = threading.Thread(target=drop)
    th.start()
    th.join()
    gc.collect()
    t0 = time.monotonic()
    with storage.transaction() as tx:
        tx.put(KEY, b"next")
    assert time.monotonic() - t0 < FAST
    with storage.snapshot() as snap:
        assert snap.get(KEY) == b"next"
    drop()  # on this thread too: the finished hold's mark must not read as nested
    gc.collect()
    _write_fast(storage)


def sync_begin_on_loop_thread_is_bounded(storage, ctx: str) -> None:
    """A sync begin or batch write on the loop thread, while a thread holds the slot, waits at most the cap.

    It raises the retryable ``StorageLockTimeoutError`` pointing at
    ``areserve_write_slot``, and the loop is never stalled for longer.
    """
    holding, release = threading.Event(), threading.Event()

    def holder() -> None:
        with storage.transaction() as t:
            t.put(KEY, b"thread")
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
                await asyncio.sleep(0.002)
                now = time.perf_counter()
                gap, last = max(gap, now - last), now

        tick = asyncio.create_task(ticker())
        await asyncio.sleep(0.01)
        t0 = time.perf_counter()
        with pytest.raises(StorageLockTimeoutError, match="areserve_write_slot"):
            if ctx == "transaction":
                storage.begin_transaction()
            else:
                batch = storage.begin_write_batch()  # LMDB takes the slot here
                batch.put(KEY, b"batch")
                batch.write()  # the others take it here
        waited = time.perf_counter() - t0
        stop.set()
        await tick
        return waited, gap

    try:
        waited, gap = asyncio.run(main())
    finally:
        release.set()
        th.join()
    assert waited < LOOP_THREAD_WAIT_CAP + 0.1, waited  # the cap bounds it, not busy_timeout
    assert gap < 0.2, gap
    _write_fast(storage)


def context_end_after_slot_reset_touches_nothing(storage, tx) -> None:
    """A context ending after the slot was replaced (fork reset) releases by identity only."""
    storage._writer.reset()  # as a forked child does
    tx.abort()  # the fresh lock is not the one this context held
    assert not storage._writer.lock.locked()
    _write_fast(storage)


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


# =============================================================================
# Review round 2 (28-review-rb R1): waiters that leave the queue by other routes
# =============================================================================


def sync_waiter_interrupted_by_signal_leaves_no_ghost(storage) -> None:
    """A sync writer whose wait is cut short by a signal handler raising frees nothing and leaks nothing.

    The next writer must get the slot at once after the holder releases.
    """
    import signal

    class BoomError(Exception):
        pass

    def handler(*_: object) -> None:
        raise BoomError

    held, release = threading.Event(), threading.Event()

    def holder() -> None:
        with storage.transaction() as t:
            t.put(KEY, b"holder")
            held.set()
            release.wait(5)

    th = threading.Thread(target=holder)
    th.start()
    assert held.wait(5)
    previous = signal.signal(signal.SIGALRM, handler)
    try:
        signal.setitimer(signal.ITIMER_REAL, 0.05)
        with pytest.raises(BoomError):
            storage.begin_transaction()  # the main thread waits in the slot
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        release.set()
        th.join(5)
    _write_fast(storage)


def areserve_task_destroyed_with_its_loop_leaves_no_ghost(storage) -> None:
    """A coroutine queued in areserve whose loop closes without cancelling it does not hold up the slot.

    The task is torn down with its loop (``GeneratorExit`` at the await,
    or never resumed at all); at the next release the slot must go to a
    live waiter or be unlocked, not sit granted to a dead one.
    """
    held, release = threading.Event(), threading.Event()

    def holder() -> None:
        with storage.transaction() as t:
            t.put(KEY, b"holder")
            held.set()
            release.wait(5)

    th = threading.Thread(target=holder)
    th.start()
    assert held.wait(5)
    loop = asyncio.new_event_loop()
    try:
        task = loop.create_task(storage.areserve_write_slot())
        loop.run_until_complete(asyncio.sleep(0.01))  # queued now
    finally:
        loop.close()  # without cancelling the task
    del task
    gc.collect()
    release.set()
    th.join(5)
    _write_fast(storage)


def areserve_task_pending_on_a_closed_loop_is_skipped_at_handoff(storage) -> None:
    """A queued waiter whose loop closed but whose task object still lives is skipped at handoff."""
    held, release = threading.Event(), threading.Event()

    def holder() -> None:
        with storage.transaction() as t:
            t.put(KEY, b"holder")
            held.set()
            release.wait(5)

    th = threading.Thread(target=holder)
    th.start()
    assert held.wait(5)
    loop = asyncio.new_event_loop()
    task = loop.create_task(storage.areserve_write_slot())
    loop.run_until_complete(asyncio.sleep(0.01))
    loop.close()
    try:
        release.set()
        th.join(5)
        _write_fast(storage)
    finally:
        del task
        gc.collect()


# =============================================================================
# Re-review (29-rereview-rb F1, F2): a grant to a loop that closes later, a timeout
# landing with a release
# =============================================================================


def grant_to_a_loop_closed_after_the_handoff_is_reclaimed(storage) -> None:
    """A coroutine queued on a loop that is stopped at the handoff holds the grant until the loop closes, not for good.

    The loop ran once, left the task queued and stopped. The holder's
    release hands the slot to that task, whose loop never runs again.
    Once the loop is closed, the next acquire takes the slot back.
    """
    held, release = threading.Event(), threading.Event()

    def holder() -> None:
        with storage.transaction() as t:
            t.put(KEY, b"holder")
            held.set()
            release.wait(5)

    th = threading.Thread(target=holder)
    th.start()
    assert held.wait(5)
    loop = asyncio.new_event_loop()
    task = loop.create_task(storage.areserve_write_slot())
    loop.run_until_complete(asyncio.sleep(0.01))  # queued; the loop is stopped, not closed
    release.set()
    th.join(5)  # the slot was handed to the task
    busy, storage._busy_timeout = storage._busy_timeout, 0.2
    try:
        with pytest.raises(StorageLockTimeoutError):
            storage.begin_transaction()  # held by a loop that may still run
    finally:
        storage._busy_timeout = busy
    loop.close()
    _write_fast(storage)  # reclaimed at this acquire
    del task
    gc.collect()
    _write_fast(storage)


def areserve_timeout_meeting_a_release_is_a_lock_timeout(storage) -> None:
    """An areserve that times out as a release lands raises the retryable lock timeout, not closed.

    On timeout the future is cancelled and the task resumes one turn
    later; a release in that window skips the waiter as dead and unlocks.
    The release here runs as a done callback on the future, standing in
    for a release from another thread in that window.
    """
    held, release = threading.Event(), threading.Event()
    box: dict = {}

    def holder() -> None:
        box["txn"] = storage.begin_transaction()
        held.set()
        release.wait(5)

    th = threading.Thread(target=holder)
    th.start()
    assert held.wait(5)
    writer = storage._writer

    async def main() -> None:
        busy, storage._busy_timeout = storage._busy_timeout, 0.05
        try:
            task = asyncio.create_task(storage.areserve_write_slot())
            await asyncio.sleep(0.01)
            waiter = writer._waiters[0]
            # The holder's transaction is ended from the callback itself, so
            # the release lands inside the window, on this very turn.
            waiter.future.add_done_callback(lambda _: box["txn"].abort())
            with pytest.raises(StorageLockTimeoutError):
                await task
        finally:
            storage._busy_timeout = busy
            release.set()
        th.join(5)
        assert waiter.state == "abandoned"

    asyncio.run(main())
    _write_fast(storage)
