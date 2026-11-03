"""Writer slot cases shared by the SQLite and LMDB storage tests.

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

from virtuals.tkv.storage import StorageLockTimeoutError, StorageNestedTransactionError
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
