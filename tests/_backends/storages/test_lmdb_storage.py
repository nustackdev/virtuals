"""LMDB-specific behaviour beyond the shared StorageProtocol compliance suite.

Covers the in-process writer slot: nested writers, threads and coroutines
queueing on it, bounded waits, and snapshots never waiting behind a writer.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from typing import TYPE_CHECKING

import pytest

from tests._backends.storages import _write_slot_cases as cases
from virtuals._backends.storages.lmdb import LMDBStorage, LMDBWriteSlot
from virtuals.codecs import BinaryCodec
from virtuals.tkv.storage import (
    StorageError,
    StorageLockTimeoutError,
    StorageNestedTransactionError,
)
from virtuals.tkv.types import EMPTY


if TYPE_CHECKING:
    from pathlib import Path


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
    assert result[1] < 0.05

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
    tx._lmdb_txn.abort()  # end LMDB's side once; the race is in giving back the slot
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

