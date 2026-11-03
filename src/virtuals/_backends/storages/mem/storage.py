"""In-memory storage backend.

Ephemeral key-value storage for tests, prototypes and per-process state,
meeting the same contract as the persistent adapters:

- Serializable transactions. One writer at a time through an in-process
  writer slot (the SQLite shape: a lock with a thread-and-task owner, a
  bounded sync wait, and an async wait that never blocks the event loop).
  A transaction reads committed state plus its own writes and commits
  atomically.
- Real snapshots. A snapshot is pinned at the commit sequence it opened
  at and reads multi-version state, so it is repeatable and never torn.
  Nothing is copied on begin.
- Ordered keys. A chunked sorted index gives seek, range and reverse scans
  in encoded key order, with first and last in O(log n).
- Notifications after commit through the publisher, like every adapter.

Single-process: nothing is shared across processes, and all data is lost
on close.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
import weakref
from contextlib import contextmanager
from logging import getLogger
from typing import TYPE_CHECKING, Literal, overload

from virtuals.tkv.storage import (
    SnapshotProtocol,
    StorageClosedError,
    StorageError,
    StorageLockTimeoutError,
    StorageNestedTransactionError,
    TransactionProtocol,
    WriteBatchProtocol,
)

from .snapshot import InMemorySnapshot
from .store import VersionedStore
from .transaction import InMemoryTransaction
from .write_batch import InMemoryWriteBatch
from .write_slot import InMemoryWriteSlot


if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import TracebackType

    from virtuals.tkv.codec import CodecProtocol
    from virtuals.tkv.publisher import PublisherProtocol
    from virtuals.tkv.types import Key


__all__ = ["InMemoryStorage"]


logger = getLogger(__name__)


DEFAULT_BUSY_TIMEOUT = 60.0  # seconds

# Poll interval bounds for a coroutine waiting on the writer slot (seconds).
_SLOT_POLL_MIN = 0.0005
_SLOT_POLL_MAX = 0.005

_Owner = tuple[int, "asyncio.Task | None"]

_NESTED_MESSAGE = (
    "This task or thread already holds the write lock on this in-memory storage; "
    "a second write transaction here would wait on itself"
)


_storages: weakref.WeakSet[InMemoryStorage] = weakref.WeakSet()
"""Every live storage in this process (weak), for the fork hooks below."""

_fork_held: list[threading.Lock] = []
"""Store locks taken by ``_before_fork``, released again in the parent."""


def _before_fork() -> None:
    """Hold every store lock across the fork, so no commit is half applied in the child."""
    for storage in list(_storages):
        lock = storage._store.lock
        lock.acquire()
        _fork_held.append(lock)


def _after_fork_in_parent() -> None:
    """Give the store locks back."""
    while _fork_held:
        _fork_held.pop().release()


def _after_fork_in_child() -> None:
    """Fresh locks in the child: a parent thread may hold the inherited ones forever."""
    _fork_held.clear()
    for storage in list(_storages):
        storage._reset_locks()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        before=_before_fork,
        after_in_parent=_after_fork_in_parent,
        after_in_child=_after_fork_in_child,
    )


def _caller() -> _Owner:
    """Who is asking for the writer slot: this thread, and its running task if any."""
    try:
        task = asyncio.current_task()
    except RuntimeError:  # no running event loop on this thread
        task = None
    return threading.get_ident(), task


class InMemoryStorage:
    """In-memory storage conforming to ``StorageProtocol``.

    Threading: contexts may be begun on one thread and finished on
    another. A sync ``begin_transaction`` waits for the writer slot up to
    ``busy_timeout`` and then raises ``StorageLockTimeoutError``. On an
    event loop, wait with ``areserve_write_slot`` and hand the slot to
    ``begin_transaction(write_slot=...)``; a sync begin on a thread whose
    loop peer holds the slot fails fast rather than stalling the loop.
    Beginning a second write transaction from the slot's own holder raises
    ``StorageNestedTransactionError``, which is not retryable.

    Attributes:
        codec: Codec for key and value encoding. Views read it directly.
        process_model: ``"single"``: one process owns the data.
    """

    process_model = "single"

    def __init__(
        self,
        codec: CodecProtocol,
        publisher: PublisherProtocol | None = None,
        read_only: bool = False,
        *,
        busy_timeout: float = DEFAULT_BUSY_TIMEOUT,
    ) -> None:
        """Initialize in-memory storage.

        Args:
            codec: Codec for key and value encoding.
            publisher: Optional publisher for change notifications.
            read_only: Refuse transactions and write batches.
            busy_timeout: Seconds a writer waits for the slot before
                ``StorageLockTimeoutError``.
        """
        self.codec = codec
        self._publisher = publisher
        self._read_only = read_only
        self._busy_timeout = busy_timeout

        self._store = VersionedStore()
        self._opened = False

        self._state_lock = threading.Lock()
        # In-process writer slot. A plain Lock (not RLock): it may be
        # released from a different thread than the one that took it.
        # Mirrors the SQLite slot; keep the two in step until a shared one is lifted.
        self._write_lock = threading.Lock()
        self._write_owner: _Owner | None = None

        self._active_transactions: set[InMemoryTransaction] = set()
        # Weak: a snapshot dropped without close() must be collectable, its
        # finalizer then releases the pin.
        self._active_snapshots: weakref.WeakSet[InMemorySnapshot] = weakref.WeakSet()
        self._active_write_batches: set[InMemoryWriteBatch] = set()
        _storages.add(self)

    @property
    def read_only(self) -> bool:
        """Storage access mode."""
        return self._read_only

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def open(self) -> None:
        """Open storage with empty state."""
        if self._opened:
            return
        self._store.reset()
        self._opened = True

    def close(self) -> None:
        """Close storage. Active contexts are aborted or closed, all data is dropped.

        The writer slot is replaced too, so a slot reserved but not yet
        taken, or a transaction still finishing on another thread, cannot
        hold or free the slot of the reopened storage.
        """
        if not self._opened:
            return

        with self._state_lock:
            transactions = list(self._active_transactions)
            snapshots = list(self._active_snapshots)
            batches = list(self._active_write_batches)
        for txn in transactions:
            try:
                txn.abort()
            except Exception as e:
                logger.error(f"Failed to abort transaction during close: {e}")
        for snap in snapshots:
            try:
                snap.close()
            except Exception as e:
                logger.error(f"Failed to close snapshot during close: {e}")
        for batch in batches:
            try:
                batch.abort()
            except Exception as e:
                logger.error(f"Failed to abort write batch during close: {e}")

        with self._state_lock:
            self._active_transactions.clear()
            self._active_snapshots.clear()
            self._active_write_batches.clear()
            self._write_lock = threading.Lock()
            self._write_owner = None
            self._opened = False
        self._store.reset()

    def __enter__(self) -> InMemoryStorage:
        """Enter context manager."""
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager."""
        self.close()

    # =========================================================================
    # Contexts
    # =========================================================================

    @overload
    def begin(self, *, read_only: Literal[True]) -> SnapshotProtocol: ...

    @overload
    def begin(self, *, write_only: Literal[True]) -> WriteBatchProtocol: ...

    @overload
    def begin(
        self, *, read_only: Literal[False], write_only: Literal[False]
    ) -> TransactionProtocol: ...

    def begin(
        self,
        *,
        read_only: bool = False,
        write_only: bool = False,
    ) -> WriteBatchProtocol | SnapshotProtocol | TransactionProtocol:
        """Begin a snapshot, write batch or transaction."""
        if read_only:
            return self.begin_snapshot()
        elif write_only:
            return self.begin_write_batch()
        else:
            return self.begin_transaction()

    def begin_snapshot(self) -> InMemorySnapshot:
        """Begin a read-only snapshot pinned at this moment."""
        self._require_open()
        snapshot = InMemorySnapshot(self, self._store.pin())
        with self._state_lock:
            self._active_snapshots.add(snapshot)
        return snapshot

    def begin_transaction(
        self, *, write_slot: InMemoryWriteSlot | None = None
    ) -> InMemoryTransaction:
        """Begin a read-write transaction, taking the writer slot now.

        Args:
            write_slot: A slot from ``areserve_write_slot``. The transaction
                takes it over instead of acquiring the slot itself, then
                releases it on commit or abort. None acquires as usual.
        """
        if self._read_only:
            raise StorageError("Cannot start transaction in read only mode.")
        self._require_open()

        if write_slot is not None:
            lock = write_slot._take(self)
        else:
            lock = self._acquire_write_slot()
        try:
            transaction = InMemoryTransaction(self, lock)
            with self._state_lock:
                self._active_transactions.add(transaction)
        except BaseException:
            if write_slot is not None:
                write_slot._give_back()
            else:
                self._release_write_slot(lock)
            raise
        return transaction

    async def areserve_write_slot(self) -> InMemoryWriteSlot:
        """Wait for the writer slot without blocking the event loop.

        Polls the slot with a short, growing sleep between tries, so other
        coroutines on the loop keep running, including the one that holds
        it. Waits behind other threads too.

        Returns:
            The held slot, for ``begin_transaction(write_slot=...)``. Call
            its ``release`` if no transaction takes it.

        Raises:
            StorageNestedTransactionError: If this task already holds the slot.
            StorageLockTimeoutError: If the slot is not free within ``busy_timeout``.
            StorageClosedError: If the storage is closed while waiting.
        """
        if self._read_only:
            raise StorageError("Cannot reserve the write slot in read only mode.")
        self._require_open()

        me = _caller()
        if self._write_owner == me:
            raise StorageNestedTransactionError(_NESTED_MESSAGE)
        deadline = time.monotonic() + max(self._busy_timeout, 0)
        delay = _SLOT_POLL_MIN
        while True:
            lock = self._write_lock
            if lock.acquire(blocking=False):
                if self._claim_slot(lock, me):
                    return InMemoryWriteSlot(self, lock)
                continue  # the slot was replaced by close(); try the current one
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise StorageLockTimeoutError(
                    f"Timed out after {self._busy_timeout}s waiting for the write lock"
                )
            await asyncio.sleep(min(delay, remaining))
            delay = min(delay * 2, _SLOT_POLL_MAX)

    def begin_write_batch(self) -> InMemoryWriteBatch:
        """Begin a write-only batch. Takes no lock until ``write``."""
        if self._read_only:
            raise StorageError("Cannot start write batch in read only mode.")
        self._require_open()
        batch = InMemoryWriteBatch(self)
        with self._state_lock:
            self._active_write_batches.add(batch)
        return batch

    @contextmanager
    def transaction(self) -> Iterator[InMemoryTransaction]:
        """Context manager for a transaction: commit on success, abort on exception."""
        txn = self.begin_transaction()
        try:
            yield txn
        except BaseException:
            txn.abort()
            raise
        else:
            if not txn.is_closed:
                txn.commit()

    @contextmanager
    def snapshot(self) -> Iterator[InMemorySnapshot]:
        """Context manager for a snapshot: always closed on exit."""
        snap = self.begin_snapshot()
        try:
            yield snap
        finally:
            snap.close()

    @contextmanager
    def batch_write(self) -> Iterator[InMemoryWriteBatch]:
        """Context manager for a write batch: write on success, abort on exception."""
        batch = self.begin_write_batch()
        try:
            yield batch
        except BaseException:
            batch.abort()
            raise
        else:
            if not batch.is_closed:
                batch.write()

    # =========================================================================
    # Writer slot
    # =========================================================================

    def _acquire_write_slot(self) -> threading.Lock:
        """Block for the writer slot, up to ``busy_timeout``, and return its lock.

        Fails at once when the slot is held from this same thread: by this
        task or thread (a nested transaction, which would wait on itself),
        or by another coroutine on this thread's loop, which cannot run to
        release it while this thread blocks.
        """
        me = _caller()
        owner = self._write_owner
        if owner is not None and owner[0] == me[0]:
            if owner == me:
                raise StorageNestedTransactionError(_NESTED_MESSAGE)
            raise StorageLockTimeoutError(
                "Another task on this thread's event loop holds the write lock on this "
                "in-memory storage; blocking for it here would stall the loop it needs to "
                "finish. Wait for the slot with areserve_write_slot instead"
            )
        lock = self._write_lock
        if lock.acquire(blocking=False) and self._claim_slot(lock, me):
            return lock  # uncontended fast path
        deadline = time.monotonic() + max(self._busy_timeout, 0)
        while True:
            lock = self._write_lock
            if not lock.acquire(timeout=max(deadline - time.monotonic(), 0)):
                raise StorageLockTimeoutError(
                    f"Timed out after {self._busy_timeout}s waiting for the write lock"
                )
            if self._claim_slot(lock, me):
                return lock
            # The slot was replaced by close() while we waited; wait on the current one.

    def _claim_slot(self, lock: threading.Lock, me: _Owner) -> bool:
        """Record ``me`` as owner of ``lock`` if it is still the storage's open slot.

        Otherwise ``lock`` is a slot ``close()`` replaced while the caller
        waited on it: it is released, nothing is recorded, and False is
        returned so the caller retries on the current slot. Raises
        ``StorageClosedError`` instead when the storage is closed.
        """
        with self._state_lock:
            if lock is self._write_lock and self._opened:
                self._write_owner = me
                return True
            closed = not self._opened
        lock.release()
        if closed:
            raise StorageClosedError("Storage was closed while waiting for the write lock")
        return False

    def _release_write_slot(self, lock: threading.Lock | None = None) -> None:
        """Release the writer slot.

        Args:
            lock: The lock the caller acquired, if known. Nothing is
                released when it is no longer the storage's lock (a fork
                or ``close`` replaced the slot since).
        """
        if lock is not None and lock is not self._write_lock:
            return
        self._write_owner = None
        self._write_lock.release()

    def _reset_locks(self) -> None:
        """Fresh locks and no slot owner, for a forked child.

        A lock held by a parent thread at the fork would never be released
        in the child. The data itself is the child's copy. Called from the
        process-wide fork hook, not under any lock.
        """
        self._state_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._write_owner = None
        self._store.lock = threading.Lock()

    # =========================================================================
    # Internal
    # =========================================================================

    def _notify_batch(self, keys: set[Key]) -> None:
        """Notify the publisher of changed keys after a commit.

        Fire-and-forget: the writer enqueues and returns. Callers that need
        a delivery barrier call ``publisher.flush()`` explicitly.
        """
        if self._publisher is not None and keys:
            try:
                self._publisher.notify(keys)
            except Exception as e:
                logger.error(f"Publisher notification failed: {e}")

    def _untrack_transaction(self, transaction: InMemoryTransaction) -> None:
        """Remove transaction from active set."""
        with self._state_lock:
            self._active_transactions.discard(transaction)

    def _untrack_snapshot(self, snapshot: InMemorySnapshot) -> None:
        """Remove snapshot from active set."""
        with self._state_lock:
            self._active_snapshots.discard(snapshot)

    def _untrack_write_batch(self, batch: InMemoryWriteBatch) -> None:
        """Remove write batch from active set."""
        with self._state_lock:
            self._active_write_batches.discard(batch)

    def _require_open(self) -> None:
        """Validate storage is open."""
        if not self._opened:
            raise StorageClosedError("Storage is not open")


if TYPE_CHECKING:
    _: type[TransactionProtocol] = InMemoryTransaction
    __: type[SnapshotProtocol] = InMemorySnapshot
