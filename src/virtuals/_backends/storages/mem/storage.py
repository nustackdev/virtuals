"""In-memory storage backend with copy-on-write transaction isolation.

Fast, ephemeral key-value storage for testing and prototyping. Uses overlay pattern
for efficient transaction isolation without full state copies.

Features:
- Copy-on-write transaction isolation (overlay pattern)
- Thread-safe with RLock
- Optional observer support for notifications
- No persistence - all data lost on close
- Implements full StorageProtocol

Limitations:
- No durability (in-memory only)
- No conflict detection (last commit wins)
- Memory-bound by dataset size
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from logging import getLogger
from typing import TYPE_CHECKING, Any, Literal, overload

from virtuals.tkv.storage import (
    SnapshotProtocol,
    StorageClosedError,
    StorageOperationError,
    TransactionProtocol,
    WriteBatchProtocol,
)

from .snapshot import InMemorySnapshot
from .state import TransactionState
from .transaction import InMemoryTransaction
from .write_batch import InMemoryWriteBatch


if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import TracebackType

    from virtuals.tkv.codec import CodecProtocol
    from virtuals.tkv.observer import ObserverProtocol, Subscription, SubscriptionOptions
    from virtuals.tkv.types import Key


__all__ = ["InMemoryStorage"]


logger = getLogger(__name__)


class InMemoryStorage:
    """In-memory storage backend with copy-on-write transaction isolation.

    Fast, ephemeral key-value storage using overlay pattern for efficient
    transaction isolation. All data is lost when storage is closed.

    Attributes:
        codec: Codec for key/value encoding
    """

    def __init__(
        self,
        codec: CodecProtocol,
        observer: ObserverProtocol | None = None,
        read_only: bool = False,
    ) -> None:
        """Initialize in-memory storage.

        Args:
            codec: Codec for key/value encoding
            observer: Optional observer for change notifications
            read_only: Mode
        """
        self.codec = codec
        self._observer = observer

        self._read_only = read_only

        # State
        self._state: dict[str, Any] = {}  # key_str -> value
        self._opened = False

        # Synchronization (use RLock to allow reentrant locking)
        self._lock = threading.RLock()

        # Context tracking
        self._active_transactions: set[InMemoryTransaction] = set()
        self._active_snapshots: set[InMemorySnapshot] = set()
        self._active_write_batches: set[InMemoryWriteBatch] = set()

    @property
    def read_only(self) -> bool:
        """Storage access."""
        return self._read_only

    def _require_open(self) -> None:
        """Validate storage is open.

        Raises:
            StorageClosedError: If storage is not open
        """
        if not self._opened:
            raise StorageClosedError("Storage is not open")

    def _notify(self, key: Key) -> None:
        """Notify observer of key change.

        Args:
            key: Key that changed
        """
        if self._observer is not None:
            try:
                self._observer.notify(key)
            except Exception:
                logger.error("Observer notification failed")

    def _untrack_transaction(self, txn: InMemoryTransaction) -> None:
        """Remove transaction from active set.

        Args:
            txn: Transaction to untrack
        """
        with self._lock:
            self._active_transactions.discard(txn)

    def _untrack_snapshot(self, snap: InMemorySnapshot) -> None:
        """Remove snapshot from active set.

        Args:
            snap: Snapshot to untrack
        """
        with self._lock:
            self._active_snapshots.discard(snap)

    def _untrack_write_batch(self, batch: InMemoryWriteBatch) -> None:
        """Remove write batch from active set.

        Args:
            batch: Write batch to untrack
        """
        with self._lock:
            self._active_write_batches.discard(batch)

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def open(self) -> None:
        """Open storage and initialize resources.

        Raises:
            StorageOperationError: If open fails
        """
        if self._opened:
            return

        try:
            self._state = {}
            self._opened = True
        except Exception as e:
            raise StorageOperationError(f"Failed to open storage: {e}") from e

    def close(self) -> None:
        """Close storage and release resources.

        All data is lost. Active transactions/snapshots are aborted/closed.

        Raises:
            StorageOperationError: If close fails
        """
        if not self._opened:
            return

        with self._lock:
            # Close all active transactions
            for txn in list(self._active_transactions):
                try:
                    txn.abort()
                except Exception as e:
                    logger.error(f"Failed to abort transaction during close: {e}")

            # Close all active snapshots
            for snap in list(self._active_snapshots):
                try:
                    snap.close()
                except Exception as e:
                    logger.error(f"Failed to close snapshot during close: {e}")

            # Close all active write batches
            for batch in list(self._active_write_batches):
                try:
                    batch.abort()
                except Exception as e:
                    logger.error(f"Failed to abort write batch during close: {e}")

            # Clear tracking sets
            self._active_transactions.clear()
            self._active_snapshots.clear()
            self._active_write_batches.clear()

            # Clear state
            self._state = {}
            self._opened = False

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
    # Subscriptions
    # =========================================================================

    def subscribe(self, options: SubscriptionOptions) -> Subscription:
        """Subscribe to key changes with flexible filtering.

        Args:
            options: Subscription options including filter specification

        Returns:
            Subscription object for binding callbacks and managing lifecycle.

        Raises:
            StorageOperationError: If subscription fails or observer not configured.
        """
        if self._observer is None:
            raise StorageOperationError("Observer not configured for this storage")

        try:
            return self._observer.subscribe(options)
        except Exception as e:
            raise StorageOperationError(f"Failed to subscribe: {e}") from e

    # =========================================================================
    # Transaction Management
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
        """Begin new transaction with specified access level.

        Args:
            read_only: If True, creates a read-only snapshot
            write_only: If True, creates a write-only batch

        Returns:
            SnapshotProtocol if read_only=True
            WriteBatchProtocol if write_only=True
            TransactionProtocol otherwise

        Raises:
            StorageOperationError: If transaction creation fails
        """
        if read_only:
            return self.begin_snapshot()
        elif write_only:
            return self.begin_write_batch()
        else:
            return self.begin_transaction()

    def begin_snapshot(self) -> InMemorySnapshot:
        """Begin read-only snapshot.

        Returns:
            New snapshot instance with full state copy

        Raises:
            StorageOperationError: If snapshot creation fails
        """
        self._require_open()

        with self._lock:
            # Reference live state directly (no copy) — read-committed semantics.
            # Safe for in-process asyncio: no true concurrency between sync operations.
            snapshot = InMemorySnapshot(self, self._state)
            self._active_snapshots.add(snapshot)
            return snapshot

    def begin_transaction(self) -> InMemoryTransaction:
        """Begin read-write transaction.

        Returns:
            New transaction instance with copy-on-write overlay

        Raises:
            StorageOperationError: If transaction creation fails
        """
        self._require_open()

        with self._lock:
            # Create transaction with overlay (no copy)
            overlay = TransactionState(self._state)
            transaction = InMemoryTransaction(self, overlay)
            self._active_transactions.add(transaction)
            return transaction

    def begin_write_batch(self) -> InMemoryWriteBatch:
        """Begin write-only batch.

        Returns:
            New write batch instance with copy-on-write overlay

        Raises:
            StorageOperationError: If batch creation fails
        """
        self._require_open()

        with self._lock:
            # Create write batch with overlay (no copy)
            overlay = TransactionState(self._state)
            write_batch = InMemoryWriteBatch(self, overlay)
            self._active_write_batches.add(write_batch)
            return write_batch

    @contextmanager
    def transaction(self) -> Iterator[InMemoryTransaction]:
        """Context manager for transactions: commit on success, abort on exception."""
        txn = self.begin_transaction()
        try:
            yield txn
        except Exception:
            if not txn._committed and not txn._aborted:
                txn.abort()
            raise
        else:
            if not txn._committed and not txn._aborted:
                txn.commit()

    @contextmanager
    def snapshot(self) -> Iterator[InMemorySnapshot]:
        """Context manager for read-only snapshots: always closes snapshot on exit."""
        snap = self.begin_snapshot()
        try:
            yield snap
        finally:
            try:
                snap.close()
            except Exception as e:
                logger.error(f"Failed to close snapshot during close: {e}")

    @contextmanager
    def batch_write(self) -> Iterator[InMemoryWriteBatch]:
        """Context manager for write batches: write on success, abort on exception."""
        batch = self.begin_write_batch()
        try:
            yield batch
        except Exception:
            if not batch._written and not batch._aborted:
                batch.abort()
            raise
        else:
            if not batch._written and not batch._aborted:
                batch.write()
