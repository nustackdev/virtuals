"""Read-write transaction for RocksDB storage."""

from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING, Any

from virtuals.tkv.storage import (
    StorageClosedError,
    StorageLockTimeoutError,
    StorageTransactionAbortedError,
    StorageTransactionConflictError,
    StorageTransactionError,
    TransactionProtocol,
)

from .context import ContextBase, ReadOperationsMixin, WriteOperationsMixin, classify_rdbpy


if TYPE_CHECKING:
    import threading
    from types import TracebackType

    from virtuals.tkv.types import Key

    from .storage import RocksDBStorage


__all__ = ["RocksDBTransaction"]


logger = getLogger(__name__)

# Commit failures that keep their class: the retryable ones, and a handle
# whose storage closed underneath it. Anything else is a transaction error.
_RAISE_AS_IS = (StorageTransactionConflictError, StorageLockTimeoutError, StorageClosedError)


class RocksDBTransaction(
    ContextBase, ReadOperationsMixin, WriteOperationsMixin, TransactionProtocol
):
    """Read-write transaction backed by a pessimistic rdbpy transaction.

    The transaction holds the storage's writer slot from begin until commit
    or abort, so it is the only writer on the storage while it runs: what
    it reads cannot move under it, point reads and scans alike. Reads go
    through the transaction's own snapshot, taken at begin, merged with its
    pending writes.

    The rocksdb row locks and snapshot validation underneath are a second
    line of defence: a writer that somehow bypasses the slot surfaces as a
    retryable conflict or lock timeout, never as a silent lost update.

    Commit or abort on every path is the contract. A transaction dropped
    without either is aborted by its finalizer once it is unreachable
    (rdbpy rolls the native transaction back on its own), so a forgotten
    transaction cannot hold the writer slot until the storage closes; that
    may be much later than the drop, e.g. when a traceback keeps it alive.
    """

    __slots__ = ("_aborted", "_committed", "_modified_keys", "_rdbpy_txn")

    def __init__(
        self,
        storage: RocksDBStorage,
        rdbpy_txn: Any,  # noqa: ANN401
        write_lock: threading.Lock,
    ) -> None:
        """Initialize transaction.

        Args:
            storage: Parent storage instance
            rdbpy_txn: rdbpy transaction handle, begun with a snapshot
            write_lock: The writer lock the transaction holds for the slot
        """
        super().__init__(storage, write_lock=write_lock)
        self._rdbpy_txn: Any | None = rdbpy_txn
        self._modified_keys: set[Key] = set()
        self._committed = False
        self._aborted = False

    @property
    def writable(self) -> bool:
        """Always True for transactions."""
        return True

    def _drop_handles(self) -> None:
        self._rdbpy_txn = None

    def _get_raw(self, encoded_key: bytes) -> bytes | None:
        """Read at the transaction snapshot, merging pending writes."""
        return self._rdbpy_txn.get(encoded_key)  # type: ignore[union-attr]

    def _iterator(self, *, keys_only: bool, reverse: bool) -> Any:  # noqa: ANN401
        """Cursor at the transaction snapshot, merging pending writes."""
        return self._rdbpy_txn.iterator(keys_only=keys_only, reverse=reverse)  # type: ignore[union-attr]

    def commit(self) -> None:
        """Commit all changes in the transaction.

        Sends notifications for all modified keys after successful commit.
        A failed commit ends the transaction (rdbpy rolls it back) and gives
        back the writer slot.

        Raises:
            StorageTransactionConflictError: If rocksdb refused the commit
                as a conflict (retryable)
            StorageLockTimeoutError: If rocksdb timed out on a lock (retryable)
            StorageTransactionError: If commit fails
            StorageClosedError: If the transaction already ended or its
                storage closed
        """
        if self._ended():
            logger.error("Cannot commit, transaction is closed")
            raise StorageClosedError("Transaction is closed")

        txn = self._rdbpy_txn
        try:
            txn.commit()  # type: ignore[union-attr]
        except Exception as e:
            self._aborted = True
            self._mark_closed()
            self._storage._untrack_transaction(self)
            logger.error("Transaction commit failed")
            mapped = classify_rdbpy(e, "Failed to commit transaction")
            if isinstance(mapped, _RAISE_AS_IS):
                raise mapped from e
            raise StorageTransactionError(f"Failed to commit transaction: {e}") from e

        logger.debug("Transaction committed")

        self._committed = True
        self._mark_closed()
        self._storage._notify_batch(self._modified_keys)
        self._storage._untrack_transaction(self)

    def abort(self) -> None:
        """Abort transaction and discard all changes.

        Raises:
            StorageTransactionAbortedError: If abort fails
        """
        if self._ended():
            logger.debug("Abort called on closed transaction")
            return

        txn = self._rdbpy_txn
        try:
            txn.rollback()  # type: ignore[union-attr]
            logger.debug("Transaction aborted")
        except Exception as e:
            logger.error("Transaction abort failed")
            raise StorageTransactionAbortedError(f"Failed to abort transaction: {e}") from e
        finally:
            self._aborted = True
            self._mark_closed()
            self._storage._untrack_transaction(self)

    def _ended(self) -> bool:
        """True once the transaction ended, finishing an end an interrupt cut short.

        The committed or aborted flag is set one statement before
        ``_mark_closed``; an interrupt between the two leaves the backend
        work done and only the bookkeeping missing. A retried commit or
        abort completes it here instead of running the end again.
        """
        if self._closed:
            return True
        if self._committed or self._aborted:
            self._mark_closed()
            self._storage._untrack_transaction(self)
            return True
        return False

    def __del__(self) -> None:
        """Last resort: end a transaction dropped without commit or abort.

        The object is unreachable, so nothing else can end it. The weak
        active set forgets it on its own; the storage's state lock is not
        taken here, since a collection may run on a thread that holds it.
        """
        if getattr(self, "_closed", True):
            return
        self._aborted = True
        try:
            self._mark_closed()
        except Exception:  # noqa: S110 (interpreter shutdown, half-built object)
            pass

    def __enter__(self) -> RocksDBTransaction:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager - auto commit or abort.

        If an exception occurred, abort the transaction.
        Otherwise, commit the transaction.
        """
        if exc_type is not None:
            try:
                self.abort()
            except Exception:
                logger.error("Transaction abort failed")
        elif not self._committed and not self._aborted:
            self.commit()
