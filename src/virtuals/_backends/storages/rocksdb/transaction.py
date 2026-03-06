"""Read-write transaction for RocksDB storage."""

from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING

from virtuals.tkv.storage import (
    StorageClosedError,
    StorageTransactionAbortedError,
    StorageTransactionError,
    TransactionProtocol,
)

from .context import ContextBase, ReadOperationsMixin, WriteOperationsMixin


if TYPE_CHECKING:
    from types import TracebackType

    from virtuals.tkv.types import Key

    from .storage import RocksDBStorage


__all__ = ["RocksDBTransaction"]


logger = getLogger(__name__)


class RocksDBTransaction(
    ContextBase, ReadOperationsMixin, WriteOperationsMixin, TransactionProtocol
):
    """Read-write transaction implementation.

    Provides full read-write access with ACID guarantees.
    Composed from base context, read operations, write operations,
    and transaction control.
    """

    __slots__ = ("_aborted", "_committed", "_modified_keys")

    def __init__(
        self,
        storage: RocksDBStorage,
        rdbpy_txn: object,
    ) -> None:
        """Initialize transaction.

        Args:
            storage: Parent storage instance
            rdbpy_txn: RocksDB transaction handle
        """
        super().__init__(storage, rdbpy_txn)
        self._modified_keys: set[Key] = set()
        self._committed = False
        self._aborted = False

    @property
    def writable(self) -> bool:
        """Always True for transactions."""
        return True

    def commit(self) -> None:
        """Commit all changes in the transaction.

        Sends notifications for all modified keys after successful commit.

        Raises:
            StorageTransactionError: If commit fails or transaction is invalid
            StorageClosedError: If context is closed
        """
        if self._closed:
            logger.error("Cannot commit, transaction is closed")
            raise StorageClosedError("Transaction is closed")
        if self._committed:
            logger.error("Cannot commit, transaction already committed")
            raise StorageTransactionError("Transaction already committed")
        if self._aborted:
            logger.error("Cannot commit, transaction already aborted")
            raise StorageTransactionError("Transaction already aborted")

        txn = self._require_active()

        # Commit to RocksDB
        try:
            txn.commit()
        except Exception as e:
            logger.error("Transaction commit failed")
            raise StorageTransactionError(f"Failed to commit transaction: {e}") from e

        logger.info("Transaction committed")

        # Mark as committed before notifications
        self._committed = True
        self._mark_closed()

        # Notify observers of all modifications
        for key in self._modified_keys:
            self._storage._notify(key)

        # Remove from active transactions
        self._storage._remove_transaction(self)

    def abort(self) -> None:
        """Abort transaction and discard all changes.

        Raises:
            StorageTransactionAbortedError: If abort fails
            StorageClosedError: If context is closed
        """
        if self._closed:
            # Already closed, nothing to do
            logger.debug("Abort called on closed transaction")
            return

        txn = self._require_active()

        try:
            txn.rollback()

            logger.info("Transaction aborted")
        except Exception as e:
            logger.error("Transaction abort failed")
            raise StorageTransactionAbortedError(f"Failed to abort transaction: {e}") from e
        finally:
            self._aborted = True
            self._mark_closed()
            self._storage._remove_transaction(self)

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
            # Exception occurred - abort
            try:
                self.abort()
            except Exception:
                logger.error("Transaction abort failed")
        else:
            # No exception - commit if not already done
            if not self._committed and not self._aborted:
                self.commit()
