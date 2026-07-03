"""Read-write transaction for LMDB storage."""

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

    from .storage import LMDBStorage


__all__ = ["LMDBTransaction"]


logger = getLogger(__name__)


class LMDBTransaction(ContextBase, ReadOperationsMixin, WriteOperationsMixin, TransactionProtocol):
    """Read-write transaction backed by an LMDB write transaction.

    LMDB serializes writers at the environment level: only one write
    transaction may be active per environment at a time. This adapter
    lets LMDB do that gating; callers see the standard transaction
    protocol with commit/abort semantics.
    """

    __slots__ = ("_aborted", "_committed", "_modified_keys")

    def __init__(self, storage: LMDBStorage, lmdb_txn: object) -> None:
        """Initialize transaction."""
        super().__init__(storage, lmdb_txn)
        self._modified_keys: set[Key] = set()
        self._committed = False
        self._aborted = False

    @property
    def writable(self) -> bool:
        """Always True for transactions."""
        return True

    def commit(self) -> None:
        """Commit all changes in the transaction."""
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

        try:
            txn.commit()
        except Exception as e:
            logger.error("Transaction commit failed")
            raise StorageTransactionError(f"Failed to commit transaction: {e}") from e

        logger.info("Transaction committed")

        self._committed = True
        self._mark_closed()

        self._storage._notify_batch(self._modified_keys)
        self._storage._untrack_transaction(self)

    def abort(self) -> None:
        """Abort transaction and discard all changes."""
        if self._closed:
            logger.debug("Abort called on closed transaction")
            return

        txn = self._require_active()

        try:
            txn.abort()
            logger.info("Transaction aborted")
        except Exception as e:
            logger.error("Transaction abort failed")
            raise StorageTransactionAbortedError(f"Failed to abort transaction: {e}") from e
        finally:
            self._aborted = True
            self._mark_closed()
            self._storage._untrack_transaction(self)

    def __enter__(self) -> LMDBTransaction:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager - auto commit or abort."""
        if exc_type is not None:
            try:
                self.abort()
            except Exception:
                logger.error("Transaction abort failed")
        else:
            if not self._committed and not self._aborted:
                self.commit()
