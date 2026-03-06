"""Read-write transaction for text storage."""

from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING, Any

from tkv.tkv.storage import (
    StorageClosedError,
    StorageTransactionError,
    TransactionProtocol,
)

from .context import ContextBase, ReadOperationsMixin, WriteOperationsMixin


if TYPE_CHECKING:
    from types import TracebackType

    from tkv.tkv.types import Key

    from .storage import TextStorage


__all__ = ["TextTransaction"]


logger = getLogger(__name__)


class TextTransaction(ContextBase, ReadOperationsMixin, WriteOperationsMixin, TransactionProtocol):
    """Read-write transaction for text storage.

    Provides isolated workspace with commit/abort semantics.
    """

    __slots__ = ("_aborted", "_committed", "_modified_keys")

    def __init__(self, storage: TextStorage, state: dict[str, Any]) -> None:
        """Initialize transaction with workspace.

        Args:
            storage: Parent storage instance
            state: Workspace state (copy of current state)
        """
        super().__init__(storage, state)
        self._modified_keys: set[Key] = set()
        self._committed = False
        self._aborted = False

    @property
    def writable(self) -> bool:
        """Check if transaction is writable.

        Returns:
            Always True for transactions.
        """
        return True

    def commit(self) -> None:
        """Commit transaction and make changes permanent.

        Raises:
            StorageTransactionError: If commit fails
            StorageClosedError: If already committed or aborted
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

        try:
            # Write state to disk
            self._storage._write_state(self._state)

            # Log operation if enabled
            if self._storage._log_operations:
                self._storage._log_operation("commit", None, None)

            logger.info(
                "Transaction committed",
                extra={"modified_keys": len(self._modified_keys)},
            )

            self._committed = True
            self._mark_closed()

            # Notify observers
            for key in self._modified_keys:
                self._storage._notify(key)

            self._storage._untrack_transaction(self)

        except Exception as e:
            logger.error(
                "Transaction commit failed",
                extra={"error": str(e)},
                exc_info=True,
            )
            raise StorageTransactionError(f"Failed to commit transaction: {e}") from e

    def abort(self) -> None:
        """Abort transaction and discard changes.

        Raises:
            StorageTransactionError: If abort fails
        """
        if self._closed:
            # Already closed, nothing to do
            logger.debug("Abort called on closed transaction")
            return

        try:
            # Log operation if enabled
            if self._storage._log_operations:
                self._storage._log_operation("abort", None, None)

            logger.info(
                "Transaction aborted",
                extra={"discarded_keys": len(self._modified_keys)},
            )

            self._aborted = True
            self._mark_closed()
            self._storage._untrack_transaction(self)
        except Exception as e:
            logger.error(
                "Transaction abort failed",
                extra={"error": str(e)},
                exc_info=True,
            )
            raise StorageTransactionError(f"Failed to abort transaction: {e}") from e

    def __enter__(self) -> TextTransaction:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager."""
        if exc_type is not None:
            # Exception occurred, abort
            self.abort()
        else:
            # Success, commit if not already done
            if not self._committed and not self._aborted:
                self.commit()
