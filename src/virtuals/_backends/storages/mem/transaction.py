"""Read-write transaction for in-memory storage."""

from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING, cast

from virtuals.tkv.storage import (
    StorageClosedError,
    StorageTransactionError,
    TransactionProtocol,
)

from .context import ContextBase, ReadOperationsMixin, WriteOperationsMixin


if TYPE_CHECKING:
    from types import TracebackType

    from virtuals.tkv.types import Key

    from .state import TransactionState
    from .storage import InMemoryStorage


__all__ = ["InMemoryTransaction"]


logger = getLogger(__name__)


class InMemoryTransaction(
    ContextBase, ReadOperationsMixin, WriteOperationsMixin, TransactionProtocol
):
    """Read-write transaction for in-memory storage.

    Provides isolated workspace with copy-on-write overlay and commit/abort semantics.
    """

    __slots__ = ("_aborted", "_committed", "_modified_keys")

    def __init__(self, storage: InMemoryStorage, state: TransactionState) -> None:
        """Initialize transaction with overlay state.

        Args:
            storage: Parent storage instance
            state: Overlay state (copy-on-write)
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
            # Apply overlay directly to parent state (O(changes) not O(state_size))
            state = cast("TransactionState", self._state)

            with self._storage._lock:
                for k in state._deleted:
                    self._storage._state.pop(k, None)
                self._storage._state.update(state._local)

            logger.info("Transaction committed")

            self._committed = True
            self._mark_closed()

            # Notify observers
            for key in self._modified_keys:
                self._storage._notify(key)

            self._storage._untrack_transaction(self)
        except Exception as e:
            logger.error("Transaction commit failed")
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
            logger.info("Transaction aborted")

            self._aborted = True
            self._mark_closed()
            self._storage._untrack_transaction(self)
        except Exception as e:
            logger.error("Transaction abort failed")
            raise StorageTransactionError(f"Failed to abort transaction: {e}") from e

    def __enter__(self) -> InMemoryTransaction:
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
