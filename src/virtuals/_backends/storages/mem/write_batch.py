"""Write-only batch for in-memory storage."""

from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING, cast

from virtuals.tkv.storage import (
    StorageClosedError,
    StorageTransactionError,
    WriteBatchProtocol,
)

from .context import ContextBase, WriteOperationsMixin


if TYPE_CHECKING:
    from types import TracebackType

    from virtuals.tkv.types import Key

    from .state import TransactionState
    from .storage import InMemoryStorage


__all__ = ["InMemoryWriteBatch"]


logger = getLogger(__name__)


class InMemoryWriteBatch(ContextBase, WriteOperationsMixin, WriteBatchProtocol):
    """Write-only batch for in-memory storage.

    Accumulates writes without read capabilities for efficient bulk operations.
    Uses copy-on-write overlay like transactions.
    """

    __slots__ = ("_aborted", "_modified_keys", "_written")

    def __init__(self, storage: InMemoryStorage, state: TransactionState) -> None:
        """Initialize write batch with overlay state.

        Args:
            storage: Parent storage instance
            state: Overlay state (copy-on-write)
        """
        super().__init__(storage, state)
        self._modified_keys: set[Key] = set()
        self._written = False
        self._aborted = False

    @property
    def writable(self) -> bool:
        """Check if write batch is writable.

        Returns:
            Always True for write batches.
        """
        return True

    def write(self) -> None:
        """Write batch and make changes permanent.

        Raises:
            StorageTransactionError: If write fails
            StorageClosedError: If already written or aborted
        """
        if self._closed:
            logger.error("Cannot write, batch is closed")
            raise StorageClosedError("Write batch is closed")
        if self._written:
            logger.error("Cannot write, batch already written")
            raise StorageTransactionError("Write batch already written")
        if self._aborted:
            logger.error("Cannot write, batch already aborted")
            raise StorageTransactionError("Write batch already aborted")

        try:
            # Apply overlay directly to parent state (O(changes) not O(state_size))
            state = cast("TransactionState", self._state)

            with self._storage._lock:
                for k in state._deleted:
                    self._storage._state.pop(k, None)
                self._storage._state.update(state._local)

            logger.info(
                "Write batch written",
                extra={"modified_keys": len(self._modified_keys)},
            )

            self._written = True
            self._mark_closed()

            # Notify observers (batch)
            self._storage._notify_batch(self._modified_keys)

            self._storage._untrack_write_batch(self)
        except Exception as e:
            logger.error(
                "Write batch write failed",
                extra={"error": str(e)},
                exc_info=True,
            )
            raise StorageTransactionError(f"Failed to write batch: {e}") from e

    def abort(self) -> None:
        """Abort write batch and discard changes.

        Raises:
            StorageTransactionError: If abort fails
        """
        if self._closed:
            # Already closed, nothing to do
            logger.debug("Abort called on closed write batch")
            return

        try:
            logger.info(
                "Write batch aborted",
                extra={"discarded_keys": len(self._modified_keys)},
            )

            self._aborted = True
            self._mark_closed()
            self._storage._untrack_write_batch(self)
        except Exception as e:
            raise StorageTransactionError(f"Failed to abort write batch: {e}") from e

    def __enter__(self) -> InMemoryWriteBatch:
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
            # Success, write if not already done
            if not self._written and not self._aborted:
                self.write()
