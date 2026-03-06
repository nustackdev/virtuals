"""Write-only batch for text storage."""

from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING, Any

from virtuals.tkv.storage import (
    StorageClosedError,
    StorageTransactionError,
    WriteBatchProtocol,
)

from .context import ContextBase, WriteOperationsMixin


if TYPE_CHECKING:
    from types import TracebackType

    from virtuals.tkv.types import Key

    from .storage import TextStorage


__all__ = ["TextWriteBatch"]


logger = getLogger(__name__)


class TextWriteBatch(ContextBase, WriteOperationsMixin, WriteBatchProtocol):
    """Write-only batch for text storage.

    Accumulates writes without read capabilities for efficient bulk operations.
    """

    __slots__ = ("_aborted", "_modified_keys", "_written")

    def __init__(self, storage: TextStorage, state: dict[str, Any]) -> None:
        """Initialize write batch with workspace.

        Args:
            storage: Parent storage instance
            state: Workspace state (copy of current state)
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
            # Write state to disk
            self._storage._write_state(self._state)

            # Log operation if enabled
            if self._storage._log_operations:
                self._storage._log_operation("write", None, None)

            logger.info(
                "Write batch written",
                extra={"modified_keys": len(self._modified_keys)},
            )

            self._written = True
            self._mark_closed()

            # Notify observers
            for key in self._modified_keys:
                self._storage._notify(key)

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
            # Log operation if enabled
            if self._storage._log_operations:
                self._storage._log_operation("abort", None, None)

            logger.info(
                "Write batch aborted",
                extra={"discarded_keys": len(self._modified_keys)},
            )

            self._aborted = True
            self._mark_closed()
            self._storage._untrack_write_batch(self)
        except Exception as e:
            raise StorageTransactionError(f"Failed to abort write batch: {e}") from e

    def __enter__(self) -> TextWriteBatch:
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
