"""Write-only batch for the in-memory storage."""

from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING

from virtuals.tkv.storage import (
    StorageClosedError,
    StorageLockTimeoutError,
    StorageNestedTransactionError,
    StorageTransactionError,
    WriteBatchProtocol,
)

from .context import ContextBase, WriteOperationsMixin


if TYPE_CHECKING:
    from types import TracebackType

    from virtuals.tkv.types import Key

    from .storage import InMemoryStorage
    from .store import Encoded, Entry


__all__ = ["InMemoryWriteBatch"]


logger = getLogger(__name__)


class InMemoryWriteBatch(ContextBase, WriteOperationsMixin, WriteBatchProtocol):
    """Buffered writes applied as one atomic commit on ``write``.

    Holds no lock while writes accumulate; ``write`` takes the writer slot
    for just as long as it takes to apply the batch.
    """

    __slots__ = ("_aborted", "_modified_keys", "_pending", "_written")

    def __init__(self, storage: InMemoryStorage) -> None:
        """Begin an empty batch."""
        super().__init__(storage)
        self._pending: dict[Encoded, Entry] = {}
        self._modified_keys: set[Key] = set()
        self._written = False
        self._aborted = False

    @property
    def writable(self) -> bool:
        """Always True for write batches."""
        return True

    def write(self) -> None:
        """Apply the batch atomically and notify."""
        if not self._claim_end():
            logger.error("Cannot write, batch is closed")
            raise StorageClosedError("Write batch is closed")

        storage = self._storage
        modified = self._modified_keys
        try:
            if self._pending:
                lock = storage._acquire_write_slot()
                try:
                    storage._store.apply(self._pending, modified)
                finally:
                    storage._release_write_slot(lock)
        except BaseException as e:
            self._aborted = True
            self._pending.clear()
            storage._untrack_write_batch(self)
            logger.error("Write batch write failed")
            if isinstance(e, (StorageLockTimeoutError, StorageNestedTransactionError)):
                raise
            if isinstance(e, Exception):
                raise StorageTransactionError(f"Failed to write batch: {e}") from e
            raise

        self._written = True
        self._pending.clear()
        storage._untrack_write_batch(self)
        logger.debug("Write batch written")
        storage._notify_batch(modified)

    def abort(self) -> None:
        """Discard the batch. No-op once closed."""
        if not self._claim_end():
            logger.debug("Abort called on closed write batch")
            return
        self._aborted = True
        self._pending.clear()
        self._storage._untrack_write_batch(self)
        logger.debug("Write batch aborted")

    def __enter__(self) -> InMemoryWriteBatch:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Write on a clean exit, abort on an exception."""
        if exc_type is not None:
            self.abort()
        elif not self._closed:
            self.write()
