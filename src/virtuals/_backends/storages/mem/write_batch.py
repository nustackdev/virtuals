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

    from .sorted_keys import SortedKeys
    from .storage import InMemoryStorage
    from .store import Encoded, Entry


__all__ = ["InMemoryWriteBatch"]


logger = getLogger(__name__)


class InMemoryWriteBatch(ContextBase, WriteOperationsMixin, WriteBatchProtocol):
    """Buffered writes applied as one atomic commit on ``write``.

    Holds no lock while writes accumulate; ``write`` takes the writer slot
    for just as long as it takes to apply the batch.
    """

    __slots__ = ("_aborted", "_modified_keys", "_pending", "_pending_index", "_written")

    def __init__(self, storage: InMemoryStorage) -> None:
        """Begin an empty batch."""
        super().__init__(storage)
        self._pending: dict[Encoded, Entry] = {}
        self._pending_index: SortedKeys[Encoded] | None = None
        self._modified_keys: set[Key] = set()
        self._written = False
        self._aborted = False

    @property
    def writable(self) -> bool:
        """Always True for write batches."""
        return True

    def write(self) -> None:
        """Apply the batch atomically and notify."""
        if self._closed:
            logger.error("Cannot write, batch is closed")
            raise StorageClosedError("Write batch is closed")
        if self._written:
            raise StorageTransactionError("Write batch already written")
        if self._aborted:
            raise StorageTransactionError("Write batch already aborted")

        storage = self._storage
        modified = self._modified_keys
        try:
            if self._pending:
                storage._check_fork()
                lock = storage._acquire_write_slot()
                try:
                    storage._store.apply(self._pending, modified)
                finally:
                    storage._release_write_slot(lock)
        except Exception as e:
            self._aborted = True
            self._closed = True
            self._pending.clear()
            storage._untrack_write_batch(self)
            logger.error("Write batch write failed")
            if isinstance(e, (StorageLockTimeoutError, StorageNestedTransactionError)):
                raise
            raise StorageTransactionError(f"Failed to write batch: {e}") from e

        self._written = True
        self._closed = True
        self._pending.clear()
        storage._untrack_write_batch(self)
        logger.debug("Write batch written")
        storage._notify_batch(modified)

    def abort(self) -> None:
        """Discard the batch. No-op once closed."""
        if self._closed:
            logger.debug("Abort called on closed write batch")
            return
        self._aborted = True
        self._closed = True
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
        elif not self._written and not self._aborted:
            self.write()
