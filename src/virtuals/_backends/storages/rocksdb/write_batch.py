"""Write-only batch for RocksDB storage."""

from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING, Any

from virtuals.tkv.storage import (
    StorageClosedError,
    StorageOperationError,
    StorageTransactionError,
    WriteBatchProtocol,
)

from .context import ContextBase


if TYPE_CHECKING:
    from types import TracebackType

    from virtuals.tkv.types import Key, Value

    from .storage import RocksDBStorage


__all__ = ["RocksDBWriteBatch"]


logger = getLogger(__name__)


class RocksDBWriteBatch(ContextBase, WriteBatchProtocol):
    """Write-only batch implementation for RocksDB.

    Buffers puts and deletes in an rdbpy.WriteBatch and applies them in one
    atomic write. The batch takes the storage's writer slot only for the
    duration of ``write``, so it queues behind transactions and never
    lands in the middle of one.
    """

    __slots__ = ("_aborted", "_modified_keys", "_rdbpy_batch", "_written")

    def __init__(self, storage: RocksDBStorage, rdbpy_batch: Any) -> None:  # noqa: ANN401
        """Initialize write batch.

        Args:
            storage: Parent storage instance
            rdbpy_batch: RocksDB write batch handle
        """
        super().__init__(storage)
        self._rdbpy_batch: Any | None = rdbpy_batch
        self._modified_keys: set[Key] = set()
        self._written = False
        self._aborted = False

    @property
    def writable(self) -> bool:
        """Always True for write batches."""
        return True

    def _drop_handles(self) -> None:
        self._rdbpy_batch = None

    def put(self, key: Key, value: Value) -> None:
        """Put key-value pair into batch.

        Args:
            key: Key to set
            value: Value to store

        Raises:
            StorageOperationError: If write fails
            StorageClosedError: If batch is closed
        """
        self._require_active()
        codec = self._storage.codec

        try:
            encoded_key = codec.encode_key(key)
            encoded_value = codec.encode_value(value)
        except Exception as e:
            raise StorageOperationError(f"Failed to encode key/value for {key}: {e}") from e

        try:
            self._rdbpy_batch.put(encoded_key, encoded_value)  # type: ignore[union-attr]
        except Exception as e:
            raise StorageOperationError(f"Failed to put key {key}: {e}") from e

        self._modified_keys.add(key)

    def delete(self, key: Key) -> None:
        """Delete key from batch (idempotent).

        Silent if key doesn't exist (never raises for missing keys).

        Args:
            key: Key to delete

        Raises:
            StorageOperationError: If deletion fails
            StorageClosedError: If batch is closed
        """
        self._require_active()
        codec = self._storage.codec

        try:
            encoded_key = codec.encode_key(key)
        except Exception as e:
            raise StorageOperationError(f"Failed to encode key {key}: {e}") from e

        try:
            self._rdbpy_batch.delete(encoded_key)  # type: ignore[union-attr]
        except Exception as e:
            raise StorageOperationError(f"Failed to delete key {key}: {e}") from e

        self._modified_keys.add(key)

    def write(self) -> None:
        """Write batch and make changes permanent.

        Takes the writer slot, applies the batch atomically, releases the
        slot. Sends notifications for all modified keys after a successful
        write.

        Raises:
            StorageLockTimeoutError: If the writer slot is not free in time
            StorageNestedTransactionError: If this task or thread already
                holds a write transaction on the storage
            StorageTransactionError: If the write fails
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

        batch = self._rdbpy_batch
        self._storage._require_open()

        try:
            self._storage._write_batch(batch)
        except Exception:
            logger.error("Write batch write failed")
            raise

        logger.debug("Write batch written")

        self._written = True
        self._mark_closed()
        self._storage._notify_batch(self._modified_keys)
        self._storage._untrack_write_batch(self)

    def abort(self) -> None:
        """Abort write batch and discard changes."""
        if self._closed:
            logger.debug("Abort called on closed write batch")
            return

        self._aborted = True
        self._mark_closed()
        self._storage._untrack_write_batch(self)
        logger.debug("Write batch aborted")

    def __enter__(self) -> RocksDBWriteBatch:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager - auto write or abort.

        If an exception occurred, abort the batch.
        Otherwise, write the batch.
        """
        if exc_type is not None:
            try:
                self.abort()
            except Exception:
                logger.error("Write batch abort failed")
        elif not self._written and not self._aborted:
            self.write()
