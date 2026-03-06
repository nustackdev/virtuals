"""Write-only batch for RocksDB storage."""

from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING

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

    Provides efficient bulk write operations using rdbpy.WriteBatch.
    Does not support read operations - optimized for write-heavy workloads.
    """

    __slots__ = ("_aborted", "_modified_keys", "_written")

    def __init__(
        self,
        storage: RocksDBStorage,
        rdbpy_batch: object,
    ) -> None:
        """Initialize write batch.

        Args:
            storage: Parent storage instance
            rdbpy_batch: RocksDB write batch handle
        """
        super().__init__(storage, rdbpy_batch)
        self._modified_keys: set[Key] = set()
        self._written = False
        self._aborted = False

    @property
    def writable(self) -> bool:
        """Always True for write batches."""
        return True

    def put(self, key: Key, value: Value) -> None:
        """Put key-value pair into batch.

        Args:
            key: Key to set
            value: Value to store

        Raises:
            StorageOperationError: If write fails
            StorageClosedError: If batch is closed
        """
        batch = self._require_active()
        codec = self._storage.codec

        # Encode key and value
        try:
            encoded_key = codec.encode_key(key)
            encoded_value = codec.encode_value(value)
        except Exception as e:
            raise StorageOperationError(f"Failed to encode key/value for {key}: {e}") from e

        # Add to batch
        try:
            batch.put(encoded_key, encoded_value)
        except Exception as e:
            raise StorageOperationError(f"Failed to put key {key}: {e}") from e

        # Track modification for notifications
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
        batch = self._require_active()
        codec = self._storage.codec

        try:
            encoded_key = codec.encode_key(key)
        except Exception as e:
            raise StorageOperationError(f"Failed to encode key {key}: {e}") from e

        try:
            # Add delete to batch
            batch.delete(encoded_key)
        except Exception as e:
            raise StorageOperationError(f"Failed to delete key {key}: {e}") from e

        # Track modification for notifications
        self._modified_keys.add(key)

    def write(self) -> None:
        """Write batch and make changes permanent.

        Sends notifications for all modified keys after successful write.

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

        batch = self._require_active()

        # Write to RocksDB
        try:
            self._storage._db.write(batch)
        except Exception as e:
            logger.error("Write batch write failed")
            raise StorageTransactionError(f"Failed to write batch: {e}") from e

        logger.info("Write batch written")

        # Mark as written before notifications
        self._written = True
        self._mark_closed()

        # Notify observers of all modifications
        for key in self._modified_keys:
            self._storage._notify(key)

        # Remove from active batches
        self._storage._remove_write_batch(self)

    def abort(self) -> None:
        """Abort write batch and discard changes.

        Raises:
            StorageTransactionError: If abort fails
            StorageClosedError: If batch is closed
        """
        if self._closed:
            # Already closed, nothing to do
            logger.debug("Abort called on closed write batch")
            return

        try:
            logger.info("Write batch aborted")

            self._aborted = True
            self._mark_closed()
            self._storage._remove_write_batch(self)
        except Exception as e:
            logger.error("Write batch abort failed")
            raise StorageTransactionError(f"Failed to abort write batch: {e}") from e

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
            # Exception occurred - abort
            try:
                self.abort()
            except Exception:
                logger.error("Write batch abort failed")
        else:
            # No exception - write if not already done
            if not self._written and not self._aborted:
                self.write()
