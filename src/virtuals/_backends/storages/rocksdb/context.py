"""Base context classes and operation mixins for RocksDB storage."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tkv.tkv.storage import (
    ScanProtocol,
    StorageClosedError,
    StorageOperationError,
    StorageScanOptions,
)
from tkv.tkv.types import EMPTY, Empty


if TYPE_CHECKING:
    from tkv.tkv.types import Key, Value

    from .storage import RocksDBStorage


__all__ = [
    "ContextBase",
    "ReadOperationsMixin",
    "WriteOperationsMixin",
]


class ContextBase:
    """Base class for RocksDB transaction/snapshot contexts.

    Provides common functionality for state management, validation, and
    resource cleanup. All contexts wrap a RocksDB transaction handle.
    """

    __slots__ = ("_closed", "_rdbpy_txn", "_storage")

    def __init__(
        self,
        storage: RocksDBStorage,
        rdbpy_txn: Any,  # noqa: ANN401
    ) -> None:
        """Initialize context with storage reference and RocksDB transaction.

        Args:
            storage: Parent storage instance
            rdbpy_txn: RocksDB transaction handle
        """
        self._storage = storage
        self._rdbpy_txn: Any | None = rdbpy_txn
        self._closed = False

    @property
    def storage(self) -> RocksDBStorage:
        """Get the storage instance.

        Returns:
            Storage this context was initiated from.
        """
        return self._storage

    def _require_active(self) -> object:
        """Validate context is active and return transaction handle.

        Returns:
            Active RocksDB transaction handle

        Raises:
            StorageClosedError: If context is closed or invalid
        """
        if self._closed:
            raise StorageClosedError("Context is closed")
        if self._rdbpy_txn is None:
            raise StorageClosedError("Context handle is invalid")
        return self._rdbpy_txn

    def _mark_closed(self) -> None:
        """Mark context as closed and clear handle."""
        self._closed = True
        self._rdbpy_txn = None

    @property
    def is_closed(self) -> bool:
        """Check if context is closed.

        Returns:
            True if closed, False otherwise.
        """
        return self._closed

    @property
    def is_active(self) -> bool:
        """Check if context is active.

        Returns:
            True if active and not closed, False otherwise.
        """
        return not self._closed


class ReadOperationsMixin:
    """Mixin providing read operations for RocksDB contexts.

    Implements point access (get, exists), batch access (multiget),
    and range access (scan) operations.
    """

    __slots__ = ()

    # Type hints for mixed-in attributes
    _storage: RocksDBStorage
    _require_active: Any  # Method from ContextBase

    def get(self, key: Key) -> Value | Empty:
        """Get value by key.

        Args:
            key: Key to retrieve

        Returns:
            Value at key, or EMPTY if key not found.
            Never raises on missing keys.

        Raises:
            StorageOperationError: If operation fails
            StorageClosedError: If context is closed
        """
        txn = self._require_active()
        codec = self._storage.codec

        # Encode key for RocksDB
        try:
            encoded_key = codec.encode_key(key)
        except Exception as e:
            raise StorageOperationError(f"Failed to encode key {key}: {e}") from e

        # Retrieve from RocksDB
        try:
            encoded_value = txn.get(encoded_key)
        except Exception as e:
            raise StorageOperationError(f"Failed to get key {key}: {e}") from e

        # Handle not found - return EMPTY instead of raising
        if encoded_value is None:
            return EMPTY

        # Decode value
        try:
            return codec.decode_value(encoded_value)
        except Exception as e:
            raise StorageOperationError(f"Failed to decode value for key {key}: {e}") from e

    def exists(self, key: Key) -> bool:
        """Check if key exists.

        Args:
            key: Key to check

        Returns:
            True if key exists, False otherwise

        Raises:
            StorageOperationError: If check fails
            StorageClosedError: If context is closed
        """
        txn = self._require_active()
        codec = self._storage.codec

        try:
            encoded_key = codec.encode_key(key)
        except Exception as e:
            raise StorageOperationError(f"Failed to encode key {key}: {e}") from e

        try:
            return txn.get(encoded_key) is not None
        except Exception as e:
            raise StorageOperationError(f"Failed to check key {key}: {e}") from e

    def multiget(self, keys: list[Key]) -> dict[Key, Value]:
        """Get multiple keys.

        Args:
            keys: List of keys to retrieve

        Returns:
            Dict mapping keys to values. Missing keys are omitted.

        Raises:
            StorageOperationError: If operation fails
            StorageClosedError: If context is closed
        """
        result: dict[Key, Value] = {}

        for key in keys:
            value = self.get(key)
            if value is not EMPTY:
                result[key] = value

        return result

    def scan(self, options: StorageScanOptions) -> ScanProtocol:
        """Create scan iterator with configured options.

        Args:
            options: Scan configuration (start, reverse, limit, filter, break_filter)

        Returns:
            Scan iterator conforming to ScanProtocol

        Raises:
            StorageOperationError: If scan creation fails
            StorageClosedError: If context is closed
        """
        from .scan import RocksDBScan

        # Validate context is active before creating scan
        self._require_active()

        return RocksDBScan(self, options)  # type: ignore[arg-type]


class WriteOperationsMixin:
    """Mixin providing write operations for RocksDB contexts.

    Implements point writes (put, delete).
    Tracks modified keys for notification on commit.
    """

    __slots__ = ()

    # Type hints for mixed-in attributes
    _storage: RocksDBStorage
    _require_active: Any  # Method from ContextBase
    _modified_keys: set[Key]  # Initialized in __init__

    def put(self, key: Key, value: Value) -> None:
        """Put key-value pair.

        Args:
            key: Key to set
            value: Value to store

        Raises:
            StorageOperationError: If write fails
            StorageClosedError: If context is closed
        """
        txn = self._require_active()
        codec = self._storage.codec

        # Encode key and value
        try:
            encoded_key = codec.encode_key(key)
            encoded_value = codec.encode_value(value)
        except Exception as e:
            raise StorageOperationError(f"Failed to encode key/value for {key}: {e}") from e

        # Write to RocksDB
        try:
            txn.put(encoded_key, encoded_value)
        except Exception as e:
            raise StorageOperationError(f"Failed to put key {key}: {e}") from e

        # Track modification for notifications
        self._modified_keys.add(key)

    def delete(self, key: Key) -> None:
        """Delete key (idempotent).

        Silent if key doesn't exist (never raises for missing keys).

        Args:
            key: Key to delete

        Raises:
            StorageOperationError: If deletion fails
            StorageClosedError: If context is closed
        """
        txn = self._require_active()
        codec = self._storage.codec

        try:
            encoded_key = codec.encode_key(key)
        except Exception as e:
            raise StorageOperationError(f"Failed to encode key {key}: {e}") from e

        # Check if key exists
        try:
            exists = txn.get(encoded_key) is not None

            if not exists:
                # Key doesn't exist, silently do nothing
                return

            # Delete the key
            txn.delete_single(encoded_key)
        except Exception as e:
            raise StorageOperationError(f"Failed to delete key {key}: {e}") from e

        # Track modification for notifications
        self._modified_keys.add(key)
