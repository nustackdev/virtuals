"""Base context classes and operation mixins for LMDB storage."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from virtuals.tkv.storage import (
    ScanProtocol,
    StorageClosedError,
    StorageOperationError,
    StorageScanOptions,
)
from virtuals.tkv.types import EMPTY, Empty


if TYPE_CHECKING:
    from virtuals.tkv.types import Key, Value

    from .storage import LMDBStorage


__all__ = [
    "ContextBase",
    "ReadOperationsMixin",
    "WriteOperationsMixin",
]


class ContextBase:
    """Base class for LMDB transaction/snapshot contexts.

    Wraps an lmdb.Transaction handle and tracks open/closed state.
    """

    __slots__ = ("_closed", "_lmdb_txn", "_storage")

    def __init__(self, storage: LMDBStorage, lmdb_txn: Any) -> None:  # noqa: ANN401
        """Initialize context with storage reference and LMDB transaction.

        Args:
            storage: Parent storage instance
            lmdb_txn: LMDB transaction handle
        """
        self._storage = storage
        self._lmdb_txn: Any | None = lmdb_txn
        self._closed = False

    @property
    def storage(self) -> LMDBStorage:
        """Get the storage instance."""
        return self._storage

    def _require_active(self) -> Any:  # noqa: ANN401
        """Validate context is active and return LMDB txn handle.

        Raises:
            StorageClosedError: If context is closed or handle missing.
        """
        if self._closed:
            raise StorageClosedError("Context is closed")
        if self._lmdb_txn is None:
            raise StorageClosedError("Context handle is invalid")
        return self._lmdb_txn

    def _mark_closed(self) -> None:
        """Mark context as closed and clear handle."""
        self._closed = True
        self._lmdb_txn = None

    @property
    def is_closed(self) -> bool:
        """Check if context is closed."""
        return self._closed

    @property
    def is_active(self) -> bool:
        """Check if context is active."""
        return not self._closed


class ReadOperationsMixin:
    """Mixin providing read operations for LMDB contexts."""

    __slots__ = ()

    # Type hints for mixed-in attributes
    _storage: LMDBStorage
    _require_active: Any  # Method from ContextBase

    def get(self, key: Key) -> Value | Empty:
        """Get value by key.

        Args:
            key: Key to retrieve.

        Returns:
            Value at key, or EMPTY if key not found.
        """
        txn = self._require_active()
        codec = self._storage.codec

        try:
            encoded_key = codec.encode_key(key)
        except Exception as e:
            raise StorageOperationError(f"Failed to encode key {key}: {e}") from e

        try:
            encoded_value = txn.get(encoded_key)
        except Exception as e:
            raise StorageOperationError(f"Failed to get key {key}: {e}") from e

        if encoded_value is None:
            return EMPTY

        try:
            return codec.decode_value(encoded_value)
        except Exception as e:
            raise StorageOperationError(f"Failed to decode value for key {key}: {e}") from e

    def exists(self, key: Key) -> bool:
        """Check if key exists."""
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

        Missing keys are omitted.
        """
        result: dict[Key, Value] = {}

        for key in keys:
            value = self.get(key)
            if value is not EMPTY:
                result[key] = value

        return result

    def scan(self, options: StorageScanOptions) -> ScanProtocol:
        """Create scan iterator with configured options."""
        from .scan import LMDBScan

        self._require_active()
        return LMDBScan(self, options)  # type: ignore[arg-type]


class WriteOperationsMixin:
    """Mixin providing write operations for LMDB contexts."""

    __slots__ = ()

    # Type hints for mixed-in attributes
    _storage: LMDBStorage
    _require_active: Any  # Method from ContextBase
    _modified_keys: set[Key]  # Initialized in __init__

    def put(self, key: Key, value: Value) -> None:
        """Put key-value pair."""
        txn = self._require_active()
        codec = self._storage.codec

        try:
            encoded_key = codec.encode_key(key)
            encoded_value = codec.encode_value(value)
        except Exception as e:
            raise StorageOperationError(f"Failed to encode key/value for {key}: {e}") from e

        try:
            txn.put(encoded_key, encoded_value)
        except Exception as e:
            raise StorageOperationError(f"Failed to put key {key}: {e}") from e

        self._modified_keys.add(key)

    def delete(self, key: Key) -> None:
        """Delete key (idempotent; no-op on missing key)."""
        txn = self._require_active()
        codec = self._storage.codec

        try:
            encoded_key = codec.encode_key(key)
        except Exception as e:
            raise StorageOperationError(f"Failed to encode key {key}: {e}") from e

        try:
            existed = txn.delete(encoded_key)
        except Exception as e:
            raise StorageOperationError(f"Failed to delete key {key}: {e}") from e

        if existed:
            self._modified_keys.add(key)
