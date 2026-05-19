"""Base context classes and operation mixins for RocksDB storage."""

from __future__ import annotations

from typing import TYPE_CHECKING

from virtuals.tkv.storage import (
    ScanProtocol,
    StorageClosedError,
    StorageLockTimeoutError,
    StorageOperationError,
    StorageScanOptions,
    StorageTransactionConflictError,
)
from virtuals.tkv.types import EMPTY, Empty


if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from virtuals.tkv.types import Key, Value

    from .storage import RocksDBStorage


__all__ = [
    "ContextBase",
    "ReadOperationsMixin",
    "WriteOperationsMixin",
]


def _classify_rdbpy(key: Key, action: str, e: Exception) -> StorageOperationError:
    """Map an rdbpy exception to the right storage exception.

    rdbpy surfaces RocksDB statuses as plain `Exception` with a bytes
    message; we sniff for the canonical status strings and map to
    typed errors so policy spans (e.g. RetryOnConflict) can target them.
    """
    msg = str(e)
    lower = msg.lower()
    if "timeout waiting to lock" in lower or "lock timeout" in lower:
        return StorageLockTimeoutError(f"Failed to {action} key {key}: {e}")
    if "busy" in lower or "conflict" in lower or "deadlock" in lower:
        return StorageTransactionConflictError(f"Failed to {action} key {key}: {e}")
    return StorageOperationError(f"Failed to {action} key {key}: {e}")


# Bounded retries for a read-only secondary whose manifest version still
# references an SST file the primary has compacted away. Each retry forces a
# fresh primary catch-up, which advances the secondary past the stale version.
_SECONDARY_STALE_RETRIES = 6


def _is_missing_file_error(e: Exception) -> bool:
    """True if a read failed because it touched a now-deleted SST file.

    This is the signature of a read-only secondary pinned to a manifest
    version older than the primary's current one: the primary compacted and
    deleted the file, but the secondary's view still references it. A
    catch-up to the current manifest drops the reference.
    """
    return "no such file or directory" in str(e).lower()


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

    def _read_with_catchup(self, key: Key, action: str, op: Callable[[], object]) -> object:
        """Run a read `op`, surviving a stale-manifest secondary.

        On a read-only secondary, `op` can fail because the secondary's
        manifest version references an SST the primary already compacted
        away. That is recoverable: force a catch-up to the current manifest
        and retry. Any other failure -- and any failure at all on a primary
        -- is classified and raised immediately, exactly as before.
        """
        storage = self._storage
        last: Exception | None = None
        for _ in range(_SECONDARY_STALE_RETRIES):
            try:
                return op()
            except Exception as e:
                if not (storage._is_secondary and _is_missing_file_error(e)):
                    raise _classify_rdbpy(key, action, e) from e
                last = e
                storage.force_catch_up_with_primary()
        raise _classify_rdbpy(key, action, last) from last  # type: ignore[arg-type]

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

        # Retrieve from RocksDB. `_read_with_catchup` absorbs a stale-manifest
        # secondary (an SST the primary compacted away) by catching up + retrying.
        encoded_value = self._read_with_catchup(key, "get", lambda: txn.get(encoded_key))

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

        return self._read_with_catchup(key, "check", lambda: txn.get(encoded_key) is not None)

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
            raise _classify_rdbpy(key, "put", e) from e

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
            raise _classify_rdbpy(key, "delete", e) from e

        # Track modification for notifications
        self._modified_keys.add(key)
