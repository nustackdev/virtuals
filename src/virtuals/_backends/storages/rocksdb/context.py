"""Base context classes and operation mixins for RocksDB storage."""

from __future__ import annotations

from typing import TYPE_CHECKING

import rdbpy

from virtuals.tkv.storage import (
    ScanProtocol,
    StorageClosedError,
    StorageError,
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
    "classify_rdbpy",
    "is_missing_file_error",
]


# Conflicts rocksdb reports on a write or a commit: the key moved after the
# transaction's snapshot (Busy, Deadlock is a Busy), the memtable history
# was too short to validate (TryAgain), or the transaction expired.
_CONFLICT = (rdbpy.Busy, rdbpy.TryAgain, rdbpy.Expired)
# A lock wait exceeded its timeout, or the lock table is full.
_LOCK_TIMEOUT = (rdbpy.TimedOut, rdbpy.LockLimit)

# rocksdb::Status::SubCode::kPathNotFound: an IO error on a file that is gone.
_SUBCODE_PATH_NOT_FOUND = 9


def classify_rdbpy(e: BaseException, message: str) -> StorageError:
    """Map an rdbpy exception onto the storage exception hierarchy.

    Mapping is by exception class (rdbpy types every rocksdb status), so
    the retryable classes ``RetryOnConflict`` targets are exact:
    Busy / Deadlock / TryAgain / Expired become
    ``StorageTransactionConflictError``, TimedOut / LockLimit become
    ``StorageLockTimeoutError``. rdbpy raises ``RuntimeError`` for a handle
    whose parent ended (transaction committed, storage closed), which is
    ``StorageClosedError`` here. Everything else is an operation error.
    """
    if isinstance(e, _CONFLICT):
        return StorageTransactionConflictError(f"{message}: {e}")
    if isinstance(e, _LOCK_TIMEOUT):
        return StorageLockTimeoutError(f"{message}: {e}")
    if isinstance(e, RuntimeError):
        return StorageClosedError(f"{message}: {e}")
    return StorageOperationError(f"{message}: {e}")


# Bounded retries for a read-only secondary whose manifest version still
# references an SST file the primary has compacted away. Each retry forces a
# fresh primary catch-up, which advances the secondary past the stale version.
_SECONDARY_STALE_RETRIES = 6


def is_missing_file_error(e: BaseException) -> bool:
    """True if a read failed because it touched a now-deleted SST file.

    This is the signature of a read-only secondary pinned to a manifest
    version older than the primary's current one: the primary compacted and
    deleted the file, but the secondary's view still references it. A
    catch-up to the current manifest drops the reference. rocksdb reports
    it as an IO error with the path-not-found subcode.
    """
    return isinstance(e, rdbpy.RocksIOError) and e.subcode == _SUBCODE_PATH_NOT_FOUND


class ContextBase:
    """Base class for RocksDB transaction, snapshot and batch contexts.

    Tracks open/closed state and, for write contexts, the storage's writer
    slot, which is given back when the context closes. Subclasses own
    their rdbpy handles and drop them in ``_drop_handles``.
    """

    __slots__ = ("_closed", "_holds_write_slot", "_storage")

    def __init__(self, storage: RocksDBStorage, *, holds_write_slot: bool = False) -> None:
        """Initialize context.

        Args:
            storage: Parent storage instance
            holds_write_slot: True if this context holds the storage's
                in-process writer slot and must release it on close.
        """
        self._storage = storage
        self._holds_write_slot = holds_write_slot
        self._closed = False

    @property
    def storage(self) -> RocksDBStorage:
        """Get the storage instance.

        Returns:
            Storage this context was initiated from.
        """
        return self._storage

    def _require_active(self) -> None:
        """Validate context is active.

        Raises:
            StorageClosedError: If context is closed
        """
        if self._closed:
            raise StorageClosedError("Context is closed")

    def _drop_handles(self) -> None:
        """Forget the rdbpy handles. Overridden by subclasses."""

    def _mark_closed(self) -> None:
        """Mark context as closed, drop its handles, and give back the writer slot."""
        self._closed = True
        self._drop_handles()
        if self._holds_write_slot:
            self._holds_write_slot = False
            self._storage._release_write_slot()

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
    and range access (scan) operations over two hooks the context
    provides: ``_get_raw(encoded_key)`` and ``_iterator(keys_only, reverse)``.
    Both read at the context's snapshot: a transaction's own snapshot
    merged with its pending writes, or a Snapshot's DB snapshot.
    """

    __slots__ = ()

    # Type hints for mixed-in attributes
    _storage: RocksDBStorage
    _require_active: Any  # Method from ContextBase
    _get_raw: Callable[[bytes], bytes | None]
    _iterator: Callable[..., Any]

    def _read_with_catchup(self, key: Key, action: str, op: Callable[[], object]) -> object:
        """Run a read `op`, surviving a stale-manifest secondary.

        On a read-only secondary, `op` can fail because the secondary's
        manifest version references an SST the primary already compacted
        away. That is recoverable: force a catch-up to the current manifest
        and retry. Any other failure, and any failure at all on a primary,
        is classified and raised immediately.
        """
        storage = self._storage
        message = f"Failed to {action} key {key}"
        last: Exception | None = None
        for _ in range(_SECONDARY_STALE_RETRIES):
            try:
                return op()
            except Exception as e:
                if not (storage._is_secondary and is_missing_file_error(e)):
                    raise classify_rdbpy(e, message) from e
                last = e
                storage.force_catch_up_with_primary()
        raise classify_rdbpy(last, message) from last  # type: ignore[arg-type]

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
        self._require_active()
        codec = self._storage.codec

        try:
            encoded_key = codec.encode_key(key)
        except Exception as e:
            raise StorageOperationError(f"Failed to encode key {key}: {e}") from e

        encoded_value = self._read_with_catchup(key, "get", lambda: self._get_raw(encoded_key))

        if encoded_value is None:
            return EMPTY

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
        self._require_active()
        codec = self._storage.codec

        try:
            encoded_key = codec.encode_key(key)
        except Exception as e:
            raise StorageOperationError(f"Failed to encode key {key}: {e}") from e

        return self._read_with_catchup(key, "check", lambda: self._get_raw(encoded_key) is not None)

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

        self._require_active()

        return RocksDBScan(self, options)  # type: ignore[arg-type]


class WriteOperationsMixin:
    """Mixin providing write operations for RocksDB transactions.

    Implements point writes (put, delete) on the rdbpy transaction.
    Tracks modified keys for notification on commit.
    """

    __slots__ = ()

    # Type hints for mixed-in attributes
    _storage: RocksDBStorage
    _require_active: Any  # Method from ContextBase
    _rdbpy_txn: Any
    _modified_keys: set[Key]  # Initialized in __init__

    def put(self, key: Key, value: Value) -> None:
        """Put key-value pair.

        Args:
            key: Key to set
            value: Value to store

        Raises:
            StorageOperationError: If write fails
            StorageTransactionConflictError: If the key moved after the
                transaction's snapshot (another writer bypassed the slot)
            StorageClosedError: If context is closed
        """
        self._require_active()
        codec = self._storage.codec

        try:
            encoded_key = codec.encode_key(key)
            encoded_value = codec.encode_value(value)
        except Exception as e:
            raise StorageOperationError(f"Failed to encode key/value for {key}: {e}") from e

        try:
            self._rdbpy_txn.put(encoded_key, encoded_value)
        except Exception as e:
            raise classify_rdbpy(e, f"Failed to put key {key}") from e

        self._modified_keys.add(key)

    def delete(self, key: Key) -> None:
        """Delete key (idempotent).

        Silent if key doesn't exist (never raises for missing keys), and
        notifies only when something was there to delete.

        Args:
            key: Key to delete

        Raises:
            StorageOperationError: If deletion fails
            StorageClosedError: If context is closed
        """
        self._require_active()
        codec = self._storage.codec

        try:
            encoded_key = codec.encode_key(key)
        except Exception as e:
            raise StorageOperationError(f"Failed to encode key {key}: {e}") from e

        try:
            if self._rdbpy_txn.get(encoded_key) is None:
                return
            self._rdbpy_txn.delete(encoded_key)
        except Exception as e:
            raise classify_rdbpy(e, f"Failed to delete key {key}") from e

        self._modified_keys.add(key)
