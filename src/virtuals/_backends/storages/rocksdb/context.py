"""Base context classes and operation mixins for RocksDB storage."""

from __future__ import annotations

import threading
import weakref
from typing import TYPE_CHECKING

import rdbpy

from virtuals._backends.storages._empty import empty_error
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
    "ScanHandle",
    "WriteOperationsMixin",
    "classify_rdbpy",
    "is_missing_file_error",
    "secondary_view_lost",
]


# Conflicts rocksdb reports on a write or a commit: the key moved after the
# transaction's snapshot (Busy, Deadlock is a Busy), the memtable history
# was too short to validate (TryAgain), or the transaction expired.
_CONFLICT = (rdbpy.Busy, rdbpy.TryAgain, rdbpy.Expired)
# A lock wait exceeded its timeout, or the lock table is full.
_LOCK_TIMEOUT = (rdbpy.TimedOut, rdbpy.LockLimit)

# rocksdb::Status::SubCode::kPathNotFound: an IO error on a file that is gone.
_SUBCODE_PATH_NOT_FOUND = 9

# Orders the one-shot end of a context between its owner, storage close on
# another thread and the finalizer, so the writer slot is given back once.
# Reentrant: a collection triggered inside the section may run a finalizer
# that ends another context on this same thread.
_END_LOCK = threading.RLock()


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


def secondary_view_lost(storage: RocksDBStorage, cause: BaseException) -> StorageError:
    """Catch the secondary up past the file it lost, and say the view is gone.

    Returns the retryable ``StorageTransactionConflictError`` for the
    caller to raise: a new snapshot reads the current state, this one
    cannot answer consistently any more. A failed catch-up is reported
    instead, non-retryable.
    """
    try:
        storage.force_catch_up_with_primary()
    except StorageError as e:
        return e
    return StorageTransactionConflictError(
        "Secondary snapshot lost its view: the primary compacted away a file it was reading, "
        f"and the storage caught up to read on. Open a new snapshot and re-read: {cause}"
    )


def is_missing_file_error(e: BaseException) -> bool:
    """True if a read failed because it touched a now-deleted SST file.

    This is the signature of a read-only secondary pinned to a manifest
    version older than the primary's current one: the primary compacted and
    deleted the file, but the secondary's view still references it. A
    catch-up to the current manifest drops the reference. rocksdb reports
    it as an IO error with the path-not-found subcode.
    """
    return isinstance(e, rdbpy.RocksIOError) and e.subcode == _SUBCODE_PATH_NOT_FOUND


class ScanHandle:
    """A scan's native iterator, closable by the context it was begun on.

    The scan generator keeps the handle in its frame, so the context's
    weak registry forgets it when the scan ends; the context closes it
    (freeing the iterator and the files it pins) when it ends first. The
    generator itself stays suspended and raises ``StorageClosedError`` if
    resumed, as before.

    Every native step runs under the handle's lock, and so does ``close``:
    a snapshot closed on another thread while a scan is inside
    ``next_batch`` waits for that step to return before the iterator is
    freed under it, and the scan's next step then sees it closed.
    """

    __slots__ = ("__weakref__", "_lock", "iterator")

    def __init__(self, iterator: Any) -> None:  # noqa: ANN401
        """Wrap a native rdbpy iterator."""
        self.iterator = iterator
        self._lock = threading.Lock()

    def step(self, method: str, *args: Any) -> Any:  # noqa: ANN401
        """Run one native call on the iterator, excluding ``close``.

        Raises:
            StorageClosedError: If the context closed the iterator since
        """
        with self._lock:
            if self.iterator.closed:
                raise StorageClosedError("Scan used after its context ended")
            return getattr(self.iterator, method)(*args)

    def close(self) -> None:
        """Close the native iterator once no step is inside it. Idempotent."""
        with self._lock:
            self.iterator.close()


class ContextBase:
    """Base class for RocksDB transaction, snapshot and batch contexts.

    Tracks open/closed state and, for write contexts, the storage's writer
    slot, which is given back when the context closes. Subclasses own
    their rdbpy handles and drop them in ``_drop_handles``. Scans begun on
    the context are closed with it, so a suspended scan does not keep a
    native iterator (and the files it pins) alive past the context.
    """

    __slots__ = ("__weakref__", "_closed", "_holds_write_slot", "_scans", "_storage", "_write_lock")

    def __init__(
        self,
        storage: RocksDBStorage,
        *,
        write_lock: threading.Lock | None = None,
    ) -> None:
        """Initialize context.

        Args:
            storage: Parent storage instance
            write_lock: The writer lock this context holds, if it holds the
                storage's in-process writer slot. It is given back on close,
                by identity: a lock the storage replaced since (close, fork)
                is not touched.
        """
        self._storage = storage
        self._write_lock = write_lock
        self._holds_write_slot = write_lock is not None
        self._scans: weakref.WeakSet[ScanHandle] | None = None
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

    def _register_scan(self, scan: ScanHandle) -> None:
        """Remember a scan's iterator, to close with the context."""
        with _END_LOCK:
            if self._scans is None:
                self._scans = weakref.WeakSet()
            self._scans.add(scan)

    def _close_scans(self) -> None:
        """Close every live scan iterator of this context, freeing what it pins.

        A scan on another thread that is inside a native step keeps its
        iterator until that step returns (``ScanHandle.close`` waits for
        it), so nothing is freed under a running call.
        """
        with _END_LOCK:
            scans = list(self._scans) if self._scans else ()
        for scan in scans:
            try:
                scan.close()
            except Exception:  # noqa: S110 (iterator already gone with its parent)
                pass

    def _mark_closed(self) -> None:
        """Mark context as closed, close its scans, drop its handles, give back the slot.

        Runs once for the slot however many callers race (the owner's end,
        storage close on another thread, the finalizer).
        """
        with _END_LOCK:
            self._closed = True
            holds, self._holds_write_slot = self._holds_write_slot, False
            lock, self._write_lock = self._write_lock, None
        self._close_scans()
        self._drop_handles()
        if holds:
            self._storage._release_write_slot(lock)

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

    def _read_raw(self, key: Key, action: str, encoded_key: bytes) -> bytes | None:
        """Point read at the context's snapshot, surfacing a stale-manifest secondary.

        On a read-only secondary, the read can fail because the secondary's
        manifest version references an SST the primary already compacted
        away. The storage then catches up to the current manifest, so the
        next snapshot reads, and this one raises the retryable
        ``StorageTransactionConflictError``: its view is gone, and
        answering from the new state would tear it (contract section 2).
        Any other failure, and any failure at all on a primary, is
        classified and raised as is.
        """
        try:
            return self._get_raw(encoded_key)
        except Exception as e:
            storage = self._storage
            if storage._is_secondary and is_missing_file_error(e):
                raise secondary_view_lost(storage, e) from e
            raise classify_rdbpy(e, f"Failed to {action} key {key}") from e

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

        encoded_value = self._read_raw(key, "get", encoded_key)

        if encoded_value is None:
            return EMPTY

        try:
            return codec.decode_value(encoded_value)
        except Exception as e:
            raise StorageOperationError(f"Failed to decode value for key {key}: {e}") from e

    def exists(self, key: Key) -> bool:
        """Check if key exists.

        A point get: it uses the bloom filters and costs about a get on
        the usual value sizes (memtable 0.4 to 1.3us, SST 0.3 to 3.4us,
        present or absent). It does read the value, so on a very large
        value it costs as much as a get of it; rdbpy exposes no
        ``KeyMayExist``, and a keys-only cursor seek, the alternative, is
        2 to 3x slower on small values and reads the neighbouring key's
        block when the key is absent.

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

        return self._read_raw(key, "check", encoded_key) is not None

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
        if value is EMPTY:
            raise empty_error(key)
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

        Silent if key doesn't exist (never raises for missing keys). The
        key is notified on commit whether or not something was there, as a
        batch delete does; a subscriber re-reads on wake either way.

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
            self._rdbpy_txn.delete(encoded_key)
        except Exception as e:
            raise classify_rdbpy(e, f"Failed to delete key {key}") from e

        self._modified_keys.add(key)
