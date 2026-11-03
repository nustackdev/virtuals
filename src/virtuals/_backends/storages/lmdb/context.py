"""Base context classes and operation mixins for LMDB storage.

Thread affinity. LMDB documents that a write transaction and its cursors
belong to one thread. In liblmdb 0.9.36 (py-lmdb 3.0.0, which opens every
environment with MDB_NOTLS) the only thread-bound state of a write
transaction is the writer mutex, taken in begin and released in commit or
abort; ``put``, ``get``, ``del`` and cursor calls read and write the
transaction's own memory with no per-thread lookup. So begin, commit and
abort run on the storage's writer thread, and the data calls run on the
caller's thread, serialized per context by ``_call_lock``, with the future
handoffs giving the needed happens-before. Re-check this against liblmdb
on any bump (``mdb_txn_renew0``, ``mdb_txn_end``, and that no data path
touches ``me_txkey``-style thread-local state for write transactions).
"""

from __future__ import annotations

import os
import threading
from logging import getLogger
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


# Guards the check-and-clear of ``_holds_write_slot``: close() may abort a
# context from another thread while its owner commits it.
_SLOT_FLAG_LOCK = threading.Lock()


def _after_fork_in_child() -> None:
    """A fresh flag lock in the child: a parent thread may hold the inherited one."""
    global _SLOT_FLAG_LOCK
    _SLOT_FLAG_LOCK = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)


__all__ = [
    "ContextBase",
    "ReadOperationsMixin",
    "WriteOperationsMixin",
]


logger = getLogger(__name__)


class ContextBase:
    """Base class for LMDB transaction/snapshot contexts.

    Wraps an lmdb.Transaction handle and tracks open/closed state.

    Every call into the handle runs under ``_call_lock``, one call at a
    time per context, and so does ending it. ``close()`` may abort a
    context from another thread: it then waits for the owner's call in
    flight instead of freeing the handle under it, and the owner's next
    call sees a closed context. The lock is never held across a scan's
    yields, so a scanning owner never keeps close() waiting.
    """

    __slots__ = (
        "__weakref__",
        "_call_lock",
        "_closed",
        "_holds_write_slot",
        "_lmdb_txn",
        "_pid",
        "_storage",
        "_write_lock",
    )

    def __init__(
        self,
        storage: LMDBStorage,
        lmdb_txn: Any,  # noqa: ANN401
        *,
        holds_write_slot: bool = False,
        write_lock: threading.Lock | None = None,
    ) -> None:
        """Initialize context with storage reference and LMDB transaction.

        Args:
            storage: Parent storage instance
            lmdb_txn: LMDB transaction handle
            holds_write_slot: True if this context holds the storage's
                in-process writer slot and must release it on close.
            write_lock: The slot's lock it holds, released by identity so
                a late end never frees a lock the slot has since replaced.
        """
        self._storage = storage
        self._lmdb_txn: Any | None = lmdb_txn
        self._holds_write_slot = holds_write_slot
        self._write_lock = write_lock
        self._closed = False
        self._call_lock = threading.Lock()
        self._pid = os.getpid()

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
        """Mark context as closed, clear handle, and give back the writer slot."""
        self._closed = True
        self._lmdb_txn = None
        with _SLOT_FLAG_LOCK:
            holds, self._holds_write_slot = self._holds_write_slot, False
        if holds:
            self._storage._release_write_slot(self._write_lock)

    def _finalize_unended_write(self, what: str) -> None:
        """Last resort for a write context dropped without commit or abort.

        Aborts its LMDB transaction on the writer thread (safe from the
        collecting thread, whichever it is) and gives the slot back, so a
        forgotten transaction cannot hold the writer lock until close().
        Takes no lock: nothing else references the context any more, and a
        collection may run on a thread that holds one of them. Inherited
        across a fork, the context is left alone (see ``_check_fork``).
        """
        if self._closed or self._pid != os.getpid():
            return
        txn, self._lmdb_txn = self._lmdb_txn, None
        self._closed = True
        holds, self._holds_write_slot = self._holds_write_slot, False
        logger.warning(f"LMDB {what} dropped without commit or abort; aborting it")
        try:
            # Bounded, and skipped when the writer thread is gone or the
            # interpreter is exiting: a wait there would never end, and
            # process exit frees LMDB's writer lock and the gate anyway.
            if txn is not None and not self._storage._end_write_from_finalizer(txn):
                logger.debug(f"LMDB {what} finalizer did not reach the writer thread")
        finally:
            if holds:
                self._storage._release_write_slot(self._write_lock)

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
    _call_lock: threading.Lock
    _require_active: Any  # Method from ContextBase

    def get(self, key: Key) -> Value | Empty:
        """Get value by key.

        Args:
            key: Key to retrieve.

        Returns:
            Value at key, or EMPTY if key not found.
        """
        codec = self._storage.codec

        try:
            encoded_key = codec.encode_key(key)
        except Exception as e:
            raise StorageOperationError(f"Failed to encode key {key}: {e}") from e

        with self._call_lock:
            txn = self._require_active()
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
        codec = self._storage.codec

        try:
            encoded_key = codec.encode_key(key)
        except Exception as e:
            raise StorageOperationError(f"Failed to encode key {key}: {e}") from e

        with self._call_lock:
            txn = self._require_active()
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
    _call_lock: threading.Lock
    _require_active: Any  # Method from ContextBase
    _modified_keys: set[Key]  # Initialized in __init__

    def put(self, key: Key, value: Value) -> None:
        """Put key-value pair."""
        codec = self._storage.codec

        try:
            encoded_key = codec.encode_key(key)
            encoded_value = codec.encode_value(value)
        except Exception as e:
            raise StorageOperationError(f"Failed to encode key/value for {key}: {e}") from e

        with self._call_lock:
            txn = self._require_active()
            try:
                txn.put(encoded_key, encoded_value)
            except Exception as e:
                raise StorageOperationError(f"Failed to put key {key}: {e}") from e

        self._modified_keys.add(key)

    def delete(self, key: Key) -> None:
        """Delete key (idempotent; no-op on missing key)."""
        codec = self._storage.codec

        try:
            encoded_key = codec.encode_key(key)
        except Exception as e:
            raise StorageOperationError(f"Failed to encode key {key}: {e}") from e

        with self._call_lock:
            txn = self._require_active()
            try:
                existed = txn.delete(encoded_key)
            except Exception as e:
                raise StorageOperationError(f"Failed to delete key {key}: {e}") from e

        if existed:
            self._modified_keys.add(key)
