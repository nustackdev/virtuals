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

Lazy begin. A write transaction's LMDB handle is begun on the writer thread
only when it is first needed: at the first read, or at commit. Until then
its puts and deletes collect in ``_pending`` (last op per key wins) and
``replay_pending`` applies them in order to the begun handle. The view is
the same either way, since the writer slot and the cross-process gate are
held from ``begin_transaction`` on, so no peer commits in between.
"""

from __future__ import annotations

import os
import threading
import time
from logging import getLogger
from typing import TYPE_CHECKING, Any

import lmdb

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

# Readers in a tight loop of short py-lmdb calls release and retake the GIL
# so often that a thread woken to take it (the writer thread after a job
# lands on its queue or after each of its C calls, or the requester after
# its result) loses the race every time and never times out into a drop
# request (CPython's convoy effect, bpo-7946). Two tiers, both only while
# a writer job is in flight: past ``WRITER_STARVED_AFTER`` each reader
# sleeps ``WRITER_YIELD`` once per ``WRITER_YIELD_EVERY`` of its own time
# (a nudge: at most a fifth of its throughput, and nothing during a commit
# that is merely fsyncing); past ``WRITER_BLOCKED_AFTER`` it stays off the
# GIL in ``WRITER_YIELD`` steps until the job is through, at most
# ``WRITER_BACKOFF_STEPS`` per call, so a writer stuck for another reason
# (behind a writer outside virtuals) slows readers down but never holds
# them.
WRITER_STARVED_AFTER = 0.001
WRITER_YIELD_EVERY = 0.001
WRITER_BLOCKED_AFTER = 0.01
WRITER_BACKOFF_STEPS = 10
WRITER_YIELD = 0.0002


def yield_to_writer(writer_thread: Any, yielded_at: float) -> float:  # noqa: ANN401
    """Give a starved writer job the GIL; see the tiers above.

    Args:
        writer_thread: The storage's ``WriterThread``.
        yielded_at: When this caller last yielded.

    Returns:
        The new ``yielded_at``.
    """
    now = time.monotonic()
    wait_since = writer_thread._wait_since
    if wait_since:
        age = now - wait_since
        if age > WRITER_BLOCKED_AFTER:
            for _ in range(WRITER_BACKOFF_STEPS):
                time.sleep(WRITER_YIELD)
                wait_since = writer_thread._wait_since
                now = time.monotonic()
                if not wait_since or now - wait_since <= WRITER_BLOCKED_AFTER:
                    break
            return now
    else:
        gate_since = writer_thread._gate_since
        if not gate_since:
            return yielded_at
        age = now - gate_since
    if age > WRITER_STARVED_AFTER and now - yielded_at > WRITER_YIELD_EVERY:
        time.sleep(WRITER_YIELD)
        return now
    return yielded_at


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
    "replay_pending",
]


logger = getLogger(__name__)


# Encoded key -> (key, encoded value), or (key, None) for a delete.
Pending = dict[bytes, "tuple[Key, bytes | None]"]


def replay_pending(lmdb_txn: Any, pending: Pending) -> set[bytes]:  # noqa: ANN401
    """Apply buffered writes to a begun LMDB write txn.

    Returns the encoded keys whose delete found nothing, so the caller can
    leave them out of the keys it notifies as modified. Errors propagate
    as raised by py-lmdb (a full map, most likely).
    """
    put = lmdb_txn.put
    delete = lmdb_txn.delete
    missing: set[bytes] = set()
    for encoded_key, (_, encoded_value) in pending.items():
        if encoded_value is None:
            if not delete(encoded_key):
                missing.add(encoded_key)
        else:
            put(encoded_key, encoded_value)
    return missing


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
        "_yielded_at",
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
        self._yielded_at = 0.0  # see yield_to_writer

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

    def _forget(self) -> Any:  # noqa: ANN401
        """Mark the context closed without touching LMDB; returns its handle.

        For a forked child: the handle is the parent's. The caller parks
        it so it is never aborted or collected here.
        """
        handle, self._lmdb_txn = self._lmdb_txn, None
        self._closed = True
        self._holds_write_slot = False
        return handle

    def _finalize_unended_write(self, what: str) -> None:
        """Last resort for a write context dropped without commit or abort.

        Queues the abort of its LMDB transaction on the writer thread
        (safe from the collecting thread, whichever it is) and gives the
        slot back at once, so a forgotten transaction cannot hold the
        writer lock until close(). Takes no lock and waits on nothing:
        nothing else references the context any more, and a collection
        may run on a thread that holds a lock the abort needs. Inherited
        across a fork, the context is left alone (see ``_check_fork``).
        """
        if self._closed or self._pid != os.getpid():
            return
        txn, self._lmdb_txn = self._lmdb_txn, None
        self._closed = True
        holds, self._holds_write_slot = self._holds_write_slot, False
        logger.warning(f"LMDB {what} dropped without commit or abort; aborting it")
        try:
            # Skipped when the writer thread is gone or the interpreter is
            # exiting: process exit frees LMDB's writer lock and the gate
            # anyway, and close() ends every live write itself.
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
    _yielded_at: float

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

        writer_thread = self._storage._writer_thread
        if writer_thread._wait_since or writer_thread._gate_since:
            self._yielded_at = yield_to_writer(writer_thread, self._yielded_at)
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

        writer_thread = self._storage._writer_thread
        if writer_thread._wait_since or writer_thread._gate_since:
            self._yielded_at = yield_to_writer(writer_thread, self._yielded_at)
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

        with self._call_lock:
            self._require_active()  # begins a pending write transaction
        return LMDBScan(self, options)  # type: ignore[arg-type]


class WriteOperationsMixin:
    """Write operations for the LMDB transaction.

    While the transaction is pending (its LMDB handle not yet begun) puts
    and deletes collect in ``_pending`` on the caller's thread; once it is
    begun they go to the handle directly, on the caller's thread as well.
    """

    __slots__ = ()

    # Type hints for mixed-in attributes
    _storage: LMDBStorage
    _call_lock: threading.Lock
    _closed: bool
    _pending: Pending | None
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
            if self._closed:
                raise StorageClosedError("Context is closed")
            pending = self._pending
            if pending is not None:
                pending[encoded_key] = (key, encoded_value)
            else:
                txn = self._require_active()
                try:
                    txn.put(encoded_key, encoded_value)
                except lmdb.MapFullError as e:
                    # The transaction is dead either way (LMDB flags it on
                    # a failed allocation); retryable when a dead reader's
                    # slot was what pinned the map full.
                    retry = self._storage._after_map_full(f"put key {key}")
                    if retry is not None:
                        raise retry from e
                    raise StorageOperationError(f"Failed to put key {key}: {e}") from e
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
            if self._closed:
                raise StorageClosedError("Context is closed")
            pending = self._pending
            if pending is not None:
                pending[encoded_key] = (key, None)
                existed = True  # settled when the pending writes are replayed
            else:
                txn = self._require_active()
                try:
                    existed = txn.delete(encoded_key)
                except lmdb.MapFullError as e:
                    retry = self._storage._after_map_full(f"delete key {key}")
                    if retry is not None:
                        raise retry from e
                    raise StorageOperationError(f"Failed to delete key {key}: {e}") from e
                except Exception as e:
                    raise StorageOperationError(f"Failed to delete key {key}: {e}") from e

        if existed:
            self._modified_keys.add(key)
