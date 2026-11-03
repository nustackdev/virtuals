"""Write-only batch for LMDB storage."""

from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING

from virtuals.tkv.storage import (
    StorageClosedError,
    StorageLockTimeoutError,
    StorageNestedTransactionError,
    StorageOperationError,
    StorageTransactionConflictError,
    StorageTransactionError,
    WriteBatchProtocol,
)

from .context import ContextBase


if TYPE_CHECKING:
    from collections.abc import Iterable
    from types import TracebackType

    from virtuals.tkv.types import Key, Value

    from .context import Pending
    from .storage import LMDBStorage


__all__ = ["LMDBWriteBatch"]


logger = getLogger(__name__)


class LMDBWriteBatch(ContextBase, WriteBatchProtocol):
    """Write-only batch, buffered in Python and applied in one write transaction.

    Like the SQLite and RocksDB batches, puts and deletes collect in memory
    (last op per key wins) and touch LMDB only at ``write``, which takes
    the writer slot and the cross-process gate, then begins, replays and
    commits in one writer-thread job. Building a batch holds no lock, so a
    slow build keeps no writer waiting, in this process or another.
    """

    __slots__ = ("_aborted", "_interrupted", "_modified_keys", "_ops", "_written")

    def __init__(self, storage: LMDBStorage) -> None:
        """Initialize write batch."""
        super().__init__(storage, None)
        self._ops: Pending = {}
        self._modified_keys: set[Key] = set()
        self._written = False
        self._aborted = False
        self._interrupted = False  # a write was left by an interrupt with its outcome open

    @property
    def writable(self) -> bool:
        """Always True for write batches."""
        return True

    def put(self, key: Key, value: Value) -> None:
        """Queue a put."""
        codec = self._storage.codec
        try:
            encoded_key = codec.encode_key(key)
            encoded_value = codec.encode_value(value)
        except Exception as e:
            raise StorageOperationError(f"Failed to encode key/value for {key}: {e}") from e
        with self._call_lock:  # a put racing write() from another thread waits for it
            if self._closed:
                raise StorageClosedError("Context is closed")
            self._storage._check_key(encoded_key, key)
            self._ops[encoded_key] = (key, encoded_value)
        self._modified_keys.add(key)

    def delete(self, key: Key) -> None:
        """Queue a delete (idempotent; no-op on missing key)."""
        codec = self._storage.codec
        try:
            encoded_key = codec.encode_key(key)
        except Exception as e:
            raise StorageOperationError(f"Failed to encode key {key}: {e}") from e
        with self._call_lock:
            if self._closed:
                raise StorageClosedError("Context is closed")
            if not self._storage._key_fits(encoded_key):
                return  # a key LMDB cannot store cannot exist: a no-op
            self._ops[encoded_key] = (key, None)
        self._modified_keys.add(key)

    def write(self) -> None:
        """Apply the batch in one write transaction and make it permanent.

        Waits for the writer slot and the gate outside the batch's call
        lock, so a ``close()`` aborting the batch meanwhile is not held up
        by that wait; once both are held the batch is checked again and
        applied under the lock.
        """
        with self._call_lock:
            if self._closed:
                logger.error("Cannot write, batch is closed")
                raise StorageClosedError("Write batch is closed")
            if self._written:
                logger.error("Cannot write, batch already written")
                raise StorageTransactionError("Write batch already written")
            if self._aborted:
                logger.error("Cannot write, batch already aborted")
                raise StorageTransactionError("Write batch already aborted")

        storage = self._storage
        try:
            if self._ops:
                lock = storage._take_write_slot()
                try:
                    with self._call_lock:
                        if self._closed:
                            raise StorageClosedError("Write batch was closed while writing")
                        ops = self._ops
                        # Interrupted (a signal) with the commit in flight,
                        # it runs to its end on the writer thread and
                        # ``_settle`` finishes from there; the interrupt
                        # propagates as is.
                        missing = storage._commit_pending(
                            ops, settle=lambda missing: self._settle(ops, missing, late=True)
                        )
                        self._settle(ops, missing)
                finally:
                    storage._release_write_slot(lock)
            else:
                with self._call_lock:
                    if self._closed:
                        raise StorageClosedError("Write batch was closed while writing")
                    self._finish()  # nothing to apply: just ends the batch
        except BaseException as e:
            if not isinstance(e, Exception):
                self._interrupted = True  # the abort that follows lets the write settle first
                raise
            with self._call_lock:
                if not self._closed:
                    self._aborted = True
                    self._ops = {}
                    self._mark_closed()
            storage._untrack_write_batch(self)
            logger.error("Write batch write failed")
            if isinstance(
                e,
                (
                    StorageLockTimeoutError,
                    StorageNestedTransactionError,
                    StorageClosedError,
                    StorageTransactionConflictError,
                ),
            ):
                raise
            raise StorageTransactionError(f"Failed to write batch: {e}") from e

        logger.debug("Write batch written")
        storage._notify_batch(self._modified_keys)
        storage._untrack_write_batch(self)

    def _finish(self) -> None:
        """Mark the batch written and closed. Under ``_call_lock``."""
        self._ops = {}
        self._written = True
        self._mark_closed()

    def _settle(self, ops: Pending, missing: Iterable[bytes], *, late: bool = False) -> None:
        """Bookkeeping of a write that landed: the keys and the flags.

        Under ``_call_lock`` on the writing thread, or ``late`` from the
        writer thread after the writing thread was interrupted: then it
        also notifies and untracks, since the interrupted ``write`` never
        got to, and the batch is marked written, which it is.
        """
        for encoded_key in missing:
            self._modified_keys.discard(ops[encoded_key][0])
        self._finish()
        if late:
            logger.warning("LMDB write batch written after its write was interrupted")
            self._storage._notify_batch(self._modified_keys)
            self._storage._untrack_write_batch(self)

    def abort(self) -> None:
        """Abort write batch and discard its buffered writes.

        Safe from any thread: a batch never holds the slot or an LMDB
        transaction outside ``write``, and an abort racing a write waits
        for it and is then a no-op.
        """
        if self._closed:
            logger.debug("Abort called on closed write batch")
            return
        with self._call_lock:
            if self._interrupted:
                # A write left by an interrupt may still be landing on the
                # writer thread; it settles there, ahead of any job queued
                # here, so after this the batch is written or not.
                self._storage._drain_writer()
            if self._closed:
                return
            self._aborted = True
            self._ops = {}
            self._mark_closed()
            logger.debug("Write batch aborted")

        self._storage._untrack_write_batch(self)

    def __enter__(self) -> LMDBWriteBatch:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager - auto write or abort."""
        if exc_type is not None:
            try:
                self.abort()
            except Exception:
                logger.error("Write batch abort failed")
        else:
            if not self._written and not self._aborted:
                self.write()
