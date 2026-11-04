"""Write-only batch for SQLite storage."""

from __future__ import annotations

import sqlite3
from logging import getLogger
from typing import TYPE_CHECKING

from virtuals._backends.storages._empty import empty_error
from virtuals.tkv.storage import (
    StorageClosedError,
    StorageError,
    StorageLockTimeoutError,
    StorageOperationError,
    StorageTransactionError,
    WriteBatchProtocol,
)
from virtuals.tkv.types import EMPTY

from .context import ContextBase, is_busy_error


if TYPE_CHECKING:
    from types import TracebackType

    from virtuals.tkv.types import Key, Value

    from .storage import SQLiteStorage


__all__ = ["SQLiteWriteBatch"]


logger = getLogger(__name__)


class SQLiteWriteBatch(ContextBase, WriteBatchProtocol):
    """Write-only batch, buffered in memory and applied in one transaction.

    Like the RocksDB batch, puts and deletes collect in Python and touch the
    database only at ``write``, which takes the write lock, applies them all
    with ``executemany`` and commits. Building a batch therefore holds no
    lock and no connection, and a large batch keeps other writers waiting
    only for the apply itself.
    """

    __slots__ = ("_aborted", "_modified_keys", "_ops", "_written")

    def __init__(self, storage: SQLiteStorage) -> None:
        """Initialize write batch."""
        super().__init__(storage, None)
        # encoded key -> encoded value, or None for a delete. Last op per key wins.
        self._ops: dict[object, object | None] = {}
        self._modified_keys: set[Key] = set()
        self._written = False
        self._aborted = False

    @property
    def writable(self) -> bool:
        """Always True for write batches."""
        return True

    def _require_open_batch(self) -> None:
        if self._closed:
            raise StorageClosedError("Context is closed")

    def put(self, key: Key, value: Value) -> None:
        """Queue a put."""
        self._require_open_batch()
        if value is EMPTY:
            raise empty_error(key)
        codec = self._storage.codec
        try:
            encoded_key = codec.encode_key(key)
            encoded_value = codec.encode_value(value)
        except Exception as e:
            raise StorageOperationError(f"Failed to encode key/value for {key}: {e}") from e
        self._ops[encoded_key] = encoded_value
        self._modified_keys.add(key)

    def delete(self, key: Key) -> None:
        """Queue a delete (idempotent; no-op on missing key)."""
        self._require_open_batch()
        codec = self._storage.codec
        try:
            encoded_key = codec.encode_key(key)
        except Exception as e:
            raise StorageOperationError(f"Failed to encode key {key}: {e}") from e
        self._ops[encoded_key] = None
        self._modified_keys.add(key)

    def write(self) -> None:
        """Apply the batch in one write transaction and make it permanent.

        A batch already written or aborted raises ``StorageClosedError``.
        """
        if self._closed:
            logger.error("Cannot write, batch is closed")
            raise StorageClosedError("Write batch is closed")

        puts = [(k, v) for k, v in self._ops.items() if v is not None]
        deletes = [(k,) for k, v in self._ops.items() if v is None]

        # An interrupt while the COMMIT is in flight never changes its
        # outcome or how it is reported (see ``SQLiteTransaction.commit``).
        deferred: list[BaseException] = []
        self._write_held(puts, deletes, deferred)

    def _write_held(self, puts: list, deletes: list, deferred: list[BaseException]) -> None:
        """``write`` with signals held."""
        ran: bool | None = True
        try:
            if not puts and not deletes:
                self._end("COMMIT")  # nothing to apply: just ends the batch
            else:
                # The lock wait at begin happens outside the context lock, so
                # close() aborting this batch meanwhile is not held up by it.
                conn, writer_lock = self._storage._begin_write_connection()
                with self._lock:
                    if self._closed or not self._storage._opened:
                        self._storage._release_connection(conn)
                        self._storage._release_write_slot(writer_lock)
                        raise StorageClosedError("Write batch was closed while writing")
                    self._conn = conn
                    self._writer_lock = writer_lock
                    self._holds_write_lock = True
                    if deletes:
                        conn.executemany("DELETE FROM kv WHERE k = ?", deletes)
                    if puts:
                        conn.executemany("INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?)", puts)
                    ran = self._end_locked("COMMIT", deferred)
        except Exception as e:
            self._aborted = True
            if not self._closed:
                self._end("ROLLBACK")
            self._ops.clear()
            self._storage._untrack_write_batch(self)
            logger.error("Write batch write failed")
            if deferred:
                raise deferred[0] from e  # the interrupt, with the failure as its cause
            if not isinstance(e, sqlite3.Error) or isinstance(e, StorageError):
                raise  # storage errors, and interrupts, as they are
            if is_busy_error(e):
                raise StorageLockTimeoutError(f"Timed out writing batch: {e}") from e
            raise StorageTransactionError(f"Failed to write batch: {e}") from e

        if not ran:  # the interrupt came before the COMMIT: rolled back, as an abort
            self._aborted = True
            self._ops.clear()
            self._storage._untrack_write_batch(self)
            raise deferred[0]

        logger.debug("Write batch written")

        self._written = True
        self._ops.clear()
        try:
            self._storage._notify_batch(self._modified_keys)
        finally:
            self._storage._untrack_write_batch(self)
            if deferred:
                raise deferred[0]

    def abort(self) -> None:
        """Abort write batch and discard changes."""
        if self._closed:
            logger.debug("Abort called on closed write batch")
            return
        self._aborted = True
        # Goes through the context lock: a write in flight on another thread
        # finishes first, and an apply begun but not committed is rolled back.
        try:
            self._end("ROLLBACK")
        except Exception as e:
            logger.error(f"Write batch abort failed: {e}")
        self._ops.clear()
        self._storage._untrack_write_batch(self)
        logger.debug("Write batch aborted")

    def __enter__(self) -> SQLiteWriteBatch:
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
