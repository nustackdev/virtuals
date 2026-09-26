"""Base context classes and operation mixins for SQLite storage."""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING, Any

from virtuals.tkv.storage import (
    ScanProtocol,
    StorageClosedError,
    StorageLockTimeoutError,
    StorageOperationError,
    StorageScanOptions,
)
from virtuals.tkv.types import EMPTY, Empty


if TYPE_CHECKING:
    from virtuals.tkv.types import Key, Value

    from .storage import SQLiteStorage


__all__ = [
    "ContextBase",
    "ReadOperationsMixin",
    "WriteOperationsMixin",
    "is_busy_error",
]


def is_busy_error(e: BaseException) -> bool:
    """True if ``e`` is SQLite reporting SQLITE_BUSY / SQLITE_LOCKED.

    Python 3.10 has no ``sqlite_errorcode`` on the exception, so match the
    message SQLite puts on both codes.
    """
    if not isinstance(e, sqlite3.OperationalError):
        return False
    msg = str(e).lower()
    return "locked" in msg or "busy" in msg


class ContextBase:
    """Base class for SQLite transaction/snapshot contexts.

    Owns one pooled connection for its whole life and hands it back to the
    storage when it closes. A context is used by one thread at a time, but
    may move between threads (begun on the event loop, finished in an
    executor), which is why pool connections are opened with
    ``check_same_thread=False``.
    """

    __slots__ = ("_closed", "_conn", "_holds_write_lock", "_storage")

    def __init__(
        self,
        storage: SQLiteStorage,
        conn: sqlite3.Connection | None,
        *,
        holds_write_lock: bool = False,
    ) -> None:
        """Initialize context.

        Args:
            storage: Parent storage instance.
            conn: Pooled connection with a transaction already begun, or
                None for contexts that take one only when they write.
            holds_write_lock: True if this context holds the storage's
                in-process writer slot and must release it on close.
        """
        self._storage = storage
        self._conn = conn
        self._holds_write_lock = holds_write_lock
        self._closed = False

    @property
    def storage(self) -> SQLiteStorage:
        """Get the storage instance."""
        return self._storage

    def _require_active(self) -> sqlite3.Connection:
        """Validate context is active and return its connection.

        Raises:
            StorageClosedError: If context is closed or connection missing.
        """
        if self._closed:
            raise StorageClosedError("Context is closed")
        if self._conn is None:
            raise StorageClosedError("Context handle is invalid")
        return self._conn

    def _end(self, sql: str) -> None:
        """Run COMMIT or ROLLBACK, then give back the connection and writer slot.

        The connection and slot are released even when the statement fails,
        so a failed commit never strands the writer lock. On a failed commit
        the pool rolls the connection back before reusing it.
        """
        conn = self._conn
        try:
            if conn is not None:
                conn.execute(sql)
        finally:
            self._closed = True
            self._conn = None
            if conn is not None:
                self._storage._release_connection(conn)
            if self._holds_write_lock:
                self._holds_write_lock = False
                self._storage._release_write_slot()

    @property
    def is_closed(self) -> bool:
        """Check if context is closed."""
        return self._closed

    @property
    def is_active(self) -> bool:
        """Check if context is active."""
        return not self._closed


class ReadOperationsMixin:
    """Mixin providing read operations for SQLite contexts."""

    __slots__ = ()

    # Type hints for mixed-in attributes
    _storage: SQLiteStorage

    _require_active: Any  # Method from ContextBase

    def get(self, key: Key) -> Value | Empty:
        """Get value by key.

        Args:
            key: Key to retrieve.

        Returns:
            Value at key, or EMPTY if key not found.
        """
        conn = self._require_active()
        codec = self._storage.codec

        try:
            encoded_key = codec.encode_key(key)
        except Exception as e:
            raise StorageOperationError(f"Failed to encode key {key}: {e}") from e

        try:
            row = conn.execute("SELECT v FROM kv WHERE k = ?", (encoded_key,)).fetchone()
        except Exception as e:
            raise StorageOperationError(f"Failed to get key {key}: {e}") from e

        if row is None:
            return EMPTY

        try:
            return codec.decode_value(row[0])
        except Exception as e:
            raise StorageOperationError(f"Failed to decode value for key {key}: {e}") from e

    def exists(self, key: Key) -> bool:
        """Check if key exists."""
        conn = self._require_active()
        codec = self._storage.codec

        try:
            encoded_key = codec.encode_key(key)
        except Exception as e:
            raise StorageOperationError(f"Failed to encode key {key}: {e}") from e

        try:
            row = conn.execute("SELECT 1 FROM kv WHERE k = ?", (encoded_key,)).fetchone()
        except Exception as e:
            raise StorageOperationError(f"Failed to check key {key}: {e}") from e
        return row is not None

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
        from .scan import SQLiteScan

        self._require_active()
        return SQLiteScan(self, options)  # type: ignore[arg-type]


class WriteOperationsMixin:
    """Mixin providing write operations for SQLite read-write transactions."""

    __slots__ = ()

    # Type hints for mixed-in attributes
    _storage: SQLiteStorage
    _modified_keys: set[Key]  # Initialized in __init__

    _require_active: Any  # Method from ContextBase

    def put(self, key: Key, value: Value) -> None:
        """Put key-value pair."""
        conn = self._require_active()
        codec = self._storage.codec

        try:
            encoded_key = codec.encode_key(key)
            encoded_value = codec.encode_value(value)
        except Exception as e:
            raise StorageOperationError(f"Failed to encode key/value for {key}: {e}") from e

        try:
            conn.execute(
                "INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?)", (encoded_key, encoded_value)
            )
        except Exception as e:
            if is_busy_error(e):
                raise StorageLockTimeoutError(f"Timed out writing key {key}: {e}") from e
            raise StorageOperationError(f"Failed to put key {key}: {e}") from e

        self._modified_keys.add(key)

    def delete(self, key: Key) -> None:
        """Delete key (idempotent; no-op on missing key)."""
        conn = self._require_active()
        codec = self._storage.codec

        try:
            encoded_key = codec.encode_key(key)
        except Exception as e:
            raise StorageOperationError(f"Failed to encode key {key}: {e}") from e

        try:
            cursor = conn.execute("DELETE FROM kv WHERE k = ?", (encoded_key,))
        except Exception as e:
            if is_busy_error(e):
                raise StorageLockTimeoutError(f"Timed out deleting key {key}: {e}") from e
            raise StorageOperationError(f"Failed to delete key {key}: {e}") from e

        if cursor.rowcount > 0:
            self._modified_keys.add(key)
