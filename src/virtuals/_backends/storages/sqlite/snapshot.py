"""Read-only snapshot for SQLite storage."""

from __future__ import annotations

from typing import TYPE_CHECKING

from virtuals.tkv.storage import StorageError

from .context import ContextBase, ReadOperationsMixin


if TYPE_CHECKING:
    import sqlite3
    from types import TracebackType

    from .storage import SQLiteStorage


__all__ = ["SQLiteSnapshot"]


class SQLiteSnapshot(ContextBase, ReadOperationsMixin):
    """Read-only snapshot backed by a deferred SQLite read transaction.

    In WAL mode a read transaction pins the database as of its first read
    and never blocks, or is blocked by, the writer. The storage runs that
    first read when the snapshot opens, so the view is fixed at
    ``begin_snapshot`` time rather than at the first ``get``.
    """

    __slots__ = ()

    def __init__(self, storage: SQLiteStorage, conn: sqlite3.Connection) -> None:
        """Initialize snapshot."""
        super().__init__(storage, conn)

    @property
    def writable(self) -> bool:
        """Always False for snapshots."""
        return False

    def close(self) -> None:
        """Close snapshot and release its connection."""
        if self._closed:
            return
        try:
            self._end("ROLLBACK")
        except Exception as e:
            raise StorageError(f"Failed to close snapshot: {e}") from e
        finally:
            self._storage._untrack_snapshot(self)

    def __enter__(self) -> SQLiteSnapshot:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager - auto close."""
        self.close()
