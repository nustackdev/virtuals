"""The in-process writer slot of a SQLite storage, taken ahead of a transaction."""

from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING

from virtuals.tkv.storage import StorageError

from .._write_slot import WriteSlot


if TYPE_CHECKING:
    import sqlite3

    from .storage import SQLiteStorage


__all__ = ["SQLiteWriteSlot"]


class SQLiteWriteSlot(WriteSlot):
    """Returned by ``SQLiteStorage.areserve_write_slot``; see ``WriteSlot``.

    On top of the in-process slot it carries a pooled connection on which
    ``BEGIN IMMEDIATE`` has already run: the reservation waited for the
    cross-process file lock too, without blocking the event loop. The
    transaction that takes the slot takes that connection with it. A slot
    no transaction takes gives the connection back (rolled back) on
    ``release``, so the file lock is never held by an idle slot.

    The connection belongs to the process that reserved the slot. In a
    forked child it is parked, never rolled back or closed, and the slot
    refuses to be taken (the child's writer slot is a fresh one).
    """

    __slots__ = ("__weakref__", "_conn", "_conn_lock", "_pid")

    def __init__(
        self, storage: SQLiteStorage, lock: threading.Lock, conn: sqlite3.Connection
    ) -> None:
        """Initialize a held slot.

        Args:
            storage: The storage whose writer slot this is.
            lock: The writer lock acquired for it.
            conn: A pooled connection with ``BEGIN IMMEDIATE`` run.
        """
        super().__init__(storage, lock)
        self._conn: sqlite3.Connection | None = conn
        # Guards taking the connection: close() may detach it from another
        # thread while the holder begins a transaction with it.
        self._conn_lock = threading.Lock()
        self._pid = os.getpid()

    def _detach_conn(self) -> sqlite3.Connection | None:
        """Take the begun connection out of the slot, once."""
        with self._conn_lock:
            conn, self._conn = self._conn, None
        return conn

    def _drop_conn(self) -> None:
        """Give the begun connection back, or park it in a forked child."""
        conn = self._detach_conn()
        if conn is None:
            return
        storage: SQLiteStorage = self._storage  # type: ignore[assignment]
        if os.getpid() != self._pid:
            storage._park_inherited([conn])
        else:
            storage._release_connection(conn)

    def release(self) -> None:
        """Give the slot back if no transaction took it."""
        if self._state != "held":
            return
        self._drop_conn()
        super().release()

    def __del__(self) -> None:
        """Last resort: free a held slot that became unreachable unreleased.

        Takes no storage lock: a collection can run on a thread that holds
        the storage state lock (e.g. inside the pool), so the connection
        is closed outright instead of being returned to the pool (SQLite
        rolls the begun transaction back on close), or parked in a forked
        child. The in-process lock is then released by identity as usual.

        In a forked child the slot is the parent's: the connection is
        parked and the writer lock is left alone. Releasing it before the
        child's first storage call ran ``_check_fork`` would make the
        inherited lock read as free, so the child would not learn that the
        parent held the file's write lock at the fork.
        """
        try:
            if getattr(self, "_state", "released") != "held":
                return
            conn = self._detach_conn()
            if os.getpid() != self._pid:
                self._state = "released"
                if conn is not None:
                    self._storage._park_inherited([conn])  # type: ignore[attr-defined]
                return
            if conn is not None:
                conn.close()
            self.release()
        except Exception:  # noqa: S110 (interpreter shutdown, half-built object)
            pass

    def _take(self, storage: object) -> None:
        """Hand the slot to a transaction being begun on ``storage``.

        Refuses a stale slot: one whose lock is no longer the storage's
        writer lock because the storage was closed, or the process forked,
        since it was reserved. Its begun connection is given back (or, in
        a forked child, parked: it is the parent's) before the refusal.
        """
        if self._state == "held" and self._lock is not self._storage._writer.lock:  # type: ignore[attr-defined]
            self._state = "released"
            self._drop_conn()
            raise StorageError(
                "Write slot is stale: the storage was closed (or the process forked) after "
                "it was reserved. Reserve a new one"
            )
        super()._take(storage)
