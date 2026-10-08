"""The in-process writer slot of a SQLite storage, taken ahead of a transaction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from virtuals.tkv.storage import StorageError


if TYPE_CHECKING:
    import threading

    from .storage import SQLiteStorage


__all__ = ["SQLiteWriteSlot"]


class SQLiteWriteSlot:
    """The storage's writer slot, held before any transaction is begun.

    Returned by ``SQLiteStorage.areserve_write_slot``, which waits for the
    slot without blocking the event loop. Passing it to
    ``begin_transaction(write_slot=...)`` hands it over: the transaction
    takes the slot as already held, on any thread, and releases it on
    commit or abort as usual. A slot no transaction took is given back with
    ``release``, which is idempotent and does nothing once a transaction
    owns the slot.

    If beginning the transaction fails, the slot returns to the holder, so
    the holder's ``release`` still frees it.
    """

    __slots__ = ("_lock", "_state", "_storage")

    def __init__(self, storage: SQLiteStorage, lock: threading.Lock) -> None:
        """Initialize a held slot.

        Args:
            storage: The storage whose writer slot this is.
            lock: The writer lock acquired for it. Kept so a slot taken
                before a fork never releases the child's fresh lock.
        """
        self._storage = storage
        self._lock = lock
        self._state: Literal["held", "taken", "released"] = "held"

    @property
    def held(self) -> bool:
        """True while the slot is held and no transaction has taken it."""
        return self._state == "held"

    def release(self) -> None:
        """Give the slot back if no transaction took it."""
        if self._state != "held":
            return
        self._state = "released"
        self._storage._release_write_slot(self._lock)

    def _take(self, storage: SQLiteStorage) -> None:
        """Hand the slot to a transaction being begun on ``storage``."""
        if storage is not self._storage:
            raise StorageError("Write slot belongs to a different SQLite storage")
        if self._state != "held":
            raise StorageError(f"Write slot is already {self._state}")
        self._state = "taken"

    def _give_back(self) -> None:
        """Return the slot to its holder after a failed begin."""
        if self._state == "taken":
            self._state = "held"
