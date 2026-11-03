"""The writer slot of an in-memory storage, taken ahead of a transaction.

Mirrors the SQLite slot (kept in step by hand until a shared one is
lifted), so code that waits for a slot by name (``areserve_write_slot``
then ``begin_transaction(write_slot=...)``) works on either storage
unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from virtuals.tkv.storage import StorageError


if TYPE_CHECKING:
    import threading

    from .storage import InMemoryStorage


__all__ = ["InMemoryWriteSlot"]


class InMemoryWriteSlot:
    """The storage's writer slot, held before any transaction is begun.

    Returned by ``InMemoryStorage.areserve_write_slot``, which waits for the
    slot without blocking the event loop. Passing it to
    ``begin_transaction(write_slot=...)`` hands it over: the transaction
    takes the slot as already held, on any thread, and releases it on
    commit or abort as usual. A slot no transaction took is given back with
    ``release``, which is idempotent and does nothing once a transaction
    owns the slot. If beginning the transaction fails, the slot returns to
    the holder, so the holder's ``release`` still frees it.
    """

    __slots__ = ("_lock", "_state", "_storage")

    def __init__(self, storage: InMemoryStorage, lock: threading.Lock) -> None:
        """Initialize a held slot over the writer lock acquired for it."""
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

    def _take(self, storage: InMemoryStorage) -> threading.Lock:
        """Hand the slot to a transaction being begun on ``storage``; returns its lock."""
        if storage is not self._storage:
            raise StorageError("Write slot belongs to a different in-memory storage")
        if self._state != "held":
            raise StorageError(f"Write slot is already {self._state}")
        if self._lock is not storage._write_lock:
            # close() replaced the slot since this one was reserved.
            self._state = "released"
            raise StorageError("Write slot is stale: the storage was closed after it was reserved")
        self._state = "taken"
        return self._lock

    def _give_back(self) -> None:
        """Return the slot to its holder after a failed begin."""
        if self._state == "taken":
            self._state = "held"
