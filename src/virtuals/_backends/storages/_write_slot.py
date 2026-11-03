"""The in-process writer slot shared by single-writer storages (SQLite, LMDB).

Both backends allow one write transaction per database at a time. Writers
in one process queue on this slot before they reach the backend, so threads
hand off directly and coroutines wait without blocking their event loop.
"""

from __future__ import annotations

import asyncio
import threading
import time
import weakref
from typing import TYPE_CHECKING, Literal, Protocol

from virtuals.tkv.storage import (
    StorageError,
    StorageLockTimeoutError,
    StorageNestedTransactionError,
)


if TYPE_CHECKING:
    from collections.abc import Callable


__all__ = ["WriteSlot", "WriterLock"]


# Poll interval bounds for a coroutine waiting on the writer slot (seconds).
_SLOT_POLL_MIN = 0.0005
_SLOT_POLL_MAX = 0.005

_Owner = tuple[int, "weakref.ref[asyncio.Task] | None"]


def _caller() -> _Owner:
    """Who is asking for the writer slot: this thread, and its running task if any.

    The task is held weakly: a task cancelled while holding a reserved slot
    keeps the slot alive through its traceback, and a strong reference here
    would keep that task alive in turn, so the slot could never be collected.
    """
    try:
        task = asyncio.current_task()
    except RuntimeError:  # no running event loop on this thread
        task = None
    return threading.get_ident(), None if task is None else weakref.ref(task)


class _SlotStorage(Protocol):
    def _release_write_slot(self, lock: threading.Lock | None = None) -> None: ...


class WriterLock:
    """A storage's writer slot: one in-process writer at a time.

    The slot belongs to a thread and, on an event loop, to the task that
    took it. It is a plain Lock (not an RLock), so it may be released from
    a different thread than the one that took it.
    """

    __slots__ = ("_label", "_lock", "_owner")

    def __init__(self, label: str) -> None:
        """Initialize a free slot.

        Args:
            label: Backend name used in error messages, e.g. ``"SQLite"``.
        """
        self._label = label
        self._lock = threading.Lock()
        self._owner: _Owner | None = None

    @property
    def lock(self) -> threading.Lock:
        """The current underlying lock (replaced by ``reset``)."""
        return self._lock

    def reset(self) -> None:
        """Forget any holder, e.g. in a forked child where it can never release."""
        self._lock = threading.Lock()
        self._owner = None

    def _nested(self) -> StorageNestedTransactionError:
        return StorageNestedTransactionError(
            f"This task or thread already holds the write lock on this {self._label} "
            "storage; a second write transaction here would wait on itself"
        )

    def _timed_out(self, timeout: float) -> StorageLockTimeoutError:
        return StorageLockTimeoutError(f"Timed out after {timeout}s waiting for the write lock")

    def acquire(self, timeout: float) -> None:
        """Block for the slot, up to ``timeout`` seconds.

        Fails at once when the slot is held from this same thread: by this
        task or thread with ``StorageNestedTransactionError``, or by another
        coroutine on this thread's loop with ``StorageLockTimeoutError``,
        since that holder cannot run to release it while this thread blocks.
        """
        me = _caller()
        owner = self._owner
        if owner is not None and owner[0] == me[0]:
            if owner == me:
                raise self._nested()
            raise StorageLockTimeoutError(
                f"Another task on this thread's event loop holds the write lock on this "
                f"{self._label} storage; blocking for it here would stall the loop it "
                "needs to finish. Wait for the slot with areserve_write_slot instead"
            )
        if not self._lock.acquire(timeout=max(timeout, 0)):
            raise self._timed_out(timeout)
        self._owner = me

    async def areserve(self, timeout: float) -> threading.Lock:
        """Wait for the slot without blocking the event loop.

        Polls with a short, growing sleep between tries, so other coroutines
        on the loop keep running, including the one that holds it. Waits
        behind other threads too. Cancelled while waiting, it holds nothing:
        the lock is only ever taken by a non-blocking try, and nothing is
        awaited between taking it and returning.

        Returns:
            The lock acquired, to hand to a ``WriteSlot``.
        """
        me = _caller()
        if self._owner == me:
            raise self._nested()
        lock = self._lock
        deadline = time.monotonic() + max(timeout, 0)
        delay = _SLOT_POLL_MIN
        while not lock.acquire(blocking=False):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise self._timed_out(timeout)
            await asyncio.sleep(min(delay, remaining))
            delay = min(delay * 2, _SLOT_POLL_MAX)
        self._owner = me
        return lock

    def release(self, lock: threading.Lock | None = None) -> None:
        """Release the slot.

        Args:
            lock: The lock the caller acquired, if known. Nothing is
                released when it is no longer the current lock (a fork
                reset the slot since).
        """
        if lock is not None and lock is not self._lock:
            return
        self._owner = None
        self._lock.release()

    def take(
        self, storage: object, write_slot: WriteSlot | None, timeout: float
    ) -> Callable[[], None]:
        """Take the slot for a write transaction being begun on ``storage``.

        With ``write_slot`` the slot is already held and is taken over;
        otherwise it is acquired, waiting up to ``timeout``.

        Returns:
            A callable that undoes the take after a failed begin: it
            releases the slot, or returns it to ``write_slot``'s holder.
        """
        if write_slot is not None:
            write_slot._take(storage)
            return write_slot._give_back
        self.acquire(timeout)
        lock = self._lock
        return lambda: self.release(lock)


class WriteSlot:
    """The storage's writer slot, held before any transaction is begun.

    Returned by ``areserve_write_slot``, which waits for the slot without
    blocking the event loop. Passing it to ``begin_transaction(write_slot=...)``
    hands it over: the transaction takes the slot as already held and
    releases it on commit or abort as usual. A slot no transaction took is
    given back with ``release``, which is idempotent and does nothing once a
    transaction owns the slot.

    If beginning the transaction fails, the slot returns to the holder, so
    the holder's ``release`` still frees it.

    Use it as a context manager (``with await storage.areserve_write_slot()
    as slot:``) so a slot no transaction took is released however the block
    exits, cancellation included. As a backstop, a held slot that is garbage
    collected is released too: once unreachable, nothing can ever hand it to
    a transaction or release it.
    """

    __slots__ = ("_lock", "_state", "_storage")

    def __init__(self, storage: _SlotStorage, lock: threading.Lock) -> None:
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

    def __enter__(self) -> WriteSlot:
        """Enter: the slot is already held."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Release the slot unless a transaction took it."""
        self.release()

    def __del__(self) -> None:
        """Release a held slot dropped without ``release``."""
        try:
            self.release()
        except Exception:  # noqa: S110 (interpreter shutdown, half-built object)
            pass

    def _take(self, storage: object) -> None:
        """Hand the slot to a transaction being begun on ``storage``."""
        if storage is not self._storage:
            raise StorageError("Write slot belongs to a different storage")
        if self._state != "held":
            raise StorageError(f"Write slot is already {self._state}")
        self._state = "taken"

    def _give_back(self) -> None:
        """Return the slot to its holder after a failed begin."""
        if self._state == "taken":
            self._state = "held"
