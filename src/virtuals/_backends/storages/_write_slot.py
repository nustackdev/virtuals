"""The in-process writer slot shared by single-writer storages.

SQLite, LMDB, RocksDB and the in-memory storage allow one write transaction
per database at a time. Writers in one process queue on this slot before
they reach the backend, so threads hand off directly and coroutines wait
without blocking their event loop.

Lifecycle: a storage's ``close()`` calls ``WriterLock.close()``, which
replaces the slot. Nothing reserved or taken before survives into a
reopen: a waiter wakes with ``StorageClosedError``, a ``WriteSlot`` from
before refuses to be taken, and a late release from an old holder touches
nothing. A forked child calls ``reset()`` instead, which replaces every
lock without touching the inherited ones.
"""

from __future__ import annotations

import asyncio
import contextvars
import threading
import time
import weakref
from typing import TYPE_CHECKING, Literal, Protocol

from virtuals.tkv.storage import (
    StorageClosedError,
    StorageError,
    StorageLockTimeoutError,
    StorageNestedTransactionError,
)


if TYPE_CHECKING:
    from collections.abc import Callable


__all__ = ["LOOP_THREAD_WAIT_CAP", "WriteSlot", "WriterLock"]


# Poll interval bounds for a coroutine waiting on the writer slot (seconds).
_SLOT_POLL_MIN = 0.0005
_SLOT_POLL_MAX = 0.005

# The most a sync ``acquire`` on an event loop thread blocks for a slot a
# thread holds (seconds). Long enough to absorb a short transaction on the
# other thread, short enough that the loop is not stalled in practice.
LOOP_THREAD_WAIT_CAP = 0.05

_CLOSED_WHILE_WAITING = "Storage was closed while waiting for the write lock"

_Owner = tuple[int, "weakref.ref[asyncio.Task] | None"]


class _Hold:
    """One acquisition of a writer slot, marked in the context that took it."""

    __slots__ = ("writer",)

    def __init__(self, writer: WriterLock) -> None:
        self.writer = writer

    @property
    def current(self) -> bool:
        return self.writer._hold is self


# The holds taken in this execution context. ``asyncio.to_thread``, new
# tasks and ``asyncio.run`` copy the context, so a call made below the
# holder sees its mark; an unrelated job on a reused pool worker does not.
# A mark outlives its hold harmlessly: it only counts while it is the
# writer's current hold.
_HELD: contextvars.ContextVar[tuple[_Hold, ...]] = contextvars.ContextVar(
    "virtuals_write_slot_held", default=()
)


def _mark_held(hold: _Hold) -> None:
    """Mark ``hold`` in this context, dropping marks of ended holds."""
    _HELD.set((*(h for h in _HELD.get() if h.current), hold))


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
    _writer: WriterLock

    def _release_write_slot(self, lock: threading.Lock | None = None) -> None: ...


class WriterLock:
    """A storage's writer slot: one in-process writer at a time.

    Mutual exclusion is the lock alone. It is a plain Lock (not an RLock),
    so it may be released from a different thread than the one that took it.

    Nesting is told apart from waiting by execution context: ``acquire`` and
    ``areserve`` mark the hold in the caller's context, and a begin that
    sees the current hold's mark runs below the holder, so it would wait on
    itself. A ``to_thread`` worker or an ``asyncio.run`` started inside an
    open transaction carries the mark; another task, or an unrelated job on
    a pool worker that once ran a handed-off transaction, does not. Limit:
    ``run_in_executor`` runs jobs in the worker's own context, so a
    transaction begun by a plain ``acquire`` in such a job and left open
    after it returns marks that worker until it ends.

    The owner record and the lock move together under ``_guard``, so a
    ``close`` racing an acquire or a release never leaves an owner on a
    free lock or a holder on a lock that is no longer the slot.
    """

    __slots__ = ("_guard", "_hold", "_label", "_lock", "_owner")

    def __init__(self, label: str) -> None:
        """Initialize a free slot.

        Args:
            label: Backend name used in error messages, e.g. ``"SQLite"``.
        """
        self._label = label
        # Reentrant: a finalizer that releases a slot may run inside one of
        # these sections, on the same thread, when a collection is triggered
        # there.
        self._guard = threading.RLock()
        self._lock = threading.Lock()
        self._owner: _Owner | None = None
        self._hold: _Hold | None = None

    @property
    def lock(self) -> threading.Lock:
        """The current underlying lock (replaced by ``close`` and ``reset``)."""
        return self._lock

    def reset(self) -> None:
        """Forget any holder, e.g. in a forked child where it can never release.

        Takes no lock: the inherited ones may be held by threads that do
        not exist in this process. Every lock is replaced.
        """
        self._guard = threading.RLock()
        self._lock = threading.Lock()
        self._owner = None
        self._hold = None

    def close(self) -> None:
        """Replace the slot, so nothing reserved or taken before survives a reopen.

        Called by the storage's ``close`` after it ended its contexts. A
        recorded holder still there (a reserved slot no transaction took, or
        a context finishing on another thread) loses the slot: its lock is
        released here on its behalf, so a writer blocked on it wakes at
        once instead of waiting out ``busy_timeout``, and the holder's own
        release is a no-op from now on. Whoever acquires the old lock sees
        it is not the slot any more and raises ``StorageClosedError``.
        """
        with self._guard:
            old, self._lock = self._lock, threading.Lock()
            had_owner = self._owner is not None
            self._owner = None
            self._hold = None
            if had_owner:
                old.release()

    def _is_nested(self, me: _Owner) -> bool:
        """True if ``me`` can never get the slot because it already holds it.

        The caller must run below the current hold (its context carries the
        mark). Then it is nested when it is outside any task (a sync call or
        a ``to_thread`` worker below the holder), is the owning task itself,
        or is a task on the owner's thread while the owner is that thread
        outside any task (an ``asyncio.run`` inside a sync transaction). A
        task spawned by the owning task carries the mark but is a waiter.
        """
        hold, owner = self._hold, self._owner
        if hold is None or owner is None or hold not in _HELD.get():
            return False
        return me[1] is None or owner[1] == me[1] or (owner[1] is None and owner[0] == me[0])

    def _held_by(self, me: _Owner, lock: threading.Lock) -> bool:
        """Record ``me`` as the owner of a fresh hold on ``lock``, marked in this context.

        Returns False, recording nothing, when ``lock`` is no longer the
        slot: ``close`` replaced it while the caller waited on it.
        """
        hold = _Hold(self)
        with self._guard:
            if lock is not self._lock:
                return False
            self._owner = me
            self._hold = hold
        _mark_held(hold)
        return True

    def _nested(self) -> StorageNestedTransactionError:
        return StorageNestedTransactionError(
            f"This task or thread already holds the write lock on this {self._label} "
            "storage; a second write transaction here would wait on itself"
        )

    def _timed_out(self, timeout: float, *, on_loop: bool = False) -> StorageLockTimeoutError:
        message = f"Timed out after {timeout}s waiting for the write lock"
        if on_loop:
            message += (
                f" on an event loop thread, where a sync begin waits at most "
                f"{LOOP_THREAD_WAIT_CAP}s so the loop is not stalled. Wait for the slot "
                "with areserve_write_slot instead"
            )
        owner = self._owner
        if owner is not None and owner[1] is not None and owner[1]() is None:
            message += (
                "; the slot is held by a task that has ended without releasing it "
                "(a reserved slot that no transaction took and nothing released)"
            )
        return StorageLockTimeoutError(message)

    def acquire(self, timeout: float) -> None:
        """Block for the slot, up to ``timeout`` seconds.

        Fails at once with ``StorageNestedTransactionError`` when called
        below the current holder (see ``_is_nested``), and with
        ``StorageLockTimeoutError`` when another task on this thread's loop
        holds it, since that holder cannot run to release it while this
        thread blocks. On an event loop thread the wait for a holder on
        another thread is capped at ``LOOP_THREAD_WAIT_CAP`` (contract
        section 4: a wait on the loop is bounded and surfaced), so a sync
        begin or batch write from a coroutine never stalls the loop for
        long; a longer wait belongs in ``areserve``.

        Raises ``StorageClosedError`` if the storage closed while waiting.
        """
        me = _caller()
        if self._is_nested(me):
            raise self._nested()
        owner = self._owner
        if (
            owner is not None
            and owner[0] == me[0]
            and owner[1] is not None
            and owner[1]() is not None
        ):
            raise StorageLockTimeoutError(
                f"Another task on this thread's event loop holds the write lock on this "
                f"{self._label} storage; blocking for it here would stall the loop it "
                "needs to finish. Wait for the slot with areserve_write_slot instead"
            )
        on_loop = asyncio._get_running_loop() is not None
        if on_loop:
            timeout = min(timeout, LOOP_THREAD_WAIT_CAP)
        lock = self._lock
        if not lock.acquire(timeout=max(timeout, 0)):
            raise self._timed_out(timeout, on_loop=on_loop)
        if not self._held_by(me, lock):
            lock.release()
            raise StorageClosedError(_CLOSED_WHILE_WAITING)

    async def areserve(self, timeout: float) -> threading.Lock:
        """Wait for the slot without blocking the event loop.

        Polls with a short, growing sleep between tries, so other coroutines
        on the loop keep running, including the one that holds it. Waits
        behind other threads too. Cancelled while waiting, it holds nothing:
        the lock is only ever taken by a non-blocking try, and nothing is
        awaited between taking it and returning. Raises
        ``StorageClosedError`` if the storage closed while waiting.

        Returns:
            The lock acquired, to hand to a ``WriteSlot``.
        """
        me = _caller()
        if self._is_nested(me):
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
        if not self._held_by(me, lock):
            lock.release()
            raise StorageClosedError(_CLOSED_WHILE_WAITING)
        return lock

    def release(self, lock: threading.Lock | None = None) -> None:
        """Release the slot.

        Args:
            lock: The lock the caller acquired, if known. Nothing is
                released when it is no longer the current lock (a close
                or a fork replaced the slot since).
        """
        with self._guard:
            if lock is not None and lock is not self._lock:
                return
            self._owner = None
            self._hold = None
            self._lock.release()

    def take(
        self, storage: object, write_slot: WriteSlot | None, timeout: float
    ) -> Callable[[], None]:
        """Take the slot for a write transaction being begun on ``storage``.

        With ``write_slot`` the slot is already held and is taken over, and
        its owner becomes the caller: the thread (and task) running the
        transaction, which may differ from the one that reserved it, e.g. a
        slot reserved on the loop and handed to ``asyncio.to_thread``. The
        hold keeps the mark of the context that reserved it, which a
        ``to_thread`` worker or a task spawned there carries; a taker that
        is a task is marked too. A plain thread taker is not, since its
        context may be a pool worker's own, shared with unrelated jobs. A
        slot from before a close or a fork is refused (``StorageError``).
        Otherwise it is acquired, waiting up to ``timeout``.

        Returns:
            A callable that undoes the take after a failed begin: it
            releases the slot, or returns it (and its owner) to
            ``write_slot``'s holder.
        """
        if write_slot is not None:
            with self._guard:
                write_slot._take(storage)
                previous = self._owner
                me = _caller()
                self._owner = me
                hold = self._hold
            if me[1] is not None and hold is not None:
                _mark_held(hold)

            def give_back() -> None:
                with self._guard:
                    self._owner = previous
                    write_slot._give_back()

            return give_back
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
    as slot:``), or hand it to a transaction or ``release`` it explicitly
    on every path: that is required. The finalizer is a last resort that
    releases a held slot only once it is unreachable, which may be much
    later, e.g. a cancelled task's traceback keeps it alive for as long as
    the task object lives, and every other writer waits until then.
    """

    __slots__ = ("_lock", "_state", "_storage")

    def __init__(self, storage: _SlotStorage, lock: threading.Lock) -> None:
        """Initialize a held slot.

        Args:
            storage: The storage whose writer slot this is.
            lock: The writer lock acquired for it. Kept so a slot from
                before a close or a fork never releases the fresh lock.
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
        """Last resort: release a held slot that became unreachable unreleased."""
        try:
            self.release()
        except Exception:  # noqa: S110 (interpreter shutdown, half-built object)
            pass

    def _take(self, storage: object) -> None:
        """Hand the slot to a transaction being begun on ``storage``.

        A slot reserved before the storage was closed (or before a fork)
        holds nothing any more: it is marked released and refused, so the
        transaction never starts on a slot it does not hold.
        """
        if storage is not self._storage:
            raise StorageError("Write slot belongs to a different storage")
        if self._state != "held":
            raise StorageError(f"Write slot is already {self._state}")
        if self._lock is not self._storage._writer.lock:
            self._state = "released"
            raise StorageError("Write slot is stale: the storage was closed after it was reserved")
        self._state = "taken"

    def _give_back(self) -> None:
        """Return the slot to its holder after a failed begin."""
        if self._state == "taken":
            self._state = "held"
