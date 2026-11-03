"""The in-process writer slot shared by single-writer storages.

SQLite, LMDB, RocksDB and the in-memory storage allow one write transaction
per database at a time. Writers in one process queue on this slot before
they reach the backend, so threads hand off directly and coroutines wait
without blocking their event loop.

The slot is fair: waiters form one FIFO queue, threads and coroutines
alike, and a release hands the slot to the head of the queue instead of
unlocking into a free-for-all. A coroutine that polled for a free slot
would never see one next to a thread writing in a loop (the slot is free
for microseconds between two of its transactions); with the handoff it
gets the slot after at most one transaction of every waiter ahead of it.
A late arrival takes a free slot directly only when nobody is queued.

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
import weakref
from collections import deque
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


# The most a sync ``acquire`` on an event loop thread blocks for a slot a
# thread holds (seconds). Long enough to absorb a short transaction on the
# other thread, short enough that the loop is not stalled in practice.
# Backends apply the same cap to their cross-process lock wait on a loop
# thread (contract section 4: a wait on the loop is bounded and surfaced).
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


def _set_future(future: asyncio.Future) -> None:
    """Wake a queued coroutine, unless it gave up (cancelled or timed out) meanwhile."""
    if not future.done():
        future.set_result(None)


class _Waiter:
    """One queued acquirer: a thread woken by an event, or a coroutine by a future.

    ``state`` moves under the writer's guard, exactly once, from
    ``waiting`` to ``granted`` (the slot was handed over: ``lock`` is the
    lock now held on the waiter's behalf) or ``closed`` (the storage
    closed). The event or future is only the wake-up; the state is what a
    waiter that woke for any other reason (timeout, cancellation) reads.
    """

    __slots__ = ("event", "future", "lock", "loop", "state")

    def __init__(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self.loop = loop
        self.future: asyncio.Future | None = None if loop is None else loop.create_future()
        self.event: threading.Event | None = threading.Event() if loop is None else None
        self.lock: threading.Lock | None = None
        self.state: Literal["waiting", "granted", "closed"] = "waiting"

    def wake(self) -> None:
        if self.event is not None:
            self.event.set()
            return
        loop, future = self.loop, self.future
        if loop is None or future is None:
            return
        try:
            loop.call_soon_threadsafe(_set_future, future)
        except RuntimeError:  # the loop is closed: nothing left to wake
            pass


class _SlotStorage(Protocol):
    _writer: WriterLock

    def _release_write_slot(self, lock: threading.Lock | None = None) -> None: ...


class WriterLock:
    """A storage's writer slot: one in-process writer at a time, waiters in FIFO order.

    Mutual exclusion is the lock alone. It is a plain Lock (not an RLock),
    so it may be released from a different thread than the one that took
    it, and it is handed from a releasing holder to the head of the queue
    without being unlocked in between: ``lock.locked()`` is True whenever
    the slot is held or in handoff.

    Nesting is told apart from waiting by execution context: ``acquire``
    and ``areserve`` mark the hold in the caller's context, and a begin that
    sees the current hold's mark runs below the holder, so it would wait on
    itself. A ``to_thread`` worker or an ``asyncio.run`` started inside an
    open transaction carries the mark; another task, or an unrelated job on
    a pool worker that once ran a handed-off transaction, does not. Limit:
    ``run_in_executor`` runs jobs in the worker's own context, so a
    transaction begun by a plain ``acquire`` in such a job and left open
    after it returns marks that worker until it ends.

    The owner record, the lock and the queue move together under
    ``_guard``, so a ``close`` racing an acquire or a release never leaves
    an owner on a free lock or a holder on a lock that is no longer the
    slot, and a waiter is granted the slot or told the storage closed,
    never both.
    """

    __slots__ = ("_guard", "_hold", "_label", "_lock", "_owner", "_waiters")

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
        self._waiters: deque[_Waiter] = deque()

    @property
    def lock(self) -> threading.Lock:
        """The current underlying lock (replaced by ``close`` and ``reset``)."""
        return self._lock

    def reset(self) -> None:
        """Forget any holder, e.g. in a forked child where it can never release.

        Takes no lock: the inherited ones may be held by threads that do
        not exist in this process. Every lock is replaced, and the queue
        (waiters are parent threads and tasks) is dropped.
        """
        self._guard = threading.RLock()
        self._lock = threading.Lock()
        self._owner = None
        self._hold = None
        self._waiters = deque()

    def close(self) -> None:
        """Replace the slot, so nothing reserved or taken before survives a reopen.

        Called by the storage's ``close`` after it ended its contexts. A
        recorded holder still there (a reserved slot no transaction took, or
        a context finishing on another thread) loses the slot: its lock is
        released here on its behalf and the holder's own release is a no-op
        from now on. Every queued waiter wakes with ``StorageClosedError``,
        at once instead of after ``busy_timeout``. A waiter the slot was
        handed to just before sees the lock is not the slot any more and
        raises the same.
        """
        with self._guard:
            old, self._lock = self._lock, threading.Lock()
            had_owner = self._owner is not None
            self._owner = None
            self._hold = None
            waiters, self._waiters = self._waiters, deque()
            if had_owner:
                old.release()
            for waiter in waiters:
                waiter.state = "closed"
                waiter.wake()

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
        if on_loop:
            return self.loop_cap_timeout()
        message = f"Timed out after {timeout}s waiting for the write lock"
        owner = self._owner
        if owner is not None and owner[1] is not None and owner[1]() is None:
            message += (
                "; the slot is held by a task that has ended without releasing it "
                "(a reserved slot that no transaction took and nothing released)"
            )
        return StorageLockTimeoutError(message)

    def loop_cap_timeout(self, what: str = "the write lock") -> StorageLockTimeoutError:
        """The error for a sync wait on an event loop thread that hit ``LOOP_THREAD_WAIT_CAP``.

        Shared with the backends' own cross-process waits, so a sync begin
        from a coroutine fails the same way whoever holds the lock.
        """
        message = (
            f"Timed out after {LOOP_THREAD_WAIT_CAP}s waiting for {what} on an event loop "
            f"thread, where a sync begin waits at most {LOOP_THREAD_WAIT_CAP}s so the loop "
            "is not stalled. Wait for the slot with areserve_write_slot instead"
        )
        owner = self._owner
        if owner is not None and owner[1] is not None and owner[1]() is None:
            message += (
                "; the slot is held by a task that has ended without releasing it "
                "(a reserved slot that no transaction took and nothing released)"
            )
        return StorageLockTimeoutError(message)

    # ------------------------------------------------------------------
    # Queue mechanics (every method here runs under ``_guard``)
    # ------------------------------------------------------------------

    def _take_free(self) -> bool:
        """Take the slot now if it is free and nobody is queued ahead."""
        return not self._waiters and self._lock.acquire(blocking=False)

    def _hand_off(self) -> None:
        """Pass a held, ownerless slot to the head of the queue, or unlock it.

        The lock stays locked across a handoff: ownership moves to the
        waiter, which records itself as the owner when it wakes.
        """
        waiters = self._waiters
        if waiters:
            waiter = waiters.popleft()
            waiter.state = "granted"
            waiter.lock = self._lock
            waiter.wake()
        else:
            self._lock.release()

    def _pass_on(self, lock: threading.Lock) -> None:
        """A waiter was granted ``lock`` but will not use it (cancelled): hand it on."""
        with self._guard:
            if lock is self._lock:
                self._hand_off()
            else:
                lock.release()  # an old lock: close() replaced the slot, nothing waits on it

    def _enqueue(
        self, loop: asyncio.AbstractEventLoop | None
    ) -> tuple[threading.Lock | None, _Waiter | None]:
        """Take the slot if free with nobody queued, else join the queue.

        Returns ``(lock, None)`` when taken, ``(None, waiter)`` when queued.
        """
        with self._guard:
            if self._take_free():
                return self._lock, None
            waiter = _Waiter(loop)
            self._waiters.append(waiter)
            return None, waiter

    def _granted(self, me: _Owner, waiter: _Waiter) -> threading.Lock:
        """Finish an acquire whose waiter is ``granted`` or ``closed``."""
        lock = waiter.lock
        if waiter.state != "granted" or lock is None:
            raise StorageClosedError(_CLOSED_WHILE_WAITING)
        if not self._held_by(me, lock):
            lock.release()
            raise StorageClosedError(_CLOSED_WHILE_WAITING)
        return lock

    def acquire(self, timeout: float) -> threading.Lock:
        """Block for the slot, up to ``timeout`` seconds, behind everyone already queued.

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

        Returns:
            The lock acquired: the slot's identity for the later ``release``.
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
        lock, waiter = self._enqueue(None)
        if lock is not None:
            if not self._held_by(me, lock):
                lock.release()
                raise StorageClosedError(_CLOSED_WHILE_WAITING)
            return lock
        if waiter is None or waiter.event is None:  # pragma: no cover (typing only)
            raise StorageError("writer slot queue is inconsistent")
        if not waiter.event.wait(max(timeout, 0)):
            with self._guard:
                if waiter.state == "waiting":
                    self._waiters.remove(waiter)
                    raise self._timed_out(timeout, on_loop=on_loop)
                # Granted or closed in the last instant: same as a wake.
        return self._granted(me, waiter)

    async def areserve(self, timeout: float) -> threading.Lock:
        """Wait for the slot without blocking the event loop, behind everyone already queued.

        Takes a free slot at once when nobody is queued; otherwise joins
        the queue and is woken on its loop when the slot is handed to it,
        so other coroutines keep running, including the one that holds it.
        Cancelled while waiting, it holds nothing: a grant that lands in
        the same instant is handed on to the next waiter. Raises
        ``StorageClosedError`` if the storage closed while waiting.

        Returns:
            The lock acquired, to hand to a ``WriteSlot``.
        """
        me = _caller()
        if self._is_nested(me):
            raise self._nested()
        lock, waiter = self._enqueue(asyncio.get_running_loop())
        if lock is not None:
            if not self._held_by(me, lock):
                lock.release()
                raise StorageClosedError(_CLOSED_WHILE_WAITING)
            return lock
        if waiter is None or waiter.future is None:  # pragma: no cover (typing only)
            raise StorageError("writer slot queue is inconsistent")
        try:
            await asyncio.wait_for(waiter.future, max(timeout, 0))
        except (asyncio.TimeoutError, asyncio.CancelledError) as e:
            cancelled = isinstance(e, asyncio.CancelledError)
            with self._guard:
                state = waiter.state
                if state == "waiting":
                    self._waiters.remove(waiter)
            if state == "waiting":
                if cancelled:
                    raise
                raise self._timed_out(timeout) from None
            if cancelled:
                if state == "granted":
                    self._pass_on(waiter.lock)  # type: ignore[arg-type]
                raise
            # Timed out in the same instant it was granted (ours) or closed (reported below).
        return self._granted(me, waiter)

    def release(self, lock: threading.Lock | None = None) -> None:
        """Release the slot: hand it to the head of the queue, or unlock it.

        Args:
            lock: The lock the caller acquired, if known. Nothing is
                released when it is no longer the current lock (a close
                or a fork replaced the slot since).
        """
        with self._guard:
            if lock is not None and lock is not self._lock:
                return
            if not self._lock.locked():
                raise RuntimeError("release unlocked lock")
            self._owner = None
            self._hold = None
            self._hand_off()

    def take_slot(
        self, storage: object, write_slot: WriteSlot | None, timeout: float
    ) -> tuple[threading.Lock, Callable[[], None]]:
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
            The lock taken, which the context keeps and releases by
            identity, and a callable that undoes the take after a failed
            begin: it releases the slot, or returns it (and its owner) to
            ``write_slot``'s holder. If the slot was replaced meanwhile
            (close), the undo only marks ``write_slot`` released and
            records nothing on the fresh slot.
        """
        if write_slot is not None:
            with self._guard:
                write_slot._take(storage)
                lock = write_slot._lock
                previous = self._owner
                me = _caller()
                self._owner = me
                hold = self._hold
            if me[1] is not None and hold is not None:
                _mark_held(hold)

            def give_back() -> None:
                with self._guard:
                    if lock is self._lock:
                        self._owner = previous
                        write_slot._give_back()
                    else:
                        write_slot._state = "released"

            return lock, give_back
        lock = self.acquire(timeout)
        return lock, lambda: self.release(lock)

    def take(
        self, storage: object, write_slot: WriteSlot | None, timeout: float
    ) -> Callable[[], None]:
        """``take_slot`` for callers that read the lock themselves; returns the undo only."""
        return self.take_slot(storage, write_slot, timeout)[1]


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
