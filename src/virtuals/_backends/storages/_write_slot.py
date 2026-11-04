"""The in-process writer slot shared by single-writer storages.

SQLite, LMDB, RocksDB and the in-memory storage allow one write transaction
per database at a time. Writers in one process queue on this slot before
they reach the backend, so threads hand off directly and coroutines wait
without blocking their event loop.

Threads compete for the lock itself: a sync ``acquire`` blocks in the
native lock and takes it the instant it is free, with no thread switch per
release. Coroutines cannot compete that way: one that polled for a free
slot would never see one next to a thread writing in a loop (the slot is
free for microseconds between two of its transactions). So coroutines
join a FIFO queue, and a release hands the slot to the head of that queue
without unlocking in between. While a coroutine is queued and threads are
also waiting, releases alternate between a handoff and an unlock, so
neither side starves: a queued coroutine waits at most one thread
transaction per coroutine ahead of it. A coroutine waiter whose loop is
blocked (a long sync call on the loop thread) holds up the queue for as
long as the loop does not run: the slot is handed to it and only its
loop can take it. Keep loop callbacks short, as the loop already asks.
A waiter whose loop closed, or whose future ended, is skipped at handoff,
and a grant held by a loop that closed after the handoff is taken back at
the next acquire. A loop that is stopped but never closed keeps the grant:
nothing can tell it apart from a loop that is about to run again.

Interrupts: a signal handler that raises (Ctrl-C), or a ``KeyboardInterrupt``
raised into the thread, runs between any two bytecodes of the main thread,
so a lock taken or given back and the bookkeeping that records it are never
one step in Python. The waits undo themselves when interrupted (a blocking
wait records its result in the same C call that waited, ``_acquire_into``,
so no handler runs between the take and the record), and the bookkeeping
around them is ordered so that the windows are a few bytecodes. They are not
zero: an interrupt that lands in one can leave the slot held in a process
that keeps running, with every later writer timing out; restart the process
then. Signal handlers are never touched.

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
from itertools import starmap
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


def _acquire_into(lock: threading.Lock, timeout: float, got: list) -> None:
    """``got.append(lock.acquire(True, timeout))`` as one C call.

    A handler cannot run between the acquire and the record (both happen
    inside ``list.extend``), so an interrupt out of this either left the
    lock alone or is reported after ``got`` says it is held.
    """
    got.extend(starmap(lock.acquire, ((True, timeout),)))


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
    """One queued coroutine, woken through a future on its loop.

    ``state`` moves under the writer's guard, exactly once, from
    ``waiting`` to ``granted`` (the slot was handed over: ``lock`` is the
    lock now held on the waiter's behalf), ``closed`` (the storage
    closed) or ``abandoned`` (its loop or future was gone at handoff).
    The future is only the wake-up; the state is what a waiter that woke
    for any other reason (timeout, cancellation, loop teardown) reads.
    """

    __slots__ = ("future", "lock", "loop", "state")

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self.future: asyncio.Future = loop.create_future()
        self.lock: threading.Lock | None = None
        self.state: Literal["waiting", "granted", "closed", "abandoned"] = "waiting"

    @property
    def dead(self) -> bool:
        """True if nothing can take a grant: the loop closed or the future ended."""
        return self.loop.is_closed() or self.future.done()

    def wake(self) -> bool:
        """Schedule the wake-up on the waiter's loop; False if the loop is gone."""
        try:
            self.loop.call_soon_threadsafe(_set_future, self.future)
        except RuntimeError:  # the loop is closed
            return False
        return True


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

    __slots__ = (
        "_barge_next",
        "_guard",
        "_hold",
        "_label",
        "_lock",
        "_owner",
        "_pending",
        "_sync_waiting",
        "_waiters",
    )

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
        # Coroutines waiting for a handoff, in arrival order.
        self._waiters: deque[_Waiter] = deque()
        # The coroutine the slot was handed to and that has not taken it yet.
        self._pending: _Waiter | None = None
        # Threads blocked in the native lock right now.
        self._sync_waiting = 0
        # After a handoff, the next release unlocks for the waiting threads.
        self._barge_next = False

    @property
    def lock(self) -> threading.Lock:
        """The current underlying lock (replaced by ``close`` and ``reset``)."""
        return self._lock

    def reset(self) -> None:
        """Forget any holder, e.g. in a forked child where it can never release.

        Takes no lock: the inherited ones may be held by threads that do
        not exist in this process. Every lock is replaced, and the queue
        (waiters are parent tasks) is dropped.
        """
        self._guard = threading.RLock()
        self._lock = threading.Lock()
        self._owner = None
        self._hold = None
        self._waiters = deque()
        self._pending = None
        self._sync_waiting = 0
        self._barge_next = False

    def close(self) -> None:
        """Replace the slot, so nothing reserved or taken before survives a reopen.

        Called by the storage's ``close`` after it ended its contexts. A
        recorded holder still there (a reserved slot no transaction took, or
        a context finishing on another thread) loses the slot: its lock is
        released here on its behalf and the holder's own release is a no-op
        from now on. The same for a handoff in flight: the lock is released
        here and the coroutine it was handed to wakes to a closed storage.
        Every waiter, a thread blocked in the old lock or a queued
        coroutine, wakes with ``StorageClosedError`` at once instead of
        after ``busy_timeout``.
        """
        with self._guard:
            old, self._lock = self._lock, threading.Lock()
            had_owner = self._owner is not None
            self._owner = None
            self._hold = None
            pending, self._pending = self._pending, None
            waiters, self._waiters = self._waiters, deque()
            self._barge_next = False
            if had_owner or pending is not None:
                old.release()
            if pending is not None:
                pending.state = "closed"  # woken already; it reads the state, not the lock
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

    def _held_by(self, me: _Owner, lock: threading.Lock, *, waited: bool = False) -> bool:
        """Record ``me`` as the owner of a fresh hold on ``lock``, marked in this context.

        Returns False, recording nothing, when ``lock`` is no longer the
        slot: ``close`` replaced it while the caller waited on it. With
        ``waited`` the caller was counted as a blocked thread and is
        uncounted here, in the same guarded section.
        """
        hold = _Hold(self)
        with self._guard:
            if waited:
                self._sync_waiting -= 1
            if lock is not self._lock:
                if waited:
                    # A release on the new slot may have unlocked for this
                    # thread while it was still counted: hand that on.
                    self._rescue_queue()
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

    def _hand_off(self) -> None:
        """Pass a held, ownerless slot to the first live queued coroutine, or unlock it.

        The lock stays locked across a handoff: ownership moves to the
        waiter, which records itself as the owner when it wakes. A waiter
        whose loop closed or whose future ended meanwhile (a task
        destroyed with its loop, a wait that gave up in the same instant)
        can never take it, so it is skipped.
        """
        waiters = self._waiters
        while waiters:
            waiter = waiters.popleft()
            if waiter.state != "waiting" or waiter.dead:
                waiter.state = "abandoned"
                continue
            waiter.state = "granted"
            waiter.lock = self._lock
            if waiter.wake():
                self._pending = waiter
                self._barge_next = True
                return
            waiter.state = "abandoned"
        self._lock.release()
        self._barge_next = False

    def _release_locked(self) -> None:
        """Give an ownerless held slot to whoever is next: a queued coroutine, or the threads.

        Hands off to the queue unless the previous release already did and
        threads are blocked in the lock: then it unlocks once, so threads
        and coroutines take turns while both are waiting.
        """
        if self._waiters and not (self._barge_next and self._sync_waiting):
            self._hand_off()
        else:
            self._lock.release()
            self._barge_next = False

    def _reclaim_dead_pending(self) -> None:
        """Take back a grant to a coroutine whose loop has closed since the handoff.

        ``_hand_off`` skips a dead waiter, but a loop that was merely
        stopped at that moment takes the grant and, closed later, can
        never run the coroutine to use or hand it on.
        """
        pending = self._pending
        if pending is not None and pending.state == "granted" and pending.loop.is_closed():
            pending.state = "abandoned"
            self._pending = None
            if pending.lock is self._lock:
                self._release_locked()

    def _rescue_queue(self) -> None:
        """A thread stopped waiting: if the lock is free and coroutines are queued, hand it on.

        An unlock meant for that thread must not leave the queue waiting on
        a free lock that nobody will release.
        """
        if self._waiters and self._lock.acquire(blocking=False):
            self._hand_off()

    def _pass_on(self, waiter: _Waiter) -> None:
        """A coroutine granted the slot will not use it (cancelled, torn down): hand it on."""
        with self._guard:
            if waiter.state != "granted":
                return  # closed: the old lock was released by close()
            waiter.state = "abandoned"
            if self._pending is waiter:
                self._pending = None
            if waiter.lock is self._lock:
                self._release_locked()

    def _granted(self, me: _Owner, waiter: _Waiter) -> threading.Lock:
        """Finish an ``areserve`` whose waiter is ``granted`` (or was closed meanwhile)."""
        hold = _Hold(self)
        with self._guard:
            lock = waiter.lock
            if waiter.state != "granted" or lock is None or lock is not self._lock:
                raise StorageClosedError(_CLOSED_WHILE_WAITING)
            if self._pending is waiter:
                self._pending = None
            self._owner = me
            self._hold = hold
        _mark_held(hold)
        return lock

    def acquire(self, timeout: float) -> threading.Lock:
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
        # From the take to the record of who holds it is a few bytecodes
        # (and the caller keeps the lock somewhere a release will find it
        # right after); the wait records its result in the same call that
        # waited, and gives back a lock taken as a handler fired.
        lock = self._lock
        if lock.acquire(blocking=False):  # the uncontended path costs no bookkeeping
            waited = False
        else:
            with self._guard:
                self._reclaim_dead_pending()
                lock = self._lock
                self._sync_waiting += 1
            waited = True
            got: list[bool] = []
            try:
                _acquire_into(lock, max(timeout, 0), got)
            except BaseException:
                # A signal handler raising: this thread is gone. The
                # lock it may have just taken (recorded in ``got``)
                # goes to whoever is next, and one unlocked for it
                # must go to the queue instead.
                with self._guard:
                    self._sync_waiting -= 1
                    if got and got[0]:
                        self._release_locked()
                    else:
                        self._reclaim_dead_pending()
                        self._rescue_queue()
                raise
            if not got[0]:
                with self._guard:  # timed out: as above
                    self._sync_waiting -= 1
                    self._reclaim_dead_pending()
                    self._rescue_queue()
                raise self._timed_out(timeout, on_loop=on_loop)
        if not self._held_by(me, lock, waited=waited):
            lock.release()
            raise StorageClosedError(_CLOSED_WHILE_WAITING)
        return lock

    async def areserve(self, timeout: float) -> threading.Lock:
        """Wait for the slot without blocking the event loop, behind the coroutines already queued.

        Takes a free slot at once when nobody is queued; otherwise joins
        the queue and is woken on its loop when the slot is handed to it,
        so other coroutines keep running, including the one that holds it.
        Cancelled or torn down while waiting, it holds nothing: a grant
        that lands in the same instant is handed on to the next waiter.
        Raises ``StorageClosedError`` if the storage closed while waiting.

        Returns:
            The lock acquired, to hand to a ``WriteSlot``.
        """
        me = _caller()
        if self._is_nested(me):
            raise self._nested()
        loop = asyncio.get_running_loop()
        waiter: _Waiter | None = None
        since = self._hold  # a hold of ours from before is not this call's to give back
        try:
            with self._guard:
                self._reclaim_dead_pending()
                # Behind queued coroutines, and behind threads blocked in the
                # lock: a free lock they are counted on is theirs to take.
                if (
                    not self._waiters
                    and not self._sync_waiting
                    and self._lock.acquire(blocking=False)
                ):
                    lock = self._lock
                    if not self._held_by(me, lock):  # pragma: no cover (close under the guard)
                        lock.release()
                        raise StorageClosedError(_CLOSED_WHILE_WAITING)
                    return lock
                waiter = _Waiter(loop)
                self._waiters.append(waiter)
        except BaseException:
            # An interrupt landing with the slot taken for this task or the
            # waiter queued: give either back.
            self.release_if_owned(me, since)
            if waiter is not None:
                self._pass_on_or_leave(waiter)
            raise
        try:
            await asyncio.wait_for(waiter.future, max(timeout, 0))
        except BaseException as e:
            # Timeout, cancellation, or the coroutine being closed with its
            # loop: leave the queue, or hand on a grant that just landed.
            with self._guard:
                state = waiter.state
                if state == "waiting":
                    self._waiters.remove(waiter)
            if state in ("waiting", "abandoned"):
                # Abandoned: a release saw the future already cancelled by
                # the timeout and skipped this waiter, which holds nothing.
                if isinstance(e, asyncio.TimeoutError):
                    raise self._timed_out(timeout) from None
                raise
            if not isinstance(e, asyncio.TimeoutError):
                self._pass_on(waiter)
                raise
            # Timed out in the same instant it was granted (ours) or closed (reported below).
        try:
            return self._granted(me, waiter)
        except BaseException:
            self.release_if_owned(me, since)  # an interrupt landed with it recorded as ours
            raise

    def _pass_on_or_leave(self, waiter: _Waiter) -> None:
        """A queued waiter whose coroutine is leaving: out of the queue, or its grant handed on."""
        with self._guard:
            if waiter.state == "waiting":
                self._waiters.remove(waiter)
                return
        self._pass_on(waiter)

    @staticmethod
    def caller() -> _Owner:
        """Who is asking: this thread and its running task, for ``release_if_owned``."""
        return _caller()

    def hold_token(self) -> object:
        """The current hold, as a token for ``release_if_owned``: taken before a call, it marks what was not that call's."""
        return self._hold

    def release_if_owned(self, me: _Owner, since: object) -> None:
        """Release the slot if ``me`` holds it through a take later than ``since``.

        For a coroutine whose lock reference was lost: an interrupt that
        lands as an ``await`` resumes leaves the slot recorded for the
        task with nothing holding the lock, and the task's own exception
        path gives it back here. Only the owner itself may call this, so
        a slot in flight for another task is never touched; a hold from
        before the call (``since``) is not touched either.
        """
        with self._guard:
            if self._owner == me and self._hold is not since and self._lock.locked():
                self._owner = None
                self._hold = None
                self._release_locked()

    def release(self, lock: threading.Lock | None = None) -> None:
        """Release the slot: hand it to a queued coroutine, or unlock it for the threads.

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
            self._release_locked()

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
