"""The writer thread and the cross-process gate behind LMDB write transactions.

LMDB's writer lock is a robust pthread mutex in the lock file, owned by the
thread that took it in ``begin`` and only unlockable by that thread. It has
no timed variant. Two helpers keep virtuals clear of both edges:

- ``WriterThread``: one long-lived thread per storage runs every
  ``env.begin(write=True)``, ``commit`` and ``abort``. Callers hand it the
  call and wait on a future, bounded for begin, so a write transaction can
  be begun on one thread and finished on any other without touching the
  mutex from the wrong thread. A begin whose requester gave up is aborted
  by the thread the moment it returns, and always on the thread: no undo
  ever runs on a requester's thread.
- ``WriterGate``: an ``flock`` on a file next to the environment, taken
  before ``env.begin(write=True)`` and released after the transaction ends.
  Peers that go through virtuals wait on it instead of blocking inside
  LMDB's mutex, so the writer thread's begin never waits behind a virtuals
  peer, and ``close()`` never has to tear down an environment with a begin
  in flight. A contended gate is waited for with a blocking ``flock`` on a
  helper thread, so the kernel wakes the waiter the moment the holder lets
  go (no polling); the requester waits on a future bounded by its timeout,
  and a lock that lands after the requester gave up is released at once.
  The kernel drops the lock when its holder dies, since no forked child
  keeps a descriptor on it (the helper's duplicate is closed at fork).

Filesystem notes: LMDB itself requires a local filesystem (its lock file is
shared memory between processes and its data file is mmapped), so the gate
is only ever used on a local disk too. On an NFS client ``flock`` is
emulated with POSIX locks (per process, dropped by closing any descriptor
on the file, held until lease expiry if the client dies), none of which
the gate relies on, since environments never live there.
"""

from __future__ import annotations

import asyncio
import fcntl
import os
import queue
import sys
import threading
import time
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from logging import getLogger
from typing import TYPE_CHECKING, Any

from virtuals.tkv.storage import StorageClosedError, StorageError, StorageLockTimeoutError


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


__all__ = ["WriterGate", "WriterThread"]


logger = getLogger(__name__)


def _set_done(done: asyncio.Future) -> None:
    if not done.done():
        done.set_result(True)


async def _await_done(future: Future, timeout: float) -> None:
    """Wait on the loop for a concurrent future, up to ``timeout``, never cancelling it.

    ``asyncio.wait`` (unlike ``wait_for``) leaves the futures alone on
    timeout, so a result that lands as the wait ends is never dropped.
    """
    loop = asyncio.get_running_loop()
    done = loop.create_future()

    def wake(_: Future) -> None:
        try:
            loop.call_soon_threadsafe(_set_done, done)
        except RuntimeError:  # loop closed: the requester is gone
            pass

    future.add_done_callback(wake)
    await asyncio.wait({done}, timeout=max(timeout, 0))


class WriterThread:
    """One thread that owns begin, commit and abort of a storage's write transactions.

    Started on first use, stopped by ``stop``. Jobs run in order; one is in
    flight at a time, which is all LMDB allows anyway (one write
    transaction per environment).
    """

    __slots__ = (
        "_lock",
        "_name",
        "_owner",
        "_queue",
        "_soft_since",
        "_stopped",
        "_thread",
        "_wait_since",
        "_waiting",
    )

    def __init__(self, name: str) -> None:
        """Initialize an idle writer thread.

        Args:
            name: Thread name, for diagnostics.
        """
        self._name = name
        # Reentrant: a collection inside ``_submit`` may run a finalizer
        # that submits too.
        self._lock = threading.RLock()
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._thread: threading.Thread | None = None
        self._stopped = False
        # When the writer side last made progress on a job that someone
        # waits for, 0.0 when none is in flight. Read by the storage's
        # readers (see ``yield_to_writer``): a stamp that ages past what a
        # hand-off or a short LMDB call takes means the thread, or the
        # requester it answers, cannot get the GIL back from threads
        # spinning on short py-lmdb calls (CPython's convoy effect). The
        # requester sets it when it queues the job and clears it once it
        # has its result; the thread refreshes it at every boundary where
        # it will need the GIL back (job pickup, after the begin, every
        # ``PROGRESS_EVERY`` replayed ops, before handing the result over).
        # One slot per storage, owned: ``_stamp`` hands out a token and only
        # its holder clears the stamp, so a job nobody waits for (an
        # abort queued by a finalizer, an undo) never clears the stamp of
        # the requester queued behind it. An overlapping requester
        # (close() ending leftovers beside a finalizer) still replaces the
        # other's stamp, which costs that one job its protection, and
        # readers of one storage never yield to another storage's writer.
        # Both accepted.
        self._wait_since = 0.0
        self._owner = 0
        # True between a requester's stamp and its clear: the thread only
        # refreshes or restores a stamp somebody still owns, so a requester
        # that left by an exception mid-job never leaves one behind.
        self._waiting = False
        # A GIL-free call that takes its time for a good reason (the fsync
        # commit, or the gate's flock behind a peer process's transaction).
        # Readers answer it with the nudge tier only: the GIL is needed
        # back once, at the end, not throughout.
        self._soft_since = 0.0

    def _stamp(self) -> int:
        """Set the stamp; returns the token that clears it."""
        self._owner += 1
        self._waiting = True
        self._wait_since = time.monotonic()
        return self._owner

    def _unstamp(self, token: int) -> None:
        """Clear the stamp, unless somebody else has stamped since."""
        if self._owner == token:
            self._waiting = False
            self._wait_since = 0.0

    def _progress(self) -> None:
        """Refresh the owner's stamp from the thread (a long replay is progress)."""
        if self._waiting:
            self._wait_since = time.monotonic()

    def _stamp_soft(self) -> float:
        stamp = time.monotonic()
        self._soft_since = stamp
        return stamp

    def _unstamp_soft(self, stamp: float) -> None:
        if self._soft_since == stamp:
            self._soft_since = 0.0

    def start(self) -> None:
        """Accept jobs (again): the storage is open. The thread itself starts lazily."""
        with self._lock:
            self._stopped = False

    def _submit(
        self,
        fn: Callable[[], Any],
        future: Future,
        abandoned: Callable[[Any], None] | None,
        then: Callable[[Any], Any] | None = None,
        done: Future | None = None,
        then_abandoned: Callable[[Any], None] | None = None,
        waited: bool = True,
    ) -> None:
        """Queue a job, starting the thread if needed.

        Thread and queue are read together under the lock: a ``stop``
        between the two would strand the job on a queue nobody serves.
        After ``stop`` nothing is accepted until ``start``: a job that
        started a fresh thread then would run against an environment
        being closed. ``waited`` says a requester waits for the result and
        will clear the progress stamp itself. ``abandoned`` and
        ``then_abandoned`` take a result its requester no longer waits for
        (see ``run`` and ``run_staged``).
        """
        with self._lock:
            if self._stopped:
                raise StorageClosedError("Storage is not open")
            thread = self._thread
            if thread is None or not thread.is_alive():
                thread = threading.Thread(
                    target=self._loop, args=(self._queue,), name=self._name, daemon=True
                )
                self._thread = thread
                thread.start()
            self._queue.put((fn, future, abandoned, then, done, then_abandoned, waited))

    def _on_thread(self) -> bool:
        return threading.current_thread() is self._thread

    def _loop(self, jobs: queue.SimpleQueue) -> None:
        while True:
            job = jobs.get()
            if job is None:
                return
            fn, future, abandoned, then, done, then_abandoned, waited = job
            # A waited job keeps the requester's stamp through its short
            # steps (pickup, begin, hand-off): each is a point where the
            # GIL is needed back, and one continuous age covers them all.
            # Long work (a replay, the commit) refreshes or softens it
            # itself. A job nobody waits for stamps and clears its own,
            # unless a requester queued behind it has stamped already: then
            # its stamp is refreshed, never replaced, and stays the
            # requester's to clear.
            token: int | None = None
            if not waited:
                if self._waiting:
                    self._progress()
                else:
                    token = self._stamp()
            try:
                self._run_job(fn, future, abandoned, then, done, then_abandoned)
            finally:
                if token is not None:
                    self._unstamp(token)

    def _run_job(
        self,
        fn: Callable[[], Any],
        future: Future,
        abandoned: Callable[[Any], None] | None,
        then: Callable[[Any], Any] | None,
        done: Future | None,
        then_abandoned: Callable[[Any], None] | None,
    ) -> None:
        try:
            result = fn()
        except BaseException as e:
            if future.set_running_or_notify_cancel():
                future.set_exception(e)
            if done is not None and done.set_running_or_notify_cancel():
                done.set_exception(e)
            return
        # The requester may have given up while ``fn`` ran. Marking the
        # future running after the call makes that race explicit: either
        # the requester's cancel wins and ``abandoned`` takes the result
        # (an abandoned begin is aborted at once), or the result lands.
        if not future.set_running_or_notify_cancel():
            self._hand_over(abandoned, result)
            return
        future.set_result(result)
        if then is not None:
            # The second stage runs right here, on the result the
            # requester now owns: no second hop, and no cancel can cut it
            # short (a commit is never abandoned half way). Its result is
            # handed over the same way if the requester gave up meanwhile.
            try:
                second = then(result)
            except BaseException as e:
                if done.set_running_or_notify_cancel():  # type: ignore[union-attr]
                    done.set_exception(e)  # type: ignore[union-attr]
                return
            if done.set_running_or_notify_cancel():  # type: ignore[union-attr]
                done.set_result(second)  # type: ignore[union-attr]
            else:
                self._hand_over(then_abandoned, second)

    @staticmethod
    def _hand_over(abandoned: Callable[[Any], None] | None, result: Any) -> None:  # noqa: ANN401
        """Give a result nobody waits for to ``abandoned``, right here on the writer thread."""
        if abandoned is None:
            return
        try:
            abandoned(result)
        except Exception as e:
            logger.error(f"Handling an abandoned LMDB writer call failed: {e}")

    def fire(self, fn: Callable[[], Any]) -> None:
        """Run ``fn`` on the writer thread; nobody waits for it. Dropped after ``stop``."""
        try:
            self._submit(fn, Future(), None, waited=False)
        except StorageClosedError:
            pass  # close() ended every begun transaction itself

    def run(
        self,
        fn: Callable[[], Any],
        *,
        timeout: float | None = None,
        abandoned: Callable[[Any], None] | None = None,
    ) -> Any:  # noqa: ANN401
        """Run ``fn`` on the writer thread and return its result.

        Blocks the caller. With ``timeout`` the wait is bounded: a call
        still in flight then is abandoned and ``StorageLockTimeoutError``
        is raised. A result that lands just as the wait ends is taken, not
        dropped. The wait left by any other exception (a signal handler
        raising, ``KeyboardInterrupt``) abandons the call the same way.

        ``abandoned`` takes the result of an abandoned call once it lands,
        on the writer thread: a begin nobody waits for is aborted, a commit
        nobody waits for has its bookkeeping finished. Without it the
        result is dropped (an abort needs nothing).
        """
        if self._on_thread():
            return fn()  # a job calling back in (e.g. a finalizer); never wait on ourselves
        future: Future = Future()
        token = self._stamp()  # queued: the thread needs the GIL to pick it up
        try:
            self._submit(fn, future, abandoned)
            try:
                return future.result(timeout)
            except FutureTimeoutError:
                pass
            except BaseException as e:
                self._left(future, future, e, abandoned)
                raise
            if future.cancel():
                raise StorageLockTimeoutError(
                    f"Timed out after {timeout}s waiting for LMDB's writer lock"
                )
            try:
                return future.result()
            except BaseException as e:
                self._left(future, future, e, abandoned)
                raise
        finally:
            self._unstamp(token)

    def run_staged(
        self,
        fn: Callable[[], Any],
        then: Callable[[Any], Any],
        *,
        timeout: float,
        abandoned: Callable[[Any], None],
        then_abandoned: Callable[[Any], None] | None = None,
    ) -> Any:  # noqa: ANN401
        """Run ``fn`` and then ``then(result)`` on the writer thread, one hop.

        The wait is bounded only while ``fn`` is in flight: a timeout then
        abandons it (``abandoned`` runs on its result when it lands) and
        raises ``StorageLockTimeoutError``. Once ``fn`` has landed, ``then``
        runs to the end and the caller waits for it. For a write-only
        transaction ``fn`` is the begin and ``then`` replays and commits,
        so a begin behind a writer outside virtuals stays bounded while a
        commit is never cut short. A wait left by any other exception
        abandons ``fn`` the same way if it has not landed; if it has,
        ``then`` runs to the end and ``then_abandoned`` takes its result.
        """
        if self._on_thread():
            return then(fn())
        begun: Future = Future()
        done: Future = Future()
        token = self._stamp()
        try:
            self._submit(fn, begun, abandoned, then, done, then_abandoned)
            try:
                return done.result(timeout)
            except FutureTimeoutError:
                pass
            except BaseException as e:
                self._left(begun, done, e, then_abandoned)
                raise
            if begun.cancel():
                raise StorageLockTimeoutError(
                    f"Timed out after {timeout}s waiting for LMDB's writer lock"
                )
            try:
                return done.result()
            except BaseException as e:
                self._left(begun, done, e, then_abandoned)
                raise
        finally:
            self._unstamp(token)

    def _left(
        self,
        job: Future,
        result: Future,
        error: BaseException,
        abandoned: Callable[[Any], None] | None,
    ) -> None:
        """A wait on ``result`` ended with ``error``: abandon the job if that was not its own failure.

        A stage still in flight is cancelled: the thread hands its result
        to its abandoned handler right after the call, before any later
        job. A result already in is handed over now, by a job queued
        here, before the caller sees the error. Either way nothing the
        thread begins or commits is left unobserved, and the handler runs
        before any job the caller queues afterwards (an abort).
        """
        if result.done() and not result.cancelled() and result.exception() is error:
            return  # the call itself failed: nothing landed
        if job.cancel():
            return  # the first stage is in flight: the thread hands over its result
        if result is not job and result.cancel():
            return  # the second stage is in flight: the thread hands over its result
        if abandoned is not None:
            result.add_done_callback(lambda f: self._abandoned_later(f, abandoned))

    def try_fire(self, fn: Callable[[], Any]) -> bool:
        """``fire`` for finalizers: queued, never waited for, never at exit.

        A finalizer runs on whatever thread triggered the collection, which
        may hold a lock the writer thread needs (``_env_lock`` during a
        begin or a close), so waiting here could stall that thread for the
        whole timeout; the job is queued and the finalizer moves on. The
        queue is FIFO, so a begin queued after it runs after it. At
        interpreter shutdown, when the daemon thread can no longer execute
        Python, or after ``stop``, nothing is queued and False is returned:
        process exit releases LMDB's writer lock and the gate anyway, and
        close() ends every live write itself.
        """
        if sys.is_finalizing():
            return False
        if self._on_thread():
            fn()
            return True
        try:
            self._submit(fn, Future(), None, waited=False)
        except StorageClosedError:
            return False
        return True

    async def arun(
        self,
        fn: Callable[[], Any],
        *,
        timeout: float,
        abandoned: Callable[[Any], None] | None = None,
    ) -> Any:  # noqa: ANN401
        """Like ``run`` but waits without blocking the event loop.

        Cancelled while waiting, it abandons the call: ``abandoned`` takes
        a result that has landed or lands later, on the writer thread. No
        cancel of the wait ever loses a result.
        """
        future: Future = Future()
        token = self._stamp()
        try:
            self._submit(fn, future, abandoned)
            try:
                await _await_done(future, timeout)
            except BaseException:
                if not future.cancel() and abandoned is not None:
                    future.add_done_callback(lambda f: self._abandoned_later(f, abandoned))
                raise
            if not future.done() and future.cancel():
                raise StorageLockTimeoutError(
                    f"Timed out after {timeout}s waiting for LMDB's writer lock"
                )
            return future.result()
        finally:
            self._unstamp(token)

    def _abandoned_later(self, future: Future, abandoned: Callable[[Any], None]) -> None:
        """Queue ``abandoned`` on a landed result on the writer thread, never here.

        Runs on the thread that completed the future (the writer thread,
        or the requester's if it was already done): either way it only
        queues the job, so an abort never runs off the writer thread. The
        result is in by then, so ``result()`` never waits.
        """
        try:
            result = future.result()
        except BaseException:  # the call failed: nothing landed
            return
        self.fire(lambda: abandoned(result))

    def stop(self, timeout: float) -> bool:
        """Ask the thread to exit after its queued jobs and wait for it.

        Returns:
            False if it is still alive after ``timeout``: it is blocked
            inside LMDB (a begin behind a writer that does not go through
            virtuals). It gets a fresh queue in any case, so a later start
            never shares jobs with the stuck thread, which drains and
            exits once LMDB lets it go.
        """
        with self._lock:
            self._stopped = True
            thread, self._thread = self._thread, None
            jobs, self._queue = self._queue, queue.SimpleQueue()
        if thread is None:
            return True
        jobs.put(None)
        thread.join(timeout)
        return not thread.is_alive()

    def forget(self) -> None:
        """Forget the thread in a forked child, where it does not exist."""
        self._lock = threading.RLock()
        self._queue = queue.SimpleQueue()
        self._thread = None
        self._stopped = False
        self._waiting = False  # the parent's requester, not the child's
        self._wait_since = 0.0
        self._soft_since = 0.0


class WriterGate:
    """Cross-process writer gate: an ``flock`` on a file next to the environment.

    The in-process writer slot is taken first, so at most one thread per
    process is ever acquiring the gate. A free gate is taken at once with a
    non-blocking ``flock``. A held one is waited for by the gate's own
    helper thread in a blocking ``flock``, which the kernel wakes as soon
    as the holder releases. The requester waits on a future, bounded; if
    it gives up, the helper releases the lock when it lands, unless the
    next requester has attached to the wait meanwhile. Two ``flock`` calls
    are never in flight on the descriptor at once: a second one would
    convert the first's lock and both would think they own it.

    Fairness is the kernel's: on release every waiter is woken and the
    first to lock wins, so a process that writes in a tight loop can keep
    winning for a while. Waits stay bounded by the requester's timeout.
    """

    __slots__ = (
        "_consumer",
        "_fd",
        "_generation",
        "_mu",
        "_path",
        "_requests",
        "_thread",
        "_wait_fd",
        "_waiting",
    )

    def __init__(self, path: Path) -> None:
        """Initialize an unopened gate.

        Args:
            path: The lock file. Created on ``open`` if missing.
        """
        self._path = path
        self._fd: int | None = None
        self._mu = threading.Lock()
        self._generation = 0  # bumped per open: a wait from an older one touches nothing
        self._waiting = False  # a blocking wait is in flight for this generation
        self._consumer: Future | None = None  # the requester attached to that wait
        self._wait_fd: int | None = None  # the helper's duplicate for that wait
        self._requests: queue.SimpleQueue = queue.SimpleQueue()
        self._thread: threading.Thread | None = None

    def open(self) -> None:
        """Open (and create) the lock file."""
        with self._mu:
            if self._fd is None:
                self._fd = os.open(self._path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o644)
                self._generation += 1
                self._waiting = False
                self._consumer = None

    def wake_waiters(self, error: Exception) -> None:
        """Fail the requester waiting for the gate, e.g. because the storage closes."""
        with self._mu:
            consumer, self._consumer = self._consumer, None
        if consumer is not None:
            consumer.set_exception(error)

    def close(self) -> None:
        """Close the lock file, dropping the lock if held, fail any waiter, stop the helper."""
        self.wake_waiters(StorageClosedError("Storage was closed while waiting for the write lock"))
        with self._mu:
            fd, self._fd = self._fd, None
            self._generation += 1  # a wait in flight on the old descriptor touches nothing
            self._waiting = False
            requests, self._requests = self._requests, queue.SimpleQueue()
            self._thread = None
        requests.put(None)  # the helper exits after the wait it may be in
        if fd is not None:
            # A blocking wait still in flight holds its own duplicate of this
            # descriptor: the lock it may land goes away with that, and it
            # is ignored by generation.
            os.close(fd)

    def reset_after_fork(self) -> None:
        """Fresh lock, queue and no helper in a forked child (``register_at_fork``).

        Takes no lock: a parent thread may hold the inherited one forever.
        The descriptor is replaced by ``reopen_after_fork`` on first use.
        A wait in flight at the fork left its duplicate descriptor in the
        child; it is closed here, or the lock it lands in the parent would
        live on in the child after the parent's death (an ``flock`` lock
        lasts as long as any descriptor on its open file description).
        """
        self._mu = threading.Lock()
        self._thread = None
        self._requests = queue.SimpleQueue()
        self._waiting = False
        self._consumer = None
        fd, self._wait_fd = self._wait_fd, None
        if fd is not None:
            try:
                os.close(fd)
            except OSError:  # closed by the helper between its clear and the fork
                pass

    def reopen_after_fork(self) -> None:
        """Replace the inherited descriptor: it shares the parent's lock state."""
        with self._mu:
            fd, self._fd = self._fd, None
        if fd is not None:
            # The parent's descriptor keeps the open file description (and any
            # lock on it) alive, so closing this one takes nothing from it.
            os.close(fd)
        self.open()

    def _try_nb(self) -> bool:
        """Take the gate if free, without waiting. Under ``_mu``, no wait in flight."""
        if self._fd is None:
            raise StorageClosedError("Storage is not open")
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        except OSError as e:
            raise StorageError(f"Failed to take the LMDB writer gate: {e}") from e
        return True

    def _attach(self) -> Future | None:
        """Take the gate now (None) or return the future the helper will deliver it on."""
        with self._mu:
            if not self._waiting:
                if self._try_nb():
                    return None
                self._waiting = True
                self._start_wait()
            consumer: Future = Future()
            consumer.set_running_or_notify_cancel()
            self._consumer = consumer
            return consumer

    def _start_wait(self) -> None:
        """Hand the helper thread a blocking wait for this generation. Under ``_mu``.

        The helper gets its own duplicate of the descriptor (same open file
        description, so the same lock), never the number: close() may free
        that number and any later open in the process may reuse it.
        """
        thread = self._thread
        if thread is None or not thread.is_alive():
            thread = threading.Thread(
                target=self._serve, args=(self._requests,), name="lmdb-gate", daemon=True
            )
            self._thread = thread
            thread.start()
        fd = os.dup(self._fd)
        self._wait_fd = fd  # for a forked child to close (``reset_after_fork``)
        self._requests.put((fd, self._generation))

    def _serve(self, requests: queue.SimpleQueue) -> None:
        while True:
            request = requests.get()
            if request is None:
                return
            fd, generation = request
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                error: Exception | None = None
            except OSError as e:
                error = StorageError(f"Failed to take the LMDB writer gate: {e}")
            consumer: Future | None = None
            with self._mu:
                if generation == self._generation:
                    self._waiting = False
                    consumer, self._consumer = self._consumer, None
                    if consumer is None and error is None:
                        fcntl.flock(fd, fcntl.LOCK_UN)  # nobody wants it any more
                # else: the gate was closed meanwhile, and closing this last
                # duplicate drops the lock with the file.
                if self._wait_fd == fd:
                    self._wait_fd = None  # cleared first: a child never closes a reused number
                os.close(fd)  # the storage's own descriptor keeps a landed lock
            if consumer is None:
                continue
            if error is not None:
                consumer.set_exception(error)
            else:
                consumer.set_result(True)

    def _detach(self, consumer: Future) -> bool:
        """Give up waiting. False if the gate landed on ``consumer`` first (it is ours)."""
        with self._mu:
            if self._consumer is consumer:
                self._consumer = None
                return True
        return False

    def release(self) -> None:
        """Give the gate back. A no-op once the file is closed."""
        with self._mu:
            fd = self._fd
            if fd is None:
                return
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError as e:
                logger.error(f"Failed to release the LMDB writer gate: {e}")

    def acquire(self, timeout: float) -> None:
        """Block for the gate up to ``timeout`` seconds.

        Raises:
            StorageLockTimeoutError: If a peer holds it past ``timeout``.
            StorageClosedError: If the storage closes while waiting.
        """
        consumer = self._attach()
        if consumer is None:
            return
        try:
            consumer.result(max(timeout, 0))
        except FutureTimeoutError:
            if self._detach(consumer):
                raise self._timed_out(timeout) from None
            consumer.result()  # landed as the wait ended: ours
        except BaseException:
            # Left by a signal or an interrupt: the lock must not land on a
            # consumer nobody reads (the gate would then be held by no one
            # until this process's next write), same as ``aacquire``.
            if not self._detach(consumer):
                try:
                    consumer.result()
                except Exception:  # noqa: S110 (failed, nothing held)
                    pass
                else:
                    self.release()  # landed, but the requester is gone
            raise

    async def aacquire(self, timeout: float) -> None:
        """Like ``acquire`` but waits on the event loop instead of blocking it."""
        consumer = self._attach()
        if consumer is None:
            return
        try:
            await _await_done(consumer, timeout)
        except BaseException:
            if not self._detach(consumer):
                try:
                    consumer.result()
                except Exception:  # noqa: S110 (failed, nothing held)
                    pass
                else:
                    self.release()  # landed, but the requester is gone
            raise
        if not consumer.done() and self._detach(consumer):
            raise self._timed_out(timeout)
        consumer.result()

    @staticmethod
    def _timed_out(timeout: float) -> StorageLockTimeoutError:
        return StorageLockTimeoutError(
            f"Timed out after {timeout}s waiting for the write lock held by another process"
        )
