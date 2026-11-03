"""The writer thread and the cross-process gate behind LMDB write transactions.

LMDB's writer lock is a robust pthread mutex in the lock file, owned by the
thread that took it in ``begin`` and only unlockable by that thread. It has
no timed variant. Two helpers keep virtuals clear of both edges:

- ``WriterThread``: one long-lived thread per storage runs every
  ``env.begin(write=True)``, ``commit`` and ``abort``. Callers hand it the
  call and wait on a future, bounded for begin, so a write transaction can
  be begun on one thread and finished on any other without touching the
  mutex from the wrong thread. A begin whose requester gave up is aborted
  by the thread the moment it returns.
- ``WriterGate``: an ``flock`` on a file next to the environment, taken
  before ``env.begin(write=True)`` and released after the transaction ends.
  Peers that go through virtuals wait on it with a bounded poll (a sleep
  in a thread, ``asyncio.sleep`` on a loop) instead of blocking inside
  LMDB's mutex, so the writer thread's begin never waits behind a virtuals
  peer, and ``close()`` never has to tear down an environment with a begin
  in flight. The kernel drops the lock when its holder dies.
"""

from __future__ import annotations

import asyncio
import fcntl
import os
import queue
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


# Poll interval bounds while waiting for the gate (seconds), the same shape
# as the in-process writer slot's.
_GATE_POLL_MIN = 0.0005
_GATE_POLL_MAX = 0.005


class WriterThread:
    """One thread that owns begin, commit and abort of a storage's write transactions.

    Started on first use, stopped by ``stop``. Jobs run in order; one is in
    flight at a time, which is all LMDB allows anyway (one write
    transaction per environment).
    """

    __slots__ = ("_lock", "_name", "_queue", "_thread")

    def __init__(self, name: str) -> None:
        """Initialize an idle writer thread.

        Args:
            name: Thread name, for diagnostics.
        """
        self._name = name
        self._lock = threading.Lock()
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._thread: threading.Thread | None = None

    def _ensure_started(self) -> threading.Thread:
        with self._lock:
            thread = self._thread
            if thread is None or not thread.is_alive():
                thread = threading.Thread(
                    target=self._loop, args=(self._queue,), name=self._name, daemon=True
                )
                self._thread = thread
                thread.start()
            return thread

    @staticmethod
    def _loop(jobs: queue.SimpleQueue) -> None:
        while True:
            job = jobs.get()
            if job is None:
                return
            fn, future, undo = job
            try:
                result = fn()
            except BaseException as e:
                if future.set_running_or_notify_cancel():
                    future.set_exception(e)
                continue
            # The requester may have given up while ``fn`` ran. Marking the
            # future running after the call makes that race explicit: either
            # the requester's cancel wins and ``undo`` takes the result back
            # (an abandoned begin is aborted at once), or the result lands.
            if future.set_running_or_notify_cancel():
                future.set_result(result)
            elif undo is not None:
                try:
                    undo(result)
                except Exception as e:
                    logger.error(f"Undo of an abandoned LMDB writer call failed: {e}")

    def run(
        self,
        fn: Callable[[], Any],
        *,
        timeout: float | None = None,
        undo: Callable[[Any], None] | None = None,
    ) -> Any:  # noqa: ANN401
        """Run ``fn`` on the writer thread and return its result.

        Blocks the caller. With ``timeout`` the wait is bounded: a call
        still in flight then is abandoned (``undo`` runs on its result when
        it lands) and ``StorageLockTimeoutError`` is raised. A result that
        lands just as the wait ends is taken, not dropped.
        """
        if threading.current_thread() is self._thread:
            return fn()  # a job calling back in (e.g. a finalizer); never wait on ourselves
        self._ensure_started()
        future: Future = Future()
        self._queue.put((fn, future, undo))
        try:
            return future.result(timeout)
        except FutureTimeoutError:
            if future.cancel():
                raise StorageLockTimeoutError(
                    f"Timed out after {timeout}s waiting for LMDB's writer lock"
                ) from None
            return future.result()

    async def arun(
        self,
        fn: Callable[[], Any],
        *,
        timeout: float,
        undo: Callable[[Any], None] | None = None,
    ) -> Any:  # noqa: ANN401
        """Like ``run`` but waits without blocking the event loop.

        Cancelled while waiting, it abandons the call: ``undo`` takes back a
        result that lands later. No cancel of the wait ever loses a result.
        """
        self._ensure_started()
        future: Future = Future()
        loop = asyncio.get_running_loop()
        done = loop.create_future()

        def wake(_: Future) -> None:
            try:
                loop.call_soon_threadsafe(_set_done, done)
            except RuntimeError:  # loop closed: the requester is gone
                pass

        future.add_done_callback(wake)
        self._queue.put((fn, future, undo))
        try:
            # ``asyncio.wait`` (unlike ``wait_for``) leaves the futures alone
            # on timeout, so a result that lands is never dropped by cancel.
            await asyncio.wait({done}, timeout=max(timeout, 0))
        except BaseException:
            if not future.cancel() and undo is not None:
                try:
                    undo(future.result())
                except Exception:  # noqa: S110 (begin failed: nothing to undo)
                    pass
            raise
        if not future.done() and future.cancel():
            raise StorageLockTimeoutError(
                f"Timed out after {timeout}s waiting for LMDB's writer lock"
            )
        return future.result()

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
            thread, self._thread = self._thread, None
            jobs, self._queue = self._queue, queue.SimpleQueue()
        if thread is None:
            return True
        jobs.put(None)
        thread.join(timeout)
        return not thread.is_alive()

    def forget(self) -> None:
        """Forget the thread in a forked child, where it does not exist."""
        self._lock = threading.Lock()
        self._queue = queue.SimpleQueue()
        self._thread = None


def _set_done(done: asyncio.Future) -> None:
    if not done.done():
        done.set_result(True)


class WriterGate:
    """Cross-process writer gate: an ``flock`` on a file next to the environment."""

    __slots__ = ("_fd", "_path")

    def __init__(self, path: Path) -> None:
        """Initialize an unopened gate.

        Args:
            path: The lock file. Created on ``open`` if missing.
        """
        self._path = path
        self._fd: int | None = None

    def open(self) -> None:
        """Open (and create) the lock file."""
        if self._fd is None:
            self._fd = os.open(self._path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o644)

    def close(self) -> None:
        """Close the lock file, dropping the lock if held."""
        fd, self._fd = self._fd, None
        if fd is not None:
            os.close(fd)

    def reopen_after_fork(self) -> None:
        """Replace the inherited descriptor: it shares the parent's lock state."""
        fd, self._fd = self._fd, None
        if fd is not None:
            # The parent's descriptor keeps the open file description (and any
            # lock on it) alive, so closing this one takes nothing from it.
            os.close(fd)
        self.open()

    def try_acquire(self) -> bool:
        """Take the gate if it is free, without waiting."""
        if self._fd is None:
            raise StorageClosedError("Storage is not open")
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        except OSError as e:
            raise StorageError(f"Failed to take the LMDB writer gate: {e}") from e
        return True

    def release(self) -> None:
        """Give the gate back. A no-op once the file is closed."""
        if self._fd is None:
            return
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        except OSError as e:
            logger.error(f"Failed to release the LMDB writer gate: {e}")

    def acquire(self, timeout: float, still_open: Callable[[], bool]) -> None:
        """Block for the gate up to ``timeout`` seconds, polling with a growing sleep.

        Raises:
            StorageLockTimeoutError: If a peer holds it past ``timeout``.
            StorageClosedError: If the storage closes while waiting.
        """
        deadline = time.monotonic() + max(timeout, 0)
        delay = _GATE_POLL_MIN
        while not self.try_acquire():
            if not still_open():
                raise StorageClosedError("Storage was closed while waiting for the write lock")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise self._timed_out(timeout)
            time.sleep(min(delay, remaining))
            delay = min(delay * 2, _GATE_POLL_MAX)

    async def aacquire(self, timeout: float, still_open: Callable[[], bool]) -> None:
        """Like ``acquire`` but sleeps on the event loop instead of blocking it."""
        deadline = time.monotonic() + max(timeout, 0)
        delay = _GATE_POLL_MIN
        while not self.try_acquire():
            if not still_open():
                raise StorageClosedError("Storage was closed while waiting for the write lock")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise self._timed_out(timeout)
            await asyncio.sleep(min(delay, remaining))
            delay = min(delay * 2, _GATE_POLL_MAX)

    @staticmethod
    def _timed_out(timeout: float) -> StorageLockTimeoutError:
        return StorageLockTimeoutError(
            f"Timed out after {timeout}s waiting for the write lock held by another process"
        )
