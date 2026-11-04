"""LMDB storage backend implementation.

Provides persistent key-value storage over LMDB with transactions,
MVCC snapshots, and optional change notifications via a publisher.
"""

from __future__ import annotations

import asyncio
import ctypes
import functools
import os
import threading
import time
import weakref
from contextlib import contextmanager
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, overload

from virtuals.tkv.storage import (
    SnapshotProtocol,
    StorageClosedError,
    StorageError,
    StorageLockTimeoutError,
    StorageOperationError,
    TransactionProtocol,
    WriteBatchProtocol,
)

from .._write_slot import LOOP_THREAD_WAIT_CAP, WriterLock, WriteSlot
from ._writer import WriterGate, WriterThread
from .context import WRITER_BACKOFF_STEPS, replay_pending, yield_to_writer
from .snapshot import LMDBSnapshot
from .transaction import LMDBTransaction
from .write_batch import LMDBWriteBatch


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from types import TracebackType

    from virtuals.tkv.codec import CodecProtocol
    from virtuals.tkv.publisher import PublisherProtocol
    from virtuals.tkv.types import Key

    from .context import Pending


try:
    import lmdb
except ImportError as e:
    raise ImportError(
        "LMDB python bindings are required for LMDBStorage. Install via: [uv] pip install lmdb"
    ) from e


__all__ = ["LMDBStorage", "LMDBWriteSlot"]


logger = getLogger(__name__)


DEFAULT_MAP_SIZE = 10 * 1024 * 1024 * 1024  # 10 GiB
# Reader slots are shared by every process on the environment and each
# open snapshot takes one (py-lmdb opens with MDB_NOTLS). A slot is a
# cache line in the lock file, so a large table costs nothing to speak
# of, while a full one turns every new snapshot away.
DEFAULT_MAX_READERS = 1024
DEFAULT_BUSY_TIMEOUT = 60.0  # seconds
# How often, at most, a write begin checks for reader slots of dead
# processes (seconds). Such a slot pins every page freed since its
# transaction began, so without the check the file grows until MAP_FULL.
READER_CHECK_INTERVAL = 1.0


class LMDBWriteSlot(WriteSlot):
    """The LMDB writer slot, held with the cross-process gate and a begun LMDB write txn.

    ``areserve_write_slot`` does all the waiting (in-process slot, then
    the gate, then the begin on the writer thread) without blocking the
    event loop, so ``begin_transaction(write_slot=...)`` only wraps the
    transaction already begun. Released untaken, the slot aborts it.
    """

    # ``__weakref__`` so ``_reserved_slots`` can hold it weakly; allowed here
    # only because ``WriteSlot`` declares slots without one.
    __slots__ = ("__weakref__", "_lmdb_txn", "_pid")

    def __init__(self, storage: LMDBStorage, lock: threading.Lock, lmdb_txn: Any) -> None:  # noqa: ANN401
        """Initialize a held slot carrying a begun LMDB write transaction."""
        super().__init__(storage, lock)
        self._lmdb_txn: Any | None = lmdb_txn
        self._pid = os.getpid()

    def release(self) -> None:
        """Give the slot back, aborting its transaction, if no transaction took it.

        In a forked child the slot is the parent's: its transaction is
        left to the parent and the inherited gate is not touched (the
        child's ``_check_fork`` has or will have reset both).
        """
        if self._state != "held":
            return
        self._state = "released"
        if os.getpid() != self._pid:
            self._storage._check_fork()  # type: ignore[attr-defined]
            return
        self._storage._release_reserved(self)  # type: ignore[attr-defined]

    def __del__(self) -> None:
        """Last resort: give back a held slot that became unreachable unreleased.

        The abort is queued, never waited for (see ``WriterThread.try_fire``).
        Inherited across a fork, the slot is left alone.
        """
        try:
            if self._state == "held" and os.getpid() == self._pid:
                self._state = "released"
                self._storage._release_reserved(self, finalizing=True)  # type: ignore[attr-defined]
        except Exception:  # noqa: S110 (interpreter shutdown, half-built object)
            pass

    def _take(self, storage: object) -> None:
        """Hand the slot to a transaction being begun on ``storage``.

        Refuses a stale slot, one whose lock is no longer the storage's
        writer lock because the storage was closed (or the process
        forked) since it was reserved, and gives back what it may still
        carry before the refusal: a begun transaction is aborted on the
        writer thread, queued and not waited for, since this runs under
        the writer lock's guard. The gate is never a stale slot's to
        release: a reservation that straddled a close never comes back
        with one (``areserve_write_slot``), and close() drops the gate
        it held. In a forked child the slot is the parent's and is left
        alone.
        """
        if self._state == "held" and self._lock is not self._storage._writer.lock:
            self._state = "released"
            if os.getpid() == self._pid:
                self._storage._release_reserved(self, finalizing=True)  # type: ignore[attr-defined]
            raise StorageError("Write slot is stale: the storage was closed after it was reserved")
        super()._take(storage)


# Storages with locks to replace in a forked child (see ``_after_fork_in_child``).
_storages: weakref.WeakSet[LMDBStorage] = weakref.WeakSet()

# py-lmdb objects inherited across a fork, kept alive for the life of the
# child (see ``_park_inherited``).
_INHERITED: list[Any] = []
# The inherited environment per path. In a forked child the path is blocked
# for good: py-lmdb refuses a second open of it in this process while the
# parked one lives, and the parked one is never used nor closed here.
_PARKED_ENVS: dict[str, Any] = {}

_FORK_MESSAGE = (
    "This LMDB environment belongs to the parent process: it was open when this process "
    "forked. Using it here is unsafe (its snapshots are torn by the parent's close and "
    "its reader slots are never freed), and py-lmdb cannot open the path again in this "
    "process. Use spawn, not fork, for processes that use LMDB, or open the storage only "
    "after forking"
)


def _park_inherited(objects: list[Any]) -> None:
    """Keep inherited environments and transactions alive, untouched, for good.

    Deallocating an inherited py-lmdb transaction runs ``mdb_txn_abort``
    and an environment ``mdb_env_close``; both write to the lock file the
    parent still uses, clearing its reader slots (a snapshot it holds
    loses the pages it reads: a torn read there) and, for the environment,
    sometimes hanging the child. An extra reference the interpreter never
    gives back makes them immortal, past interpreter shutdown too (a
    normal exit clears module globals), so only process exit frees them,
    which touches nothing shared.
    """
    for obj in objects:
        ctypes.pythonapi.Py_IncRef(ctypes.py_object(obj))
    _INHERITED.extend(objects)


def _after_fork_in_child() -> None:
    """Fresh locks in the child, then forget what the parent had open.

    Runs before any user code in the child (``register_at_fork``). A
    parent thread may hold the inherited locks forever, so they are
    replaced first; then ``_check_fork`` parks every inherited handle
    before anything could end or collect one.
    """
    for storage in list(_storages):
        storage._reset_locks_after_fork()
        try:
            storage._check_fork()
        except Exception as e:  # retried lazily by the next storage call
            logger.error(f"LMDB fork handling in the child failed: {e}")


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)


class LMDBStorage:
    """LMDB storage implementation conforming to StorageProtocol.

    Provides persistent key-value storage with MVCC snapshots and
    single-writer serialized transactions.

    LMDB semantics worth knowing:
      - Only one write transaction may be active per environment at a
        time. LMDB's own writer lock has no timeout and belongs to the
        thread that took it.
      - Read transactions are MVCC snapshots and never block writers.
      - `map_size` sets the maximum on-disk size; the mmap is grown to
        `map_size` even on empty databases.

    Writers in this process first take an in-process writer slot, the same
    one SQLite uses, so threads queue on it for up to ``busy_timeout`` and
    then raise ``StorageLockTimeoutError`` instead of LMDB refusing a
    second writer. Then they take a cross-process gate (an ``flock`` on
    ``writer.lock`` next to the environment) with the same bound, so a
    writer in another process surfaces as ``StorageLockTimeoutError`` too
    instead of an unbounded block inside LMDB. A transaction holds both
    from ``begin_transaction`` to its end; a write batch, buffered in
    Python like the other backends' batches, only for its ``write``. The
    LMDB begin, commit and abort themselves run on one writer thread per
    storage, the only thread that ever touches LMDB's writer lock, so a
    transaction may be begun on one thread and committed or aborted on any
    other; the LMDB transaction is begun lazily, at the first read or at
    commit, so a write-only transaction or a batch costs one hop to that
    thread. Two consequences: a full map surfaces at the commit of a
    write-only transaction, not at its puts (a key LMDB would refuse is
    still refused at the put), and the first ``get``, ``exists`` or
    ``scan`` of a transaction is where the wait for the writer thread
    happens, so it can raise the retryable ``StorageLockTimeoutError``
    (after ``busy_timeout``, or ``LOOP_THREAD_WAIT_CAP`` on an event loop
    thread) when that thread is stuck behind a writer outside virtuals;
    the transaction stays pending and may be retried or aborted. A
    signal or ``KeyboardInterrupt`` that lands in such a wait aborts the
    begin on the writer thread once it lands; one that lands in a
    commit's wait never changes the commit's outcome or how it is
    reported: the commit is waited out, the transaction ends committed
    (closed, notified, slot and gate given back) and the interrupt is
    then raised as it is, so a caller never sees a failure for a commit
    that happened. An interrupt before the commit started aborts the
    transaction and is raised the same way. The bookkeeping around the
    waits is a few bytecodes, not zero: an interrupt that lands there
    can leave the slot, the gate or a begun write transaction held in a
    process that keeps running, and every later writer times out until
    the process is restarted. Signal handlers are never touched.
    Snapshots
    never touch the slot or the gate, so a waiting writer never delays a
    reader; the one exception is a writer starved of the GIL by reader
    threads spinning on short calls, which readers answer with brief
    sleeps (see ``yield_to_writer``).

    Coroutines: the slot belongs to a thread and, on an event loop, to the
    task that took it. Coroutines sharing one loop wait for it with
    ``areserve_write_slot``, which also waits for the gate and begins the
    LMDB transaction, all without blocking the loop, and hand the slot to
    ``begin_transaction(write_slot=...)``. Threads compete for the slot's
    lock and take it the instant it is free, so a thread writing in a
    loop barges ahead of other threads; a coroutine is handed the slot at
    a release instead, and threads and coroutines take turns at each
    release, so a queued coroutine waits at most one thread transaction
    per coroutine ahead of it. A second writer from the task or thread
    that already holds
    the slot raises ``StorageNestedTransactionError``; a sync begin while
    another task on the same loop holds it raises
    ``StorageLockTimeoutError`` at once. A sync begin or batch write from
    a coroutine never stalls its loop for long: on an event loop thread
    each wait (the slot, the gate, the begin on the writer thread) is
    capped at ``LOOP_THREAD_WAIT_CAP``, whether the holder is a thread or
    another process, and surfaces as the retryable
    ``StorageLockTimeoutError``.

    close() ends open contexts from the closing thread. One whose owner is
    inside a call is ended after that call returns, never under it, and
    the owner's next call raises ``StorageClosedError``.

    Multiple processes: open the storage in each process with the
    ``spawn`` start method; that is the only supported path. A process
    forked while this storage is open must not use LMDB at all, as LMDB
    itself requires: the inherited environment carries the parent's pid,
    so the parent's close tears any snapshot the child holds and no
    reader check ever frees a slot the child leaves behind, and py-lmdb
    refuses to open the path again in the child. So in a forked child
    every new transaction, snapshot, batch or reservation on the
    inherited storage raises a non-retryable ``StorageError`` saying to
    use spawn, and so does ``open()`` on that path (this object or a new
    one). Whatever the parent had open at the fork is closed in the child
    without touching LMDB, and ``close()`` there parks the environment
    instead of closing it: both would clear the parent's reader slots.

    The environment must live on a local filesystem, as LMDB requires (its
    lock file is shared memory and its data file is mmapped); the gate is
    an ``flock`` on that same disk and is never relied on across hosts. A
    process writing in a tight loop can keep re-taking the gate ahead of a
    woken peer for a while; peers' waits stay bounded by ``busy_timeout``.

    ``map_size`` is a ceiling every process should agree on. If a peer
    opened the environment with a larger map and grew the file past this
    process's mapping, the next begin here remaps to the file's size, which
    LMDB only allows while this process has no open snapshot or
    transaction; otherwise it raises ``StorageError`` asking to close them.
    """

    process_model: ClassVar[Literal["single", "multi"]] = "multi"

    def __init__(
        self,
        path: Path | str,
        codec: CodecProtocol[bytes, bytes],
        publisher: PublisherProtocol | None = None,
        *,
        read_only: bool = False,
        map_size: int = DEFAULT_MAP_SIZE,
        max_readers: int = DEFAULT_MAX_READERS,
        subdir: bool = True,
        sync: bool = True,
        metasync: bool = True,
        writemap: bool = False,
        lock: bool = True,
        create_if_missing: bool = True,
        env_options: dict[str, Any] | None = None,
        busy_timeout: float = DEFAULT_BUSY_TIMEOUT,
    ) -> None:
        """Initialize LMDB storage.

        Args:
            path: Database directory (subdir=True) or file (subdir=False).
            codec: Codec for key/value encoding.
            publisher: Optional publisher for change notifications.
            read_only: Open database in read-only mode.
            map_size: Maximum on-disk size in bytes (mmap size).
            max_readers: Maximum concurrent reader slots.
            subdir: If True, `path` is a directory containing the
                environment files; if False, `path` is the environment
                file itself.
            sync: If True, fsync data pages after each commit.
            metasync: If True, fsync the metapage after each commit.
            writemap: If True, use a writable mmap (higher throughput,
                lower durability on crash).
            lock: If True, use LMDB's file locking (and the cross-process
                writer gate).
            create_if_missing: Create database if it doesn't exist.
            env_options: Extra kwargs passed through to `lmdb.open()`.
            busy_timeout: Seconds a writer waits for the writer slot, and
                then for the cross-process gate, before
                ``StorageLockTimeoutError``.
        """
        self._codec = codec
        self._publisher = publisher

        self._read_only = read_only
        self._path = Path(path) if isinstance(path, str) else path
        self._map_size = map_size
        self._max_readers = max_readers
        self._subdir = subdir
        self._sync = sync
        self._metasync = metasync
        self._writemap = writemap
        self._lock = lock
        self._create_if_missing = create_if_missing
        self._env_options = dict(env_options or {})
        self._busy_timeout = busy_timeout
        self._writer = WriterLock("LMDB")
        self._writer_thread = WriterThread(f"lmdb-writer:{self._path.name}")
        gate_path = self._path / "writer.lock" if subdir else Path(f"{self._path}-writer.lock")
        self._gate: WriterGate | None = None if read_only or not lock else WriterGate(gate_path)
        self._gate_held = False

        self._env: lmdb.Environment | None = None
        self._env_lock = threading.RLock()
        # Snapshots begin, register and unregister under their own lock, so
        # reader threads never queue writers behind an env.begin() in
        # flight (a Python lock held across a C call is a GIL convoy, see
        # yield_to_writer). Taken before _env_lock wherever both are held.
        self._snap_lock = threading.RLock()
        # Weak: a context nobody ended is collected. A snapshot's read
        # transaction then ends under py-lmdb, freeing its shared reader
        # slot instead of pinning it (and the pages it saw) until close();
        # a transaction or batch is aborted by its finalizer on the writer
        # thread and gives the slot back.
        self._active_transactions: weakref.WeakSet[LMDBTransaction] = weakref.WeakSet()
        self._active_write_batches: weakref.WeakSet[LMDBWriteBatch] = weakref.WeakSet()
        self._active_snapshots: weakref.WeakSet[LMDBSnapshot] = weakref.WeakSet()
        # Slots reserved and not yet taken: each carries a begun write txn
        # that close() must end on the writer thread.
        self._reserved_slots: weakref.WeakSet[LMDBWriteSlot] = weakref.WeakSet()
        # Every live LMDB write txn: added by the writer thread's begin job,
        # removed by the writer thread's end job, nothing else. close()
        # ends whatever is still here before stopping the thread and
        # closing the environment, whichever path (context, slot,
        # finalizer, abandoned begin) lost the race to end it first.
        self._live_writes: set[Any] = set()
        self._opened = False
        self._pid = os.getpid()
        self._env_key = ""  # set by open()
        self._max_key_size = 511
        self._forked = False  # inherited across a fork: the env is the parent's
        self._last_reader_check = 0.0
        self._snapshot_yielded_at = 0.0  # shared by every thread opening snapshots; benign races
        _storages.add(self)

    @property
    def read_only(self) -> bool:
        """Storage access mode."""
        return self._read_only

    @property
    def codec(self) -> CodecProtocol[bytes, bytes]:
        """Get codec for key/value encoding."""
        return self._codec

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def open(self) -> None:
        """Open LMDB environment and initialize resources."""
        if self._opened:
            return

        with self._env_lock:
            try:
                if self._create_if_missing and self._subdir and not self._read_only:
                    self._path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise StorageError(f"Failed to create database directory: {e}") from e

            open_kwargs: dict[str, Any] = {
                "path": str(self._path),
                "map_size": self._map_size,
                "max_readers": self._max_readers,
                "subdir": self._subdir,
                "readonly": self._read_only,
                "sync": self._sync,
                "metasync": self._metasync,
                "writemap": self._writemap,
                "lock": self._lock,
                "create": self._create_if_missing and not self._read_only,
            }
            open_kwargs.update(self._env_options)

            self._env_key = self._path_key()
            if self._env_key in _PARKED_ENVS:
                raise StorageError(_FORK_MESSAGE)
            try:
                self._env = lmdb.open(**open_kwargs)
            except Exception as e:
                raise StorageError(f"Failed to open LMDB environment: {e}") from e
            try:
                self._max_key_size = self._env.max_key_size()
            except Exception:  # older bindings
                self._max_key_size = 511

            # Reader slots of processes that died mid-snapshot are shared by
            # every opener and never swept by LMDB itself.
            self._sweep_dead_readers()

            if self._gate is not None:
                try:
                    self._gate.open()
                except OSError as e:
                    self._env.close()
                    self._env = None
                    raise StorageError(f"Failed to open the LMDB writer gate: {e}") from e
            self._gate_held = False
            self._writer_thread.start()

            self._pid = os.getpid()
            self._forked = False
            self._opened = True

    def _path_key(self) -> str:
        """The file identity ``_PARKED_ENVS`` is keyed by."""
        try:
            return str(self._path.resolve())
        except OSError:
            return str(self._path)

    def _key_fits(self, encoded_key: bytes) -> bool:
        """Whether LMDB can store a key of this size (1 to ``max_key_size`` bytes)."""
        return 0 < len(encoded_key) <= self._max_key_size

    def _check_key(self, encoded_key: bytes, key: Key) -> None:
        """Refuse a key LMDB would refuse, so a buffered put fails where an eager one did."""
        if not self._key_fits(encoded_key):
            raise StorageOperationError(
                f"Failed to put key {key}: encoded key is {len(encoded_key)} bytes, LMDB "
                f"takes 1 to {self._max_key_size}"
            )

    def _sweep_dead_readers(self) -> int:
        """Clear reader slots left by dead processes; returns how many."""
        env = self._env
        if env is None:
            return 0
        self._last_reader_check = time.monotonic()
        try:
            cleared = env.reader_check()
        except Exception as e:  # read-only lock file, or an env without locking
            logger.debug(f"LMDB reader check skipped: {e}")
            return 0
        if cleared:
            logger.info(f"LMDB reader check cleared {cleared} slot(s) of dead processes")
        return cleared

    def _after_map_full(self, what: str) -> StorageLockTimeoutError | None:
        """The retryable error for ``MDB_MAP_FULL`` behind a dead reader, else None.

        LMDB cannot reuse pages freed since the oldest reader's transaction
        began, and the slot of a process that died inside a snapshot stays
        until a reader check clears it. Cleared here, a retry has those
        pages back; a map that is full with no such slot is a hard limit
        and stays non-retryable.
        """
        if not self._sweep_dead_readers():
            return None
        return StorageLockTimeoutError(
            f"Failed to {what}: the LMDB map was full behind a reader slot of a dead "
            "process; the slot is cleared now, retry"
        )

    def _reset_locks_after_fork(self) -> None:
        """Replace locks a parent thread may hold at the fork (``register_at_fork``)."""
        self._env_lock = threading.RLock()
        self._snap_lock = threading.RLock()
        self._writer_thread.forget()
        if self._gate is not None:
            self._gate.reset_after_fork()

    def _check_fork(self) -> None:
        """Park what was inherited across a fork; the storage is unusable here.

        If this process is not the one that opened the storage, the
        parent's open contexts are forgotten, not ended: aborting them
        here would write to the lock file the parent still uses (its
        reader slots and writer mutex). Their handles and the environment
        are parked, never used again: the environment carries the parent's
        pid, so the parent's close clears any reader slot taken here (a
        torn snapshot) and no reader check ever frees one left behind,
        and py-lmdb refuses a second open of the path in this process.
        Every begin raises ``StorageError`` (see ``_require_not_forked``);
        ``close()`` only parks. The writer slot is reset so an inherited
        holder never blocks the error path, and the gate descriptor is
        reopened, since the inherited one shares the parent's lock.
        """
        if self._pid == os.getpid():
            return
        with self._snap_lock, self._env_lock:
            if self._pid == os.getpid():
                return
            self._forked = True
            # The parent's contexts are forgotten, not ended, and their
            # handles parked: ending or collecting one here would write to
            # the parent's reader slots and writer mutex in the lock file.
            inherited: list[Any] = [*self._live_writes]
            if self._env is not None:
                inherited.append(self._env)
                _PARKED_ENVS[self._env_key] = self._env
            for context in [
                *self._active_transactions,
                *self._active_snapshots,
                *self._active_write_batches,
            ]:
                handle = context._forget()
                if handle is not None:
                    inherited.append(handle)
            for slot in self._reserved_slots:
                handle, slot._lmdb_txn = slot._lmdb_txn, None
                slot._state = "released"
                if handle is not None:
                    inherited.append(handle)
            _park_inherited(inherited)
            self._active_transactions = weakref.WeakSet()
            self._active_write_batches = weakref.WeakSet()
            self._active_snapshots = weakref.WeakSet()
            self._reserved_slots = weakref.WeakSet()
            self._live_writes = set()
            self._writer.reset()
            self._writer_thread.forget()
            self._gate_held = False
            if self._gate is not None and self._opened:
                try:
                    self._gate.reopen_after_fork()
                except OSError as e:
                    logger.error(f"Failed to reopen the LMDB writer gate after fork: {e}")
            self._pid = os.getpid()

    def _require_not_forked(self) -> None:
        """Refuse every new context on a storage inherited across a fork."""
        if self._forked:
            raise StorageError(_FORK_MESSAGE)

    def close(self) -> None:
        """Close database and release all resources.

        Marks the storage closed first, so begins and gate waits in other
        threads fail with ``StorageClosedError``, then ends every open
        transaction, batch, reserved slot and snapshot (each on the writer
        thread where LMDB requires it, and never under a call its owner
        has in flight), stops the writer thread and closes the
        environment. If the writer thread is stuck inside LMDB behind a
        writer that does not go through virtuals, the environment is left
        open rather than torn down under it, with an error logged.
        """
        if not self._opened:
            return
        self._check_fork()  # never end contexts inherited from the parent

        with self._snap_lock, self._env_lock:
            if not self._opened:
                return
            self._opened = False
            transactions = list(self._active_transactions)
            write_batches = list(self._active_write_batches)
            snapshots = list(self._active_snapshots)
            for slot in self._reserved_slots:
                slot._lmdb_txn = None  # ended below with every other live write
            self._active_transactions.clear()
            self._active_write_batches.clear()
            self._active_snapshots.clear()
            self._reserved_slots.clear()
            if self._gate is not None:
                self._gate.wake_waiters(
                    StorageClosedError("Storage was closed while waiting for the write lock")
                )

        for transaction in transactions:
            try:
                transaction.abort()
            except Exception as e:
                logger.error(f"Transaction abort failed during close: {e}")

        for write_batch in write_batches:
            try:
                write_batch.abort()
            except Exception as e:
                logger.error(f"Write batch abort failed during close: {e}")

        for snapshot in snapshots:
            try:
                snapshot.close()
            except Exception as e:
                logger.error(f"Snapshot close failed during close: {e}")

        # Whatever is still live lost a race to its own end (a reserved
        # slot, a finalizer, an abandoned begin): end it here, on the
        # writer thread, so env.close() never meets a live write txn.
        with self._env_lock:
            live = list(self._live_writes)
        for lmdb_txn in live:
            try:
                self._end_write(lmdb_txn, commit=False)
            except Exception as e:
                logger.error(f"Transaction abort failed during close: {e}")

        self._writer.close()  # a reserved slot nobody took does not survive the reopen
        stopped = self._writer_thread.stop(self._busy_timeout)

        with self._snap_lock, self._env_lock:  # no snapshot begin in flight under env.close()
            if self._gate is not None:
                self._gate.close()  # drops the gate if a reserved slot still held it
            self._gate_held = False
            env, self._env = self._env, None
            if env is None:
                return
            if self._forked:
                # An inherited environment is never closed: mdb_env_close
                # clears the reader slots of its pid, the parent's, and
                # can hang the child. It was parked in _check_fork; the
                # child's own contexts were ended above as usual.
                return
            if not stopped:
                logger.error(
                    "LMDB writer thread is blocked inside LMDB behind a writer outside "
                    "virtuals; leaving the environment open rather than closing it under "
                    "that thread"
                )
                return
            try:
                env.close()
            except Exception as e:
                raise StorageError(f"Failed to close database: {e}") from e

    def __enter__(self) -> LMDBStorage:
        """Enter context manager - open storage."""
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager - close storage."""
        self.close()

    # =========================================================================
    # Transaction Management
    # =========================================================================

    @overload
    def begin(self, *, read_only: Literal[True]) -> SnapshotProtocol: ...

    @overload
    def begin(self, *, write_only: Literal[True]) -> WriteBatchProtocol: ...

    @overload
    def begin(
        self, *, read_only: Literal[False], write_only: Literal[False]
    ) -> TransactionProtocol: ...

    def begin(
        self,
        *,
        read_only: bool = False,
        write_only: bool = False,
    ) -> WriteBatchProtocol | SnapshotProtocol | TransactionProtocol:
        """Begin transaction, snapshot, or write batch."""
        if read_only:
            return self.begin_snapshot()
        elif write_only:
            return self.begin_write_batch()
        else:
            return self.begin_transaction()

    def begin_snapshot(self) -> LMDBSnapshot:
        """Begin read-only snapshot."""
        self._require_open()
        self._check_fork()
        self._require_not_forked()
        writer_thread = self._writer_thread
        if writer_thread._wait_since or writer_thread._soft_since:  # see yield_to_writer
            self._snapshot_yielded_at, _ = yield_to_writer(
                writer_thread, self._snapshot_yielded_at, WRITER_BACKOFF_STEPS
            )

        with self._snap_lock:
            self._require_open()
            try:
                lmdb_txn = self._begin_read()
            except StorageError:
                raise
            except Exception as e:
                raise StorageError(f"Failed to begin snapshot: {e}") from e

            snapshot = LMDBSnapshot(self, lmdb_txn)
            self._active_snapshots.add(snapshot)
            return snapshot

    def _begin_read(self) -> Any:  # noqa: ANN401
        """Begin an LMDB read transaction, recovering once from a full reader table or a grown map."""
        env = self._env
        if env is None:
            raise StorageClosedError("Storage is not open")
        try:
            return env.begin(write=False)
        except lmdb.ReadersFullError as e:
            if self._sweep_dead_readers():
                return env.begin(write=False)
            # Every slot is a live snapshot somewhere; one closes soon, so
            # the wait is the caller's to retry (contract section 3).
            raise StorageLockTimeoutError(
                f"LMDB reader table is full: {self._max_readers} snapshots are open across "
                f"every process on this environment (max_readers, set by whoever created "
                f"its lock file). Close one or retry: {e}"
            ) from e
        except lmdb.MapResizedError:
            self._remap_to_file_size()
            return env.begin(write=False)

    def _begin_lmdb_write(self) -> Any:  # noqa: ANN401
        """Begin an LMDB write transaction. Runs on the writer thread, gate held.

        The begun transaction is registered as pending before the job
        returns, under ``_env_lock``, so close() either finds it there and
        ends it, or has already closed, in which case it is ended here.
        """
        env = self._env
        if env is None:
            raise StorageClosedError("Storage is not open")
        if time.monotonic() - self._last_reader_check >= READER_CHECK_INTERVAL:
            self._sweep_dead_readers()  # before a dead reader pins the map full
        try:
            lmdb_txn = env.begin(write=True)
        except lmdb.MapResizedError:
            self._remap_to_file_size()
            lmdb_txn = env.begin(write=True)
        with self._env_lock:
            if self._opened:
                self._live_writes.add(lmdb_txn)
                return lmdb_txn
        lmdb_txn.abort()
        raise StorageClosedError("Storage was closed while beginning the transaction")

    def _end_live(self, lmdb_txn: Any, commit: bool) -> None:  # noqa: ANN401
        """End a live write txn exactly once. Runs on the writer thread.

        A txn already ended (close() got there first, or a second abort
        request) is left alone; a commit of one raises.
        """
        with self._env_lock:
            if lmdb_txn not in self._live_writes:
                if commit:
                    raise StorageClosedError("Transaction was ended by close()")
                return
            # Taken off the live set before the end: nothing else ends a
            # live txn, and close() joins this thread before env.close().
            self._live_writes.discard(lmdb_txn)
        if not commit:
            lmdb_txn.abort()
            return
        # The commit (its fsync above all) is long for a good reason: a
        # soft stamp for readers while it runs, the progress stamp as it
        # was once it is done (the hand-off after it is a wait like any).
        writer_thread = self._writer_thread
        previous = writer_thread._wait_since
        writer_thread._wait_since = 0.0  # the owner keeps waiting; only the stamp rests
        soft = writer_thread._stamp_soft()
        try:
            lmdb_txn.commit()
        except lmdb.MapFullError as e:
            retry = self._after_map_full("commit")
            if retry is not None:
                raise retry from e
            raise
        finally:
            writer_thread._unstamp_soft(soft)
            if previous and writer_thread._waiting:  # not if the owner left meanwhile
                writer_thread._wait_since = previous

    def _remap_to_file_size(self) -> None:
        """Adopt the map size a peer grew the file to (``MDB_MAP_RESIZED``).

        LMDB remaps in place, which is only safe while this process has no
        transaction over the old mapping.
        """
        with self._snap_lock, self._env_lock:
            env = self._env
            if env is None:
                raise StorageClosedError("Storage is not open")
            # A pending transaction (not begun) holds nothing over the mapping.
            if (
                self._active_snapshots
                or any(t._lmdb_txn is not None for t in self._active_transactions)
                or self._reserved_slots
            ):
                raise StorageError(
                    "Another process grew the LMDB map past this process's mapping, and "
                    "remapping is only possible with no open snapshot or transaction here. "
                    "Close them and retry, or open every process with the same map_size"
                )
            try:
                env.set_mapsize(0)
            except Exception as e:
                raise StorageError(f"Failed to adopt the grown LMDB map: {e}") from e
            logger.info("LMDB map remapped to the size another process grew it to")

    def begin_transaction(self, *, write_slot: WriteSlot | None = None) -> LMDBTransaction:
        """Begin read-write transaction, taking the writer slot now.

        Args:
            write_slot: A slot from ``areserve_write_slot``. The transaction
                takes it over instead of acquiring the slot itself, then
                releases it on commit or abort. None acquires as usual.
        """
        if self._read_only:
            raise StorageError("Cannot start transaction in read only mode.")

        self._require_open()
        self._check_fork()
        self._require_not_forked()

        # From the take of the slot to the transaction that owns it (its
        # finalizer gives everything back from then on) the bookkeeping is
        # a few bytecodes; the waits inside (slot, gate, begin) undo
        # themselves when interrupted.
        deadline = time.monotonic() + self._busy_timeout
        lock, give_back_slot = self._writer.take_slot(self, write_slot, self._busy_timeout)
        lmdb_txn: Any = None
        try:
            if write_slot is None:
                self._acquire_gate(lock, deadline)
            with self._env_lock:
                if write_slot is not None:
                    lmdb_txn, write_slot._lmdb_txn = write_slot._lmdb_txn, None  # type: ignore[attr-defined]
                    self._reserved_slots.discard(write_slot)  # type: ignore[arg-type]
                    if lmdb_txn is None:
                        raise StorageClosedError(
                            "Write slot was reserved on a storage that has since closed"
                        )
                    if not self._opened:
                        raise StorageClosedError("Storage is not open")  # close() ends the txn
                else:
                    # A close() and open() since the gate take dropped the
                    # gate and replaced the slot: nothing to begin with.
                    self._require_slot_current(lock)
                transaction = LMDBTransaction(self, lmdb_txn, lock)  # pending unless reserved
                self._active_transactions.add(transaction)
                return transaction
        except BaseException as e:
            try:
                if lmdb_txn is not None:
                    try:
                        self._end_write(lmdb_txn, commit=False)
                    except Exception as abort_error:
                        logger.error(f"Abort after a failed begin failed: {abort_error}")
            finally:
                # A lock that is no longer the writer's: a close() dropped
                # the gate since, and the reopened storage's gate may be a
                # new holder's. Only the slot's own holder gives it back.
                if write_slot is None and lock is self._writer.lock:
                    self._release_gate()
                give_back_slot()
            if isinstance(e, lmdb.Error):
                raise StorageError(f"Failed to begin transaction: {e}") from e
            raise  # storage errors, and interrupts, as they are

    async def areserve_write_slot(self) -> WriteSlot:
        """Wait for the writer slot without blocking the event loop.

        Waits for the in-process slot, then for the cross-process gate,
        then for the writer thread to begin the LMDB transaction, each with
        a short, growing sleep between tries, so other coroutines on the
        loop keep running, including the one that holds the slot. Waits
        behind other threads and processes too. The whole wait is bounded
        by ``busy_timeout``.

        Returns:
            The held slot, for ``begin_transaction(write_slot=...)``. Call
            its ``release`` if no transaction takes it.

        Raises:
            StorageNestedTransactionError: If this task already holds the
                slot.
            StorageLockTimeoutError: If it is not free within
                ``busy_timeout``.
        """
        if self._read_only:
            raise StorageError("Cannot reserve the write slot in read only mode.")
        self._require_open()
        self._check_fork()
        self._require_not_forked()
        deadline = time.monotonic() + self._busy_timeout
        # A handler may raise as any ``await`` resumes, before its result
        # is bound, and a cancellation lands there too. So what this task
        # holds is kept where its exception path can find it: the slot by
        # its owner (``release_if_owned``), the begun transaction in
        # ``begun``, filled on the writer thread.
        me = self._writer.caller()
        since = self._writer.hold_token()
        lock: threading.Lock | None = None
        begun: list[Any] = []

        def begin() -> Any:  # noqa: ANN401
            with self._env_lock:
                # A close() and open() since the gate take: refused before
                # a transaction is begun on the reopened environment.
                self._require_slot_current(lock)
            lmdb_txn = self._begin_lmdb_write()
            begun.append(lmdb_txn)
            return lmdb_txn

        def abandoned(lmdb_txn: Any) -> None:  # noqa: ANN401
            # Whichever of this and the exception path below takes the txn
            # out of ``begun`` aborts it, once.
            try:
                begun.remove(lmdb_txn)
            except ValueError:
                return
            self._abort_abandoned(lmdb_txn)

        # A close() and open() can both land between any two steps below,
        # and the reopened storage has a fresh slot: this reservation then
        # holds nothing it can hand over. Each step checks under
        # ``_env_lock`` that the slot is still the storage's before it
        # takes anything more (the gate attach runs inside that check), so
        # a stale reservation is refused with what it has already given
        # back, never returned as a slot that carries the reopened
        # storage's gate or a begun transaction.
        try:
            lock = await self._writer.areserve(self._busy_timeout)
            if self._gate is not None:
                stamp = self._writer_thread._stamp_soft()
                try:
                    await self._gate.aacquire(
                        deadline - time.monotonic(), guard=self._slot_guard(lock)
                    )
                finally:
                    self._writer_thread._unstamp_soft(stamp)
                self._gate_held = True
            # With the gate held the begin is not a wait on a virtuals peer,
            # so it gets the full timeout (a writer outside virtuals bounds).
            lmdb_txn = await self._writer_thread.arun(
                begin, timeout=self._busy_timeout, abandoned=abandoned
            )
            with self._env_lock:
                # close() ends the txn; a reopen since replaced the slot.
                self._require_slot_current(lock)
                begun.remove(lmdb_txn)
                slot = LMDBWriteSlot(self, lock, lmdb_txn)
                self._reserved_slots.add(slot)
                return slot
        except BaseException:
            # A begin that landed as the await resumed is aborted on the
            # writer thread, never waited on here. ``abandoned`` may take
            # it out of ``begun`` from that thread first: the pop is the
            # one that settles who aborts it.
            while begun:
                try:
                    lmdb_txn = begun.pop()
                except IndexError:
                    break
                self._writer_thread.try_fire(functools.partial(self._abort_abandoned, lmdb_txn))
            if lock is not None:
                # A stale lock: the gate was never this reservation's to keep.
                self._release_write_slot(lock)
            else:
                self._release_gate()
                self._writer.release_if_owned(me, since)
            raise

    def begin_write_batch(self) -> LMDBWriteBatch:
        """Begin write-only batch. Takes no lock until ``write``."""
        if self._read_only:
            raise StorageError("Cannot start write batch in read only mode.")

        self._require_open()
        self._check_fork()
        self._require_not_forked()

        write_batch = LMDBWriteBatch(self)
        with self._env_lock:
            self._require_open()
            self._active_write_batches.add(write_batch)
        return write_batch

    def _require_slot_current(self, lock: threading.Lock | None) -> None:
        """Under ``_env_lock``: the storage is open and ``lock`` is still its writer slot.

        Raises ``StorageClosedError`` otherwise. A lock that is not the
        writer's means a close() (and an open()) landed since it was
        taken: its holder holds nothing on the reopened storage.
        """
        if not self._opened:
            raise StorageClosedError("Storage is not open")
        if lock is not self._writer.lock:
            raise StorageClosedError(
                "Storage was closed and reopened while the write slot was being taken"
            )

    @contextmanager
    def _slot_guard(self, lock: threading.Lock | None) -> Iterator[None]:
        """The gate is attached inside this: the slot is the storage's as it is taken.

        Without it a close() and open() landing between the slot grant
        and the gate take would hand the reopened storage's gate to a
        holder of a replaced slot, which nothing would ever release.
        """
        with self._env_lock:
            self._require_slot_current(lock)
            yield

    def _acquire_gate(self, lock: threading.Lock, deadline: float) -> None:
        """Take the cross-process gate, waiting until ``deadline`` at most.

        ``lock`` is the writer slot the caller holds: the gate is taken
        only while it is still the storage's (``_slot_guard``). On an
        event loop thread the wait is capped at ``LOOP_THREAD_WAIT_CAP``
        like the in-process slot's (contract section 4: a wait on the
        loop is bounded and surfaced); the longer wait belongs in
        ``areserve_write_slot``.
        """
        if self._gate is None:
            return
        timeout = deadline - time.monotonic()
        on_loop = asyncio._get_running_loop() is not None
        if on_loop:
            timeout = min(timeout, LOOP_THREAD_WAIT_CAP)
        stamp = self._writer_thread._stamp_soft()  # flock releases the GIL: see yield_to_writer
        try:
            self._gate.acquire(timeout, guard=self._slot_guard(lock))
            self._gate_held = True
        except StorageLockTimeoutError:
            if not on_loop:
                raise
            raise self._writer.loop_cap_timeout("the write lock held by another process") from None
        except StorageClosedError:
            raise  # refused before the take, or closed while waiting: nothing is held
        except BaseException:
            # An interrupt between the acquire and the flag would leave the
            # gate held and unflagged, so never released: give it back
            # whatever the state. The slot is held, so nobody else in this
            # process can own the gate, and an unlock of a lock not held
            # is a no-op.
            self._gate_held = False
            self._gate.release()
            raise
        finally:
            self._writer_thread._unstamp_soft(stamp)

    def _hop(self, run: Callable[[float], Any]) -> Any:  # noqa: ANN401
        """Run a begin hop with its bound: ``busy_timeout``, capped on a loop thread.

        With the gate held the begin is not a wait on a virtuals peer, so
        it gets the full timeout (a writer outside virtuals bounds). On an
        event loop thread the wait is capped like the slot's and the
        gate's, with the same error.
        """
        on_loop = asyncio._get_running_loop() is not None
        timeout = min(self._busy_timeout, LOOP_THREAD_WAIT_CAP) if on_loop else self._busy_timeout
        try:
            return run(timeout)
        except StorageLockTimeoutError:
            if on_loop:
                raise self._writer.loop_cap_timeout("LMDB's writer lock") from None
            raise

    def _begin_on_writer(self) -> Any:  # noqa: ANN401
        """Begin an LMDB write txn on the writer thread, bounded; the caller holds the gate."""
        return self._hop(
            lambda timeout: self._writer_thread.run(
                self._begin_lmdb_write, timeout=timeout, abandoned=self._abort_abandoned
            )
        )

    def _take_write_slot(self) -> threading.Lock:
        """Take the in-process slot, then the gate, bounded together by ``busy_timeout``.

        For a write that holds both only while it is applied (a batch's
        ``write``). Give both back with ``_release_write_slot(lock)``.
        """
        self._require_not_forked()
        deadline = time.monotonic() + self._busy_timeout
        lock = self._writer.acquire(self._busy_timeout)
        try:
            self._acquire_gate(lock, deadline)
        except BaseException:
            self._writer.release(lock)
            raise
        return lock

    def _begin_pending(self, pending: Pending) -> tuple[Any, set[bytes]]:
        """Begin a pending transaction's LMDB txn and replay its writes on this thread.

        Returns the handle and the keys whose delete found nothing. A
        failed replay aborts the handle again and raises, leaving the
        transaction pending.
        """
        swept = False
        while True:
            lmdb_txn = self._begin_on_writer()
            if not pending:
                return lmdb_txn, ()  # a read-first transaction: nothing to replay
            try:
                return lmdb_txn, replay_pending(lmdb_txn, pending)
            except BaseException as e:  # a signal mid-replay too: the begun txn is nobody's else
                try:
                    self._end_write(lmdb_txn, commit=False)
                except Exception as abort_error:
                    logger.error(f"Abort after a failed replay failed: {abort_error}")
                if not isinstance(e, lmdb.Error):
                    raise  # an interrupt, as it is
                if not swept and isinstance(e, lmdb.MapFullError) and self._sweep_dead_readers():
                    swept = True  # a dead reader pinned the pages: once more with them back
                    continue
                raise StorageOperationError(f"Failed to apply pending writes: {e}") from e

    def _commit_pending(self, pending: Pending, deferred: list[BaseException]) -> Any:  # noqa: ANN401
        """Begin, replay and commit on the writer thread in one job.

        The caller holds the slot and the gate. The begin is bounded like
        any other; once it has landed the commit runs to the end. Returns
        the keys whose delete found nothing, or ``NOT_RUN`` if an
        interrupt arrived before the begin landed (nothing is committed;
        the interrupt is in ``deferred``). An interrupt after that is
        deferred and the commit is waited out: its outcome is what the
        caller reports, then the interrupt (see ``WriterThread.run``).
        """
        return self._hop(
            lambda timeout: self._writer_thread.run_staged(
                self._begin_lmdb_write,
                lambda lmdb_txn: self._apply_and_commit(lmdb_txn, pending),
                timeout=timeout,
                abandoned=self._abort_abandoned,
                deferred=deferred,
            )
        )

    def _apply_and_commit(self, lmdb_txn: Any, pending: Pending) -> set[bytes]:  # noqa: ANN401
        """Second stage of a write-only commit, on the writer thread.

        A replay that fills the map behind a dead reader's slot is retried
        once after the sweep, on a fresh begin (the requester still holds
        the gate); a commit that does so raises the retryable error.
        """
        swept = False
        progress = self._writer_thread._progress
        while True:
            try:
                missing = replay_pending(lmdb_txn, pending, progress)
            except BaseException as e:
                self._end_live(lmdb_txn, False)
                if not swept and isinstance(e, lmdb.MapFullError) and self._sweep_dead_readers():
                    swept = True
                    lmdb_txn = self._begin_lmdb_write()
                    continue
                raise
            self._end_live(lmdb_txn, True)
            return missing

    def _abort_abandoned(self, lmdb_txn: Any) -> None:  # noqa: ANN401
        """End a begin whose requester gave up (runs on the writer thread).

        The requester has already given the slot and gate back by then, so
        a peer may block briefly inside LMDB until this runs; bounded by
        the one abort.
        """
        self._end_live(lmdb_txn, commit=False)

    def _end_write(
        self,
        lmdb_txn: Any,  # noqa: ANN401
        *,
        commit: bool,
        deferred: list[BaseException] | None = None,
    ) -> None:
        """Commit or abort an LMDB write transaction on the writer thread.

        Called under the context's call lock from any thread. The gate and
        the slot are given back by the context's ``_mark_closed`` after.
        The end runs to its end whatever the caller does. With
        ``deferred`` (a commit) an interrupt while waiting is deferred and
        the commit waited out, so the caller reports the outcome first
        and the interrupt after (see ``WriterThread.run``); an abort
        needs no such care, its requester's cleanup runs either way.
        """
        self._writer_thread.run(lambda: self._end_live(lmdb_txn, commit), deferred=deferred)

    def _release_gate(self) -> None:
        if self._gate_held:
            self._gate_held = False
            if self._gate is not None:
                stamp = self._writer_thread._stamp_soft()  # flock releases the GIL
                try:
                    self._gate.release()
                finally:
                    self._writer_thread._unstamp_soft(stamp)

    def _release_write_slot(self, lock: Any = None) -> None:  # noqa: ANN401
        """Release the cross-process gate, then the in-process writer slot.

        The slot is released in a ``finally``: a slot nobody holds and
        nobody releases wedges every writer in this process until close().
        """
        if lock is not None and lock is not self._writer.lock:
            return  # a fork reset the slot since; the gate was reopened too
        try:
            self._release_gate()
        finally:
            self._writer.release(lock)

    def _release_reserved(self, slot: LMDBWriteSlot, *, finalizing: bool = False) -> None:
        """Give back a reserved slot no transaction took: abort its begun txn first.

        From the slot's finalizer the abort is queued, not waited for, and
        skipped when the writer thread is gone or the interpreter is
        exiting; the slot and the gate are given back at once.
        """
        with self._env_lock:
            lmdb_txn, slot._lmdb_txn = slot._lmdb_txn, None
            self._reserved_slots.discard(slot)
        try:
            if lmdb_txn is not None:
                try:
                    if finalizing:
                        self._end_write_from_finalizer(lmdb_txn)
                    else:
                        self._end_write(lmdb_txn, commit=False)
                except StorageClosedError:
                    pass  # close() got there first and ended it
                except Exception as e:
                    logger.error(f"Abort of a reserved transaction failed: {e}")
        finally:
            self._release_write_slot(slot._lock)

    def _end_write_from_finalizer(self, lmdb_txn: Any) -> bool:  # noqa: ANN401
        """Queue the abort of a dropped write txn on the writer thread; False if skipped.

        Never waits: the collecting thread may hold ``_env_lock``, which
        the abort needs. Skipped means the writer thread is gone (close()
        stopped it after ending every live write itself) or the
        interpreter is exiting.
        """
        return self._writer_thread.try_fire(lambda: self._end_live(lmdb_txn, False))

    @contextmanager
    def transaction(self) -> Iterator[LMDBTransaction]:
        """Context manager for a read-write transaction.

        Aborts on any exception out of the block, including cancellation
        and KeyboardInterrupt, so the writer slot is never stranded.
        """
        txn = self.begin_transaction()
        try:
            yield txn
            if not txn._committed and not txn._aborted:
                txn.commit()
        except BaseException:
            try:
                if not txn._committed and not txn._aborted:
                    txn.abort()
            except Exception as e:
                logger.error(f"Transaction abort failed: {e}")
            raise

    @contextmanager
    def snapshot(self) -> Iterator[LMDBSnapshot]:
        """Context manager for a read-only snapshot."""
        snap = self.begin_snapshot()
        try:
            yield snap
        finally:
            try:
                snap.close()
            except Exception as e:
                logger.error(f"Snapshot close failed: {e}")

    @contextmanager
    def batch_write(self) -> Iterator[LMDBWriteBatch]:
        """Context manager for a write batch.

        Aborts on any exception out of the block, including cancellation
        and KeyboardInterrupt, so the writer slot is never stranded.
        """
        batch = self.begin_write_batch()
        try:
            yield batch
            if not batch._written and not batch._aborted:
                batch.write()
        except BaseException:
            try:
                if not batch._written and not batch._aborted:
                    batch.abort()
            except Exception as e:
                logger.error(f"Write batch abort failed: {e}")
            raise

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _notify_batch(self, keys: set[Key]) -> None:
        """Notify publisher of key changes (batch).

        Fire-and-forget: writer enqueues and returns. Callers that need a
        delivery barrier call publisher.flush() explicitly.
        """
        if self._publisher is not None and keys:
            try:
                self._publisher.notify(keys)
            except Exception as e:
                logger.error(f"Publisher notification failed: {e}")

    def _untrack_transaction(self, transaction: LMDBTransaction) -> None:
        """Remove transaction from active set."""
        with self._env_lock:
            self._active_transactions.discard(transaction)

    def _untrack_snapshot(self, snapshot: LMDBSnapshot) -> None:
        """Remove snapshot from active set."""
        with self._snap_lock:
            self._active_snapshots.discard(snapshot)

    def _untrack_write_batch(self, write_batch: LMDBWriteBatch) -> None:
        """Remove write batch from active set."""
        with self._env_lock:
            self._active_write_batches.discard(write_batch)

    def _require_open(self) -> None:
        """Validate storage is open."""
        if not self._opened or self._env is None:
            raise StorageClosedError("Storage is not open")


if TYPE_CHECKING:
    _: type[TransactionProtocol] = LMDBTransaction
    __: type[SnapshotProtocol] = LMDBSnapshot
