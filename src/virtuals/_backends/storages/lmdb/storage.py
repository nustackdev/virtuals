"""LMDB storage backend implementation.

Provides persistent key-value storage over LMDB with transactions,
MVCC snapshots, and optional change notifications via a publisher.
"""

from __future__ import annotations

import os
import threading
import time
import weakref
from contextlib import contextmanager
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, overload

from virtuals.tkv.storage import (
    SnapshotProtocol,
    StorageClosedError,
    StorageError,
    StorageLockTimeoutError,
    StorageOperationError,
    TransactionProtocol,
    WriteBatchProtocol,
)

from .._write_slot import WriterLock, WriteSlot
from ._writer import WriterGate, WriterThread
from .snapshot import LMDBSnapshot
from .transaction import LMDBTransaction
from .write_batch import LMDBWriteBatch


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from types import TracebackType

    from virtuals.tkv.codec import CodecProtocol
    from virtuals.tkv.publisher import PublisherProtocol
    from virtuals.tkv.types import Key


try:
    import lmdb
except ImportError as e:
    raise ImportError(
        "LMDB python bindings are required for LMDBStorage. Install via: [uv] pip install lmdb"
    ) from e


__all__ = ["LMDBStorage", "LMDBWriteSlot"]


logger = getLogger(__name__)


DEFAULT_MAP_SIZE = 10 * 1024 * 1024 * 1024  # 10 GiB
DEFAULT_MAX_READERS = 126
DEFAULT_BUSY_TIMEOUT = 60.0  # seconds

# The hop to the writer thread for a begin is not a wait on a lock once the
# gate is held, so it gets at least this much even when the gate wait used
# the whole ``busy_timeout``. Only a writer outside virtuals makes it matter.
_BEGIN_HOP_FLOOR = 0.01


class LMDBWriteSlot(WriteSlot):
    """The LMDB writer slot, held with the cross-process gate and a begun LMDB write txn.

    ``areserve_write_slot`` does all the waiting (in-process slot, then
    the gate, then the begin on the writer thread) without blocking the
    event loop, so ``begin_transaction(write_slot=...)`` only wraps the
    transaction already begun. Released untaken, the slot aborts it.
    """

    __slots__ = ("__weakref__", "_lmdb_txn")

    def __init__(self, storage: LMDBStorage, lock: threading.Lock, lmdb_txn: Any) -> None:  # noqa: ANN401
        """Initialize a held slot carrying a begun LMDB write transaction."""
        super().__init__(storage, lock)
        self._lmdb_txn: Any | None = lmdb_txn

    def release(self) -> None:
        """Give the slot back, aborting its transaction, if no transaction took it."""
        if self._state != "held":
            return
        self._state = "released"
        self._storage._release_reserved(self)  # type: ignore[attr-defined]


# Storages with locks to replace in a forked child (see ``_after_fork_in_child``).
_storages: weakref.WeakSet[LMDBStorage] = weakref.WeakSet()


def _after_fork_in_child() -> None:
    """Fresh locks in the child: a parent thread may hold the inherited ones forever."""
    for storage in list(_storages):
        storage._reset_locks_after_fork()


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

    Writers in this process (transactions and write batches) first take an
    in-process writer slot, the same one SQLite uses, so threads queue on
    it for up to ``busy_timeout`` and then raise ``StorageLockTimeoutError``
    instead of LMDB refusing a second writer. Then they take a
    cross-process gate (an ``flock`` on ``writer.lock`` next to the
    environment) with the same bound, so a writer in another process
    surfaces as ``StorageLockTimeoutError`` too instead of an unbounded
    block inside LMDB. The LMDB begin, commit and abort themselves run on
    one writer thread per storage, the only thread that ever touches
    LMDB's writer lock, so a transaction may be begun on one thread and
    committed or aborted on any other. Snapshots never touch the slot or
    the gate, so a waiting writer never delays a reader.

    Coroutines: the slot belongs to a thread and, on an event loop, to the
    task that took it. Coroutines sharing one loop wait for it with
    ``areserve_write_slot``, which also waits for the gate and begins the
    LMDB transaction, all without blocking the loop, and hand the slot to
    ``begin_transaction(write_slot=...)``. A second writer from the task or
    thread that already holds the slot raises
    ``StorageNestedTransactionError``; a sync begin while another task on
    the same loop holds it raises ``StorageLockTimeoutError`` at once.

    close() ends open contexts from the closing thread. One whose owner is
    inside a call is ended after that call returns, never under it, and
    the owner's next call raises ``StorageClosedError``.

    Multiple processes: open the storage in each process (e.g. with the
    ``spawn`` start method); that is the supported path. Reusing an open
    storage across ``os.fork`` is best effort. If no write was in progress
    at the fork, the child keeps using it from any thread, but its readers
    stay registered under the parent's pid, so a reader check elsewhere
    can misjudge them. If the parent held a write transaction or the
    writer slot at the fork, LMDB's writer state in the child is unusable
    and py-lmdb cannot reopen the same files there, so every write in the
    child raises a non-retryable ``StorageError`` saying so.

    ``map_size`` is a ceiling every process should agree on. If a peer
    opened the environment with a larger map and grew the file past this
    process's mapping, the next begin here remaps to the file's size, which
    LMDB only allows while this process has no open snapshot or
    transaction; otherwise it raises ``StorageError`` asking to close them.
    """

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
        self._opened = False
        self._pid = os.getpid()
        self._forked_during_write = False
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

            try:
                self._env = lmdb.open(**open_kwargs)
            except Exception as e:
                raise StorageError(f"Failed to open LMDB environment: {e}") from e

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

            self._pid = os.getpid()
            self._forked_during_write = False
            self._opened = True

    def _sweep_dead_readers(self) -> int:
        """Clear reader slots left by dead processes; returns how many."""
        env = self._env
        if env is None:
            return 0
        try:
            return env.reader_check()
        except Exception as e:  # read-only lock file, or an env without locking
            logger.debug(f"LMDB reader check skipped: {e}")
            return 0

    def _reset_locks_after_fork(self) -> None:
        """Replace locks a parent thread may hold at the fork (``register_at_fork``)."""
        self._env_lock = threading.RLock()
        self._writer_thread.forget()

    def _check_fork(self) -> None:
        """Drop writer state inherited across a fork.

        A writer slot held by a parent thread would never be released in
        the child. If this process is not the one that opened the storage,
        the slot is reset and the parent's open contexts are forgotten, not
        ended: aborting them here would write to the lock file the parent
        still uses (its reader slots and writer mutex). py-lmdb frees
        nothing for objects inherited across a fork, and it refuses a
        second open of the same files in one process, so the child keeps
        the inherited environment. The writer thread does not exist in the
        child and is started afresh; the gate descriptor is reopened, since
        the inherited one shares the parent's lock.

        If a write was in progress at the fork, LMDB's inherited writer
        state rejects or blocks every write in the child, so writes fail
        with a clear error instead (see ``_require_writable_after_fork``).
        """
        if self._pid == os.getpid():
            return
        with self._env_lock:
            if self._pid == os.getpid():
                return
            self._forked_during_write = bool(
                self._writer.lock.locked()
                or self._active_transactions
                or self._active_write_batches
                or self._reserved_slots
            )
            self._active_transactions = weakref.WeakSet()
            self._active_write_batches = weakref.WeakSet()
            self._active_snapshots = weakref.WeakSet()
            self._reserved_slots = weakref.WeakSet()
            self._writer.reset()
            self._writer_thread.forget()
            self._gate_held = False
            if self._gate is not None and self._opened:
                try:
                    self._gate.reopen_after_fork()
                except OSError as e:
                    logger.error(f"Failed to reopen the LMDB writer gate after fork: {e}")
            self._pid = os.getpid()

    def _require_writable_after_fork(self) -> None:
        """Refuse writes on a storage inherited mid-write across a fork."""
        if self._forked_during_write:
            raise StorageError(
                "This LMDB storage was inherited across a fork while the parent held a "
                "write transaction, so it cannot write in this process (LMDB's inherited "
                "writer state is unusable here, and py-lmdb cannot reopen the same files "
                "in it). Open the storage in a spawned process instead, or fork only "
                "while no write is in progress"
            )

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

        with self._env_lock:
            if not self._opened:
                return
            self._opened = False
            transactions = list(self._active_transactions)
            write_batches = list(self._active_write_batches)
            snapshots = list(self._active_snapshots)
            reserved = [(slot, slot._lmdb_txn) for slot in self._reserved_slots]
            for slot, _ in reserved:
                slot._lmdb_txn = None
            self._active_transactions.clear()
            self._active_write_batches.clear()
            self._active_snapshots.clear()
            self._reserved_slots.clear()

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

        for _, lmdb_txn in reserved:
            if lmdb_txn is not None:
                try:
                    self._writer_thread.run(lmdb_txn.abort)
                except Exception as e:
                    logger.error(f"Reserved transaction abort failed during close: {e}")

        for snapshot in snapshots:
            try:
                snapshot.close()
            except Exception as e:
                logger.error(f"Snapshot close failed during close: {e}")

        self._writer.close()  # a reserved slot nobody took does not survive the reopen
        stopped = self._writer_thread.stop(self._busy_timeout)

        with self._env_lock:
            if self._gate is not None:
                self._gate.close()  # drops the gate if a reserved slot still held it
            self._gate_held = False
            env, self._env = self._env, None
            if env is None:
                return
            if self._forked_during_write:
                # Closing an environment inherited mid-write can hang in
                # LMDB; drop it instead (py-lmdb frees nothing inherited).
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

        with self._env_lock:
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
        except lmdb.ReadersFullError:
            if self._sweep_dead_readers() == 0:
                raise
            return env.begin(write=False)
        except lmdb.MapResizedError:
            self._remap_to_file_size()
            return env.begin(write=False)

    def _begin_lmdb_write(self) -> Any:  # noqa: ANN401
        """Begin an LMDB write transaction. Runs on the writer thread, gate held."""
        env = self._env
        if env is None:
            raise StorageClosedError("Storage is not open")
        try:
            return env.begin(write=True)
        except lmdb.MapResizedError:
            self._remap_to_file_size()
            return env.begin(write=True)

    def _remap_to_file_size(self) -> None:
        """Adopt the map size a peer grew the file to (``MDB_MAP_RESIZED``).

        LMDB remaps in place, which is only safe while this process has no
        transaction over the old mapping.
        """
        with self._env_lock:
            env = self._env
            if env is None:
                raise StorageClosedError("Storage is not open")
            if (
                self._active_snapshots
                or self._active_transactions
                or self._active_write_batches
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
        self._require_writable_after_fork()

        return self._begin_write(write_slot, StorageError, "transaction", LMDBTransaction)

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
        self._require_writable_after_fork()
        deadline = time.monotonic() + self._busy_timeout
        lock = await self._writer.areserve(self._busy_timeout)
        try:
            if self._gate is not None:
                await self._gate.aacquire(deadline - time.monotonic(), self._is_open)
                self._gate_held = True
            lmdb_txn = await self._writer_thread.arun(
                self._begin_lmdb_write,
                timeout=max(deadline - time.monotonic(), _BEGIN_HOP_FLOOR),
                undo=self._abort_abandoned,
            )
            with self._env_lock:
                if self._opened:
                    slot = LMDBWriteSlot(self, lock, lmdb_txn)
                    self._reserved_slots.add(slot)
                    return slot
            self._writer_thread.run(lmdb_txn.abort)  # closed meanwhile; never hop under the lock
            raise StorageClosedError("Storage is not open")
        except BaseException:
            self._release_write_slot(lock)
            raise

    def begin_write_batch(self) -> LMDBWriteBatch:
        """Begin write-only batch, taking the writer slot now."""
        if self._read_only:
            raise StorageError("Cannot start write batch in read only mode.")

        self._require_open()
        self._check_fork()
        self._require_writable_after_fork()

        return self._begin_write(None, StorageOperationError, "write batch", LMDBWriteBatch)

    def _begin_write(
        self,
        write_slot: WriteSlot | None,
        error: type[StorageError],
        what: str,
        make: Callable[[LMDBStorage, Any, threading.Lock], Any],
    ) -> Any:  # noqa: ANN401
        """Take the slot and the gate, begin on the writer thread, register the context.

        Nothing here runs under ``_env_lock`` while it can wait, so
        snapshots and close() never queue behind a writer. ``busy_timeout``
        bounds the slot wait and the gate wait together. On failure the
        slot is released, or returned to ``write_slot``'s holder.
        """
        deadline = time.monotonic() + self._busy_timeout
        lock, give_back_slot = self._writer.take_slot(self, write_slot, self._busy_timeout)
        lmdb_txn: Any = None
        try:
            if write_slot is None:
                if self._gate is not None:
                    self._gate.acquire(deadline - time.monotonic(), self._is_open)
                    self._gate_held = True
                lmdb_txn = self._writer_thread.run(
                    self._begin_lmdb_write,
                    timeout=max(deadline - time.monotonic(), _BEGIN_HOP_FLOOR),
                    undo=self._abort_abandoned,
                )
            with self._env_lock:
                if write_slot is not None:
                    lmdb_txn, write_slot._lmdb_txn = write_slot._lmdb_txn, None  # type: ignore[attr-defined]
                    self._reserved_slots.discard(write_slot)  # type: ignore[arg-type]
                    if lmdb_txn is None:
                        raise StorageClosedError(
                            "Write slot was reserved on a storage that has since closed"
                        )
                if not self._opened:
                    raise StorageClosedError("Storage is not open")
                context = make(self, lmdb_txn, lock)
                if isinstance(context, LMDBTransaction):
                    self._active_transactions.add(context)
                else:
                    self._active_write_batches.add(context)
                return context
        except BaseException as e:
            if lmdb_txn is not None:
                try:
                    self._writer_thread.run(lmdb_txn.abort)
                except Exception as abort_error:
                    logger.error(f"Abort after a failed begin failed: {abort_error}")
            if write_slot is None:
                self._release_gate()
            give_back_slot()
            if isinstance(e, (StorageClosedError, StorageLockTimeoutError)):
                raise
            if isinstance(e, Exception):
                raise error(f"Failed to begin {what}: {e}") from e
            raise

    def _is_open(self) -> bool:
        return self._opened

    def _abort_abandoned(self, lmdb_txn: Any) -> None:  # noqa: ANN401
        """End a begin whose requester gave up (runs on the writer thread)."""
        lmdb_txn.abort()

    def _end_write(self, lmdb_txn: Any, *, commit: bool) -> None:  # noqa: ANN401
        """Commit or abort an LMDB write transaction on the writer thread.

        Called under the context's call lock from any thread. The gate and
        the slot are given back by the context's ``_mark_closed`` after.
        """
        self._writer_thread.run(lmdb_txn.commit if commit else lmdb_txn.abort)

    def _release_gate(self) -> None:
        if self._gate_held:
            self._gate_held = False
            if self._gate is not None:
                self._gate.release()

    def _release_write_slot(self, lock: Any = None) -> None:  # noqa: ANN401
        """Release the cross-process gate, then the in-process writer slot."""
        if lock is not None and lock is not self._writer.lock:
            return  # a fork reset the slot since; the gate was reopened too
        self._release_gate()
        self._writer.release(lock)

    def _release_reserved(self, slot: LMDBWriteSlot) -> None:
        """Give back a reserved slot no transaction took: abort its begun txn first."""
        with self._env_lock:
            lmdb_txn, slot._lmdb_txn = slot._lmdb_txn, None
            self._reserved_slots.discard(slot)
        if lmdb_txn is not None:
            try:
                self._writer_thread.run(lmdb_txn.abort)
            except Exception as e:
                logger.error(f"Abort of a reserved transaction failed: {e}")
        self._release_write_slot(slot._lock)

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
        with self._env_lock:
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
