"""SQLite storage backend implementation.

Provides persistent key-value storage over one SQLite file in WAL mode,
with serialized write transactions, non-blocking snapshot reads, and
optional change notifications via a publisher. Needs nothing beyond the
standard library.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import threading
import time
import weakref
from contextlib import contextmanager
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Literal, overload

from virtuals.tkv.storage import (
    SnapshotProtocol,
    StorageClosedError,
    StorageError,
    StorageLockTimeoutError,
    TransactionProtocol,
    WriteBatchProtocol,
)

from .._write_slot import WriterLock
from .context import is_busy_error
from .snapshot import SQLiteSnapshot
from .transaction import SQLiteTransaction
from .write_batch import SQLiteWriteBatch
from .write_slot import SQLiteWriteSlot


if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import TracebackType

    from virtuals.tkv.codec import CodecProtocol
    from virtuals.tkv.publisher import PublisherProtocol
    from virtuals.tkv.types import Key


__all__ = ["SQLiteStorage"]


logger = getLogger(__name__)


DEFAULT_BUSY_TIMEOUT = 60.0  # seconds
DEFAULT_MAX_IDLE_CONNECTIONS = 8

Synchronous = Literal["OFF", "NORMAL", "FULL", "EXTRA"]

_SCHEMA = "CREATE TABLE IF NOT EXISTS kv (k BLOB PRIMARY KEY, v BLOB) WITHOUT ROWID"

# Poll interval bounds for a coroutine waiting on the cross-process write
# lock (seconds), same shape as the in-process slot wait.
_LOCK_POLL_MIN = 0.0005
_LOCK_POLL_MAX = 0.005

# Connections a forked child inherited. SQLite forbids using a connection in
# a child, and closing one counts: it would roll back the parent's open
# transaction in the shared WAL index and tear down lock state the parent
# still relies on. The child keeps them referenced, untouched, until it exits.
_INHERITED_CONNECTIONS: list[sqlite3.Connection] = []

# Database files whose write lock the parent held when this process was
# forked off it. The inherited lock state marks the lock as held by this
# process for as long as the inherited connections live (forever, see above),
# so every new connection here would wait out ``busy_timeout`` and fail. A
# write on such a file is refused with a clear error instead.
_FORK_WRITE_POISONED: set[str] = set()

# Storages open in this process, so a fork can be noticed as it happens.
_OPEN_STORAGES: weakref.WeakSet[SQLiteStorage] = weakref.WeakSet()


def _after_fork_in_child() -> None:
    """Forget the parent's state on every open storage, before the child uses any."""
    for storage in list(_OPEN_STORAGES):
        try:
            storage._check_fork()
        except Exception as e:  # a fork hook must never raise
            logger.error(f"Fork handling failed for {storage.path}: {e}")


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)


class SQLiteStorage:
    """SQLite storage implementation conforming to StorageProtocol.

    One table, ``kv (k BLOB PRIMARY KEY, v BLOB) WITHOUT ROWID``: a B-tree
    clustered on the encoded key, compared as raw bytes, so range scans see
    keys in the same order as LMDB and RocksDB.

    SQLite semantics worth knowing:
      - WAL mode (the default, stored in the file): readers never block the
        writer and the writer never blocks readers. A snapshot is a read
        transaction pinned when it opens.
      - One writer per database file across every process. A transaction
        takes the write lock at begin (``BEGIN IMMEDIATE``); other writers
        wait up to ``busy_timeout`` and then raise
        ``StorageLockTimeoutError``, which ``RetryOnConflict`` retries.
      - Locks are OS file locks: a process killed mid-transaction releases
        them, and its uncommitted writes are simply gone.
      - ``synchronous="NORMAL"`` in WAL never corrupts the database on
        power loss and skips the fsync per commit; the last commits before
        a power cut may be lost. ``"FULL"`` fsyncs every commit.

    Threading: every transaction, snapshot and batch checks out its own
    connection from a small pool and returns it on close. Contexts in
    different threads therefore never share a connection or a lock, a
    snapshot never waits behind a writer, and a context may be begun on
    one thread and finished on another. Writers in this process also take
    an in-process writer slot before ``BEGIN IMMEDIATE`` so they hand off
    to each other directly rather than through SQLite's sleeping busy
    handler. A sync writer blocks on the slot for up to ``busy_timeout``.

    Coroutines: the slot belongs to a thread and, on an event loop, to the
    task that took it. Coroutines sharing one loop wait for it with
    ``areserve_write_slot``, which polls without blocking the loop, and
    hand the slot to ``begin_transaction(write_slot=...)``. A sync begin
    on a thread whose own task already holds the slot raises
    ``StorageNestedTransactionError``; one whose loop has another task
    holding it raises ``StorageLockTimeoutError``. Both fail at once
    instead of waiting: the holder could never run to release it.
    ``areserve_write_slot`` also waits for the cross-process file lock
    without blocking the loop, so a transaction begun with the slot never
    stalls the loop while another process writes. A plain sync begin
    waits in SQLite's busy handler, which is what a sync caller asked for.

    Multiple processes: open the storage in each process (e.g. with the
    ``spawn`` start method); that is the supported path. Reusing an open
    storage across ``os.fork`` is best effort. Connections inherited from
    the parent are never used in the child, not even to close them (SQLite
    forbids it: it would undo the parent's open transaction in the shared
    WAL index). If no write was in progress at the fork, the child keeps
    using the storage with fresh connections. If the parent held a write
    transaction or the writer slot at the fork, the inherited lock state
    marks the file's write lock as taken by the child for good, so every
    write in the child, on this storage or any other opened on the same
    file, raises a non-retryable ``StorageError`` saying so.
    """

    def __init__(
        self,
        path: Path | str,
        codec: CodecProtocol[bytes, bytes],
        publisher: PublisherProtocol | None = None,
        *,
        read_only: bool = False,
        create_if_missing: bool = True,
        synchronous: Synchronous = "NORMAL",
        busy_timeout: float = DEFAULT_BUSY_TIMEOUT,
        mmap_size: int | None = None,
        cache_size: int | None = None,
        wal: bool = True,
        pragmas: dict[str, object] | None = None,
        max_idle_connections: int = DEFAULT_MAX_IDLE_CONNECTIONS,
    ) -> None:
        """Initialize SQLite storage.

        Args:
            path: The database file. Its ``-wal`` and ``-shm`` companions
                live next to it.
            codec: Codec for key/value encoding. Keys should encode to
                bytes so they sort as BLOBs.
            publisher: Optional publisher for change notifications.
            read_only: Open the file with ``mode=ro``. No transactions or
                batches; snapshots only. See ``open`` for what a WAL reader
                still needs from the filesystem.
            create_if_missing: Create the file (and parent directory) if
                absent. Ignored when ``read_only``.
            synchronous: ``PRAGMA synchronous``. ``"NORMAL"`` is crash- and
                power-loss-safe in WAL without an fsync per commit;
                ``"FULL"`` adds one for durability of every commit.
            busy_timeout: Seconds a writer waits for the write lock (or a
                reader for a recovering WAL) before failing with
                ``StorageLockTimeoutError``.
            mmap_size: ``PRAGMA mmap_size`` in bytes, to read through a
                memory map. None leaves SQLite's default (off).
            cache_size: ``PRAGMA cache_size`` (pages if positive, KiB if
                negative). None leaves SQLite's default.
            wal: Put the file in WAL mode when creating/opening read-write.
                False leaves whatever journal mode the file has.
            pragmas: Extra ``PRAGMA name = value`` run on every connection,
                e.g. ``{"wal_autocheckpoint": 10000}``.
            max_idle_connections: Idle pooled connections kept open for
                reuse; extras are closed when returned.
        """
        self._codec = codec
        self._publisher = publisher

        self._read_only = read_only
        self._path = Path(path) if isinstance(path, str) else path
        self._create_if_missing = create_if_missing
        self._synchronous = synchronous
        self._busy_timeout = busy_timeout
        self._mmap_size = mmap_size
        self._cache_size = cache_size
        self._wal = wal
        self._pragmas = dict(pragmas or {})
        self._max_idle_connections = max_idle_connections

        self._state_lock = threading.Lock()
        self._idle: list[sqlite3.Connection] = []
        self._writer = WriterLock("SQLite")

        # Weak: a context dropped without being ended must be collectable,
        # so its finalizer can give back its connection and the writer slot.
        self._active_transactions: weakref.WeakSet[SQLiteTransaction] = weakref.WeakSet()
        self._active_write_batches: weakref.WeakSet[SQLiteWriteBatch] = weakref.WeakSet()
        self._active_snapshots: weakref.WeakSet[SQLiteSnapshot] = weakref.WeakSet()
        # Slots reserved and not yet taken by a transaction. Each holds a
        # begun connection that close() must take back.
        self._reserved_slots: weakref.WeakSet[SQLiteWriteSlot] = weakref.WeakSet()
        self._opened = False
        self._pid = os.getpid()
        self._forked_during_write = False

    @property
    def read_only(self) -> bool:
        """Storage access mode."""
        return self._read_only

    @property
    def codec(self) -> CodecProtocol[bytes, bytes]:
        """Get codec for key/value encoding."""
        return self._codec

    @property
    def path(self) -> Path:
        """The database file."""
        return self._path

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def open(self) -> None:
        """Open the database, set it up, and seed the connection pool.

        Read-write: creates the file and parent directory if missing, turns
        on WAL (persisted in the file) and creates the ``kv`` table.

        Read-only: opens ``file:<path>?mode=ro``. The file must exist and
        already hold the table. SQLite's WAL readers still write the
        ``-shm`` index next to the file, so the directory (or an existing
        ``-shm``) must be writable by the reader; a reader on a truly
        read-only filesystem needs the file out of WAL mode.
        """
        if self._opened:
            return

        if not self._read_only:
            if self._create_if_missing:
                try:
                    self._path.parent.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    raise StorageError(f"Failed to create database directory: {e}") from e
            elif not self._path.exists():
                raise StorageError(f"Database does not exist: {self._path}")
        elif not self._path.exists():
            raise StorageError(f"Database does not exist: {self._path}")

        try:
            conn = self._connect()
        except Exception as e:
            raise StorageError(f"Failed to open SQLite database: {e}") from e

        try:
            if not self._read_only:
                if self._wal:
                    mode = conn.execute("PRAGMA journal_mode = WAL").fetchone()[0]
                    if str(mode).lower() != "wal":
                        raise StorageError(f"Could not enable WAL mode (got {mode!r})")
                conn.execute(_SCHEMA)
            else:
                conn.execute("SELECT 1 FROM kv LIMIT 1").fetchall()
        except StorageError:
            conn.close()
            raise
        except Exception as e:
            conn.close()
            raise StorageError(f"Failed to initialize SQLite database: {e}") from e

        with self._state_lock:
            self._idle.append(conn)
            self._pid = os.getpid()
            self._opened = True
        _OPEN_STORAGES.add(self)

    def close(self) -> None:
        """Close database and release all resources.

        Aborts open transactions and batches, closes open snapshots, gives
        back reserved write slots' connections, then closes every pooled
        connection. The last connection to close (in any process)
        checkpoints the WAL back into the file. The storage can be opened
        again afterwards.

        A context in use on another thread is ended once its current
        statement finishes (see ``ContextBase``), so close never pulls a
        connection away mid-statement. In a forked child nothing inherited
        from the parent is touched.
        """
        self._check_fork()  # never end contexts inherited from the parent

        with self._state_lock:
            if not self._opened:
                return
            # Closed from here on: begins fail at once, and connections
            # given back by the contexts ended below are closed, not pooled.
            self._opened = False
            transactions = list(self._active_transactions)
            snapshots = list(self._active_snapshots)
            write_batches = list(self._active_write_batches)
            reserved = list(self._reserved_slots)
            self._active_transactions.clear()
            self._active_write_batches.clear()
            self._active_snapshots.clear()
            self._reserved_slots.clear()
            idle, self._idle = self._idle, []

        for transaction in transactions:
            try:
                transaction.abort()
            except Exception as e:
                logger.error(f"Transaction abort failed during close: {e}")

        for snapshot in snapshots:
            try:
                snapshot.close()
            except Exception as e:
                logger.error(f"Snapshot close failed during close: {e}")

        for write_batch in write_batches:
            try:
                write_batch.abort()
            except Exception as e:
                logger.error(f"Write batch abort failed during close: {e}")

        for slot in reserved:
            # The slot keeps its in-process lock (its holder releases that);
            # only the begun connection, and with it the file lock, goes.
            try:
                slot._drop_conn()
            except Exception as e:
                logger.error(f"Write slot release failed during close: {e}")

        self._writer.close()  # a reserved slot nobody took does not survive the reopen

        _OPEN_STORAGES.discard(self)
        errors: list[Exception] = []
        for conn in idle:
            try:
                conn.close()
            except Exception as e:
                errors.append(e)
        if errors:
            raise StorageError(f"Failed to close database: {errors[0]}") from errors[0]

    def __enter__(self) -> SQLiteStorage:
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

    def begin_snapshot(self) -> SQLiteSnapshot:
        """Begin read-only snapshot, pinned at this moment."""
        self._require_open()
        conn = self._acquire_connection()
        try:
            conn.execute("BEGIN")
            # A deferred transaction only takes its read mark at the first
            # read; do one now so the snapshot is as of begin, not first use.
            conn.execute("SELECT 1 FROM kv LIMIT 1").fetchall()
        except Exception as e:
            self._release_connection(conn)
            if is_busy_error(e):
                raise StorageLockTimeoutError(f"Timed out beginning snapshot: {e}") from e
            raise StorageError(f"Failed to begin snapshot: {e}") from e

        snapshot = SQLiteSnapshot(self, conn)
        with self._state_lock:
            self._active_snapshots.add(snapshot)
        return snapshot

    def begin_transaction(self, *, write_slot: SQLiteWriteSlot | None = None) -> SQLiteTransaction:
        """Begin read-write transaction, taking the write lock now.

        Args:
            write_slot: A slot from ``areserve_write_slot``. The transaction
                takes it over instead of acquiring the slot itself, then
                releases it on commit or abort. None acquires as usual.
        """
        if self._read_only:
            raise StorageError("Cannot start transaction in read only mode.")
        self._require_open()

        conn, writer_lock = self._begin_write_connection(write_slot)
        transaction = SQLiteTransaction(self, conn, writer_lock)
        with self._state_lock:
            self._active_transactions.add(transaction)
        return transaction

    async def areserve_write_slot(self) -> SQLiteWriteSlot:
        """Wait for the write lock without blocking the event loop.

        Two waits, both polled with a short, growing sleep between tries so
        other coroutines on the loop keep running, including the one that
        holds the lock: first the in-process writer slot (behind other
        tasks and threads), then the cross-process file lock, taken with
        ``BEGIN IMMEDIATE`` on a pooled connection with the busy handler
        off, so each try returns at once. Both share one ``busy_timeout``
        deadline. The slot carries the begun connection into
        ``begin_transaction(write_slot=...)``.

        Cancelled while waiting, it holds nothing: the lock is only ever
        taken by a non-blocking try, nothing is awaited between taking it
        and returning, and a cancel in the file lock wait releases the
        slot and gives the connection back.

        Returns:
            The held slot, for ``begin_transaction(write_slot=...)``. Call
            its ``release`` if no transaction takes it.

        Raises:
            StorageNestedTransactionError: If this task already holds the
                slot.
            StorageLockTimeoutError: If the slot or the file lock is not
                free within ``busy_timeout``.
        """
        if self._read_only:
            raise StorageError("Cannot reserve the write slot in read only mode.")
        self._require_open()
        self._check_fork()
        self._require_writable_after_fork()
        deadline = time.monotonic() + max(self._busy_timeout, 0)
        lock = await self._writer.areserve(self._busy_timeout)
        try:
            conn = await self._abegin_immediate(deadline)
        except BaseException:
            self._writer.release(lock)
            raise
        slot = SQLiteWriteSlot(self, lock, conn)
        with self._state_lock:
            self._reserved_slots.add(slot)
        return slot

    async def _abegin_immediate(self, deadline: float) -> sqlite3.Connection:
        """Take the file write lock on a pooled connection, polling off the loop.

        ``PRAGMA busy_timeout = 0`` makes each ``BEGIN IMMEDIATE`` a single
        non-blocking attempt (SQLite returns busy at once instead of
        sleeping in its handler); the wait between attempts is an
        ``asyncio.sleep``. The connection's busy handler is restored
        before it goes anywhere, since the pool shares it with sync paths.
        """
        conn = self._acquire_connection()
        try:
            try:
                conn.execute("PRAGMA busy_timeout = 0")
                delay = _LOCK_POLL_MIN
                while True:
                    try:
                        conn.execute("BEGIN IMMEDIATE")
                        return conn
                    except Exception as e:
                        if not is_busy_error(e):
                            raise StorageError(f"Failed to begin write transaction: {e}") from e
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise StorageLockTimeoutError(
                                f"Timed out after {self._busy_timeout}s waiting for the "
                                f"write lock: {e}"
                            ) from e
                    await asyncio.sleep(min(delay, remaining))
                    delay = min(delay * 2, _LOCK_POLL_MAX)
            finally:
                conn.execute(f"PRAGMA busy_timeout = {int(self._busy_timeout * 1000)}")
        except BaseException:
            self._release_connection(conn)
            raise

    def begin_write_batch(self) -> SQLiteWriteBatch:
        """Begin write-only batch. Takes no lock until ``write``."""
        if self._read_only:
            raise StorageError("Cannot start write batch in read only mode.")
        self._require_open()
        self._check_fork()
        self._require_writable_after_fork()

        write_batch = SQLiteWriteBatch(self)
        with self._state_lock:
            self._active_write_batches.add(write_batch)
        return write_batch

    @contextmanager
    def transaction(self) -> Iterator[SQLiteTransaction]:
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
    def snapshot(self) -> Iterator[SQLiteSnapshot]:
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
    def batch_write(self) -> Iterator[SQLiteWriteBatch]:
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
    # Connections
    # =========================================================================

    def _connect(self) -> sqlite3.Connection:
        """Open and configure a new connection."""
        if self._read_only:
            uri = self._path.resolve().as_uri() + "?mode=ro"
            conn = sqlite3.connect(
                uri,
                uri=True,
                timeout=self._busy_timeout,
                isolation_level=None,
                check_same_thread=False,
            )
        else:
            conn = sqlite3.connect(
                str(self._path),
                timeout=self._busy_timeout,
                isolation_level=None,
                check_same_thread=False,
            )
        try:
            conn.execute(f"PRAGMA synchronous = {self._synchronous}")
            if self._mmap_size is not None:
                conn.execute(f"PRAGMA mmap_size = {int(self._mmap_size)}")
            if self._cache_size is not None:
                conn.execute(f"PRAGMA cache_size = {int(self._cache_size)}")
            for name, value in self._pragmas.items():
                conn.execute(f"PRAGMA {name} = {value}")
        except Exception:
            conn.close()
            raise
        return conn

    def _acquire_connection(self) -> sqlite3.Connection:
        """Take an idle pooled connection, or open a new one."""
        self._check_fork()
        with self._state_lock:
            if self._idle:
                return self._idle.pop()
        try:
            return self._connect()
        except Exception as e:
            raise StorageError(f"Failed to open SQLite connection: {e}") from e

    def _release_connection(self, conn: sqlite3.Connection) -> None:
        """Return a connection to the pool, rolled back; close it if not wanted."""
        try:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
        except Exception as e:
            logger.error(f"Rollback on connection release failed: {e}")
            conn.close()
            return
        with self._state_lock:
            if self._opened and len(self._idle) < self._max_idle_connections:
                self._idle.append(conn)
                return
        conn.close()

    def _begin_write_connection(
        self, write_slot: SQLiteWriteSlot | None = None
    ) -> tuple[sqlite3.Connection, threading.Lock]:
        """Take the writer slot and a connection with ``BEGIN IMMEDIATE`` run.

        With ``write_slot`` the slot is already held and is taken over, not
        acquired, and its begun connection comes with it, so nothing waits
        here. Otherwise the slot is acquired and ``BEGIN IMMEDIATE`` waits
        in SQLite's busy handler for up to ``busy_timeout``. On failure
        nothing is held: the connection is given back, and the slot is
        released, or returned to ``write_slot``'s holder.

        Returns:
            The connection, and the writer lock held for it (to release by
            identity when the transaction ends).
        """
        self._check_fork()
        self._require_writable_after_fork()
        give_back_slot = self._writer.take(self, write_slot, self._busy_timeout)
        writer_lock = self._writer.lock

        conn = write_slot._detach_conn() if write_slot is not None else None
        if conn is not None:
            return conn, writer_lock

        try:
            conn = self._acquire_connection()
        except Exception:
            give_back_slot()
            raise

        try:
            conn.execute("BEGIN IMMEDIATE")
        except Exception as e:
            self._release_connection(conn)
            give_back_slot()
            if is_busy_error(e):
                raise StorageLockTimeoutError(
                    f"Timed out after {self._busy_timeout}s waiting for the write lock: {e}"
                ) from e
            raise StorageError(f"Failed to begin write transaction: {e}") from e
        return conn, writer_lock

    def _check_fork(self) -> None:
        """Drop state inherited across a fork.

        Runs from the fork hook in the child, and lazily from every entry
        point in case the hook did not. If this process is not the one that
        opened the storage: the parent's open contexts are marked ended
        without being ended (no ROLLBACK on their connections), every
        inherited connection is parked for the life of the process (see
        ``_INHERITED_CONNECTIONS``), reserved slots lose their connections
        the same way, and the writer slot is reset so a lock held by a
        parent thread does not wait forever here. If the parent held the
        write lock at the fork, writes on this file are refused from now
        on (see ``_require_writable_after_fork``).
        """
        if self._pid == os.getpid():
            return
        # Another parent thread may have held the state lock at the fork;
        # it does not exist here, so a fresh lock is the only usable one.
        self._state_lock = threading.Lock()
        with self._state_lock:
            if self._pid == os.getpid():
                return
            contexts = [
                *self._active_transactions,
                *self._active_snapshots,
                *self._active_write_batches,
            ]
            reserved = list(self._reserved_slots)
            held_write = bool(
                self._writer.lock.locked()
                or self._active_transactions
                or any(b._conn is not None for b in self._active_write_batches)
                or reserved
            )
            inherited = list(self._idle)
            for context in contexts:
                if context._conn is not None:
                    inherited.append(context._conn)
                context._forget()
            for slot in reserved:
                conn = slot._detach_conn()
                if conn is not None:
                    inherited.append(conn)
            self._idle = []
            self._active_transactions = weakref.WeakSet()
            self._active_snapshots = weakref.WeakSet()
            self._active_write_batches = weakref.WeakSet()
            self._reserved_slots = weakref.WeakSet()
            self._writer.reset()
            if held_write:
                self._forked_during_write = True
                _FORK_WRITE_POISONED.add(self._poison_key())
            self._pid = os.getpid()
        self._park_inherited(inherited)

    def _poison_key(self) -> str:
        """The file identity ``_FORK_WRITE_POISONED`` is keyed by."""
        try:
            return str(self._path.resolve())
        except OSError:
            return str(self._path)

    def _require_writable_after_fork(self) -> None:
        """Refuse writes on a file whose write lock a forked parent held."""
        if self._forked_during_write or (
            _FORK_WRITE_POISONED and self._poison_key() in _FORK_WRITE_POISONED
        ):
            raise StorageError(
                "This SQLite database was inherited across a fork while the parent held "
                "its write lock, so it cannot be written from this process (the inherited "
                "lock state marks the lock as taken here for good, and SQLite forbids "
                "touching inherited connections to undo that). Open the storage in a "
                "spawned process instead, or fork only while no write is in progress"
            )

    @staticmethod
    def _park_inherited(conns: list[sqlite3.Connection]) -> None:
        """Keep inherited connections alive, untouched, for the life of the process."""
        _INHERITED_CONNECTIONS.extend(conns)

    def _release_write_slot(self, lock: threading.Lock | None = None) -> None:
        """Release the in-process writer slot.

        Args:
            lock: The lock the caller acquired, if known. Nothing is
                released when it is no longer the storage's lock (a fork
                reset the slot since).
        """
        self._writer.release(lock)

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

    def _untrack_transaction(self, transaction: SQLiteTransaction) -> None:
        """Remove transaction from active set."""
        with self._state_lock:
            self._active_transactions.discard(transaction)

    def _untrack_snapshot(self, snapshot: SQLiteSnapshot) -> None:
        """Remove snapshot from active set."""
        with self._state_lock:
            self._active_snapshots.discard(snapshot)

    def _untrack_write_batch(self, write_batch: SQLiteWriteBatch) -> None:
        """Remove write batch from active set."""
        with self._state_lock:
            self._active_write_batches.discard(write_batch)

    def _require_open(self) -> None:
        """Validate storage is open."""
        if not self._opened:
            raise StorageClosedError("Storage is not open")


if TYPE_CHECKING:
    _: type[TransactionProtocol] = SQLiteTransaction
    __: type[SnapshotProtocol] = SQLiteSnapshot
