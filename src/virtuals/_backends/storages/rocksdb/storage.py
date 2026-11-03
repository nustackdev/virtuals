"""RocksDB storage backend implementation.

Provides persistent key-value storage with serializable transactions,
point-in-time snapshots, and optional change notifications via a publisher.
"""

from __future__ import annotations

import contextlib
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
    StorageTransactionError,
    TransactionProtocol,
    WriteBatchProtocol,
)

from .._write_slot import WriterLock, WriteSlot
from .context import classify_rdbpy
from .snapshot import RocksDBSnapshot
from .transaction import RocksDBTransaction
from .write_batch import RocksDBWriteBatch


if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import TracebackType

    from virtuals.tkv.codec import CodecProtocol
    from virtuals.tkv.publisher import PublisherProtocol
    from virtuals.tkv.types import Key

try:
    import rdbpy
except ImportError as e:
    raise ImportError(
        "RocksDB python bindings (rdbpy) are required for RocksDBStorage. "
        "Install via: [uv] pip install rdbpython"
    ) from e


__all__ = ["RocksDBStorage", "RocksDBWriteSlot"]


logger = getLogger(__name__)


DEFAULT_BUSY_TIMEOUT = 60.0  # seconds

# Native handles a forked child inherited and was asked to close. Freeing
# them there would run rocksdb teardown (a memtable flush, waits on
# background jobs, lock manager mutexes) against threads and locks that only
# exist in the parent: it can hang or write into the parent's directory. The
# child keeps them referenced instead, until it exits.
_INHERITED_HANDLES: list[Any] = []

RocksDBWriteSlot = WriteSlot
"""Returned by ``RocksDBStorage.areserve_write_slot``; see ``WriteSlot``."""

# Keys of the ``txn_db_options`` dict, as rdbpy.TransactionDB spells them,
# plus the names rdbpy 0.2 used (values in milliseconds either way).
_TXN_DB_OPTION_ALIASES = {
    "transaction_lock_timeout": "lock_timeout_ms",
    "default_lock_timeout": "default_lock_timeout_ms",
}
# Keys of the ``txn_options`` dict, as rdbpy ``TransactionDB.begin`` spells
# them, plus the rdbpy 0.2 names.
_TXN_OPTION_ALIASES = {
    "lock_timeout": "lock_timeout_ms",
    "expiration": "expiration_ms",
}


def _is_lock_file_error(e: BaseException) -> bool:
    """True if an open failed on the directory's LOCK file: it is open elsewhere."""
    text = str(e)
    return isinstance(e, rdbpy.RocksIOError) and (
        "lock hold" in text or ("LOCK" in text and "Resource temporarily unavailable" in text)
    )


class RocksDBStorage:
    """RocksDB storage implementation conforming to StorageProtocol.

    The serial variant: one writer at a time per storage, real snapshots
    for every read.

    Transactions: a write transaction holds the storage's in-process writer
    slot from begin until commit or abort. With one writer at a time and
    every read inside the transaction at its own begin snapshot (merged
    with its pending writes), committed transactions are serializable,
    scans included: no lost update, write skew or phantom is possible.
    Underneath, a pessimistic rocksdb transaction with snapshot validation
    turns any writer that bypassed the slot into a retryable conflict.

    Snapshots: a rocksdb DB snapshot, taken when the Snapshot begins, pins
    every get and scan in it. They never wait on a writer.

    Coroutines: the slot belongs to a thread and, on an event loop, to the
    task that took it. Coroutines sharing one loop wait for it with
    ``areserve_write_slot``, which polls without blocking the loop, and
    hand it to ``begin_transaction(write_slot=...)``. A second writer from
    the task or thread that already holds the slot raises
    ``StorageNestedTransactionError``; a sync begin while another task on
    the same loop holds it raises ``StorageLockTimeoutError`` at once. A
    sync writer on another thread blocks for up to ``busy_timeout``, except
    on an event loop thread, where a sync begin or batch write waits at
    most ``LOOP_THREAD_WAIT_CAP`` for a holder on another thread before
    raising ``StorageLockTimeoutError`` (retryable), so the loop is never
    stalled for long. Write batches take the slot for the duration of
    ``write``.

    Process model: single-process. RocksDB locks its directory, so one
    process opens a storage read-write; others open it as a read-only
    secondary (``read_only=True, secondary_path=...``) or go through the
    owning process. A storage handle must not be used in a forked child.

    A transaction, snapshot or scan belongs to one thread: do not share it
    between threads, and close the storage only once every thread is done.
    """

    process_model: ClassVar[Literal["single", "multi"]] = "single"
    """Contract section 4: one process opens the storage read-write."""

    def __init__(
        self,
        path: Path | str,
        codec: CodecProtocol[bytes, bytes],
        publisher: PublisherProtocol | None = None,
        *,
        read_only: bool = False,
        secondary_path: Path | str | None = None,
        secondary_refresh_interval: float | None = 0.05,
        wal_path: Path | str | None = None,
        options: dict[str, Any] | None = None,
        txn_db_options: dict[str, Any] | None = None,
        txn_options: dict[str, Any] | None = None,
        create_if_missing: bool = True,
        sync_writes: bool = False,
        disable_wal: bool = False,
        busy_timeout: float = DEFAULT_BUSY_TIMEOUT,
    ) -> None:
        """Initialize RocksDB storage.

        Args:
            path: Database directory path
            codec: Codec for key/value encoding
            publisher: Optional publisher for change notifications
            read_only: Open database in read-only mode
            secondary_path: Path to open db via "rocksdb::DB::OpenAsSecondary"
                (allows multiple parallel readers)
            secondary_refresh_interval: Minimum interval in seconds between
                try_catch_up_with_primary() calls on secondary DBs. Catch-up
                runs lazily inside begin_snapshot when the last catch-up is
                older than this. Set to 0 to catch up on every snapshot, or
                None to never catch up (callers drive freshness manually).
            wal_path: Optional separate WAL directory
            options: ``rdbpy.Options`` kwargs. ``block_cache_mb`` (an int)
                and ``block_table`` (a dict of block table kwargs) are also
                accepted and folded in.
            txn_db_options: ``rdbpy.TransactionDB`` kwargs: ``lock_timeout_ms``,
                ``default_lock_timeout_ms``, ``max_num_locks``, ``num_stripes``,
                ``max_num_deadlocks``. Row locks only matter for a writer that
                bypassed the writer slot.
            txn_options: ``rdbpy.TransactionDB.begin`` kwargs: ``lock_timeout_ms``,
                ``deadlock_detect``, ``deadlock_detect_depth``, ``expiration_ms``.
                Transactions always begin with a snapshot.
            create_if_missing: Create database if it doesn't exist
            sync_writes: fsync every commit and batch write
            disable_wal: Disable write-ahead log
            busy_timeout: Seconds a sync writer waits for the writer slot
                before ``StorageLockTimeoutError``.
        """
        # Core dependencies
        self._codec = codec
        self._publisher = publisher

        # Open options
        self._read_only = read_only
        self._is_secondary = bool(secondary_path)
        self._secondary_path: Path | None = None
        if secondary_path:
            self._secondary_path = (
                Path(secondary_path) if isinstance(secondary_path, str) else secondary_path
            )

        if not self._read_only and self._is_secondary:
            raise ValueError("Secondary dbs can only be opened in readonly mode.")

        self._secondary_refresh_interval = secondary_refresh_interval
        self._last_catchup_monotonic: float = 0.0

        # Paths
        self._path = Path(path) if isinstance(path, str) else path
        self._wal_path = Path(wal_path) if isinstance(wal_path, str) else wal_path

        # Configuration
        self._options_dict = options or {}
        self._txn_db_options_dict = txn_db_options or {}
        self._txn_options_dict = txn_options or {}
        self._create_if_missing = create_if_missing
        self._sync_writes = sync_writes
        self._disable_wal = disable_wal
        self._busy_timeout = busy_timeout
        self._begin_kwargs = self._build_begin_kwargs()

        # State
        self._db: rdbpy.TransactionDB | rdbpy.DB | None = None
        self._state_lock = threading.Lock()
        self._writer = WriterLock("RocksDB")
        # Weak: a transaction or batch dropped without an end must be
        # collectable, the transaction's finalizer then frees the slot.
        self._active_transactions: weakref.WeakSet[RocksDBTransaction] = weakref.WeakSet()
        self._active_write_batches: weakref.WeakSet[RocksDBWriteBatch] = weakref.WeakSet()
        self._active_snapshots: set[RocksDBSnapshot] = set()
        self._opened = False
        self._pid = os.getpid()

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
        """The database directory."""
        return self._path

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def _build_options(self) -> rdbpy.Options:
        """Build RocksDB options from configuration.

        Returns:
            Configured Options object

        Raises:
            StorageError: If options are invalid
        """
        options_dict = dict(self._options_dict)
        if "create_if_missing" not in options_dict:
            options_dict["create_if_missing"] = self._create_if_missing
        if self._read_only:
            # rocksdb creates the directory on a read-only open too when
            # asked to; a read-only open of a missing path must leave nothing.
            options_dict["create_if_missing"] = False

        block_table_kwargs = options_dict.pop("block_table", None)
        if block_table_kwargs:
            options_dict.update(block_table_kwargs)
        block_cache_mb = options_dict.pop("block_cache_mb", None)
        if block_cache_mb is not None and "block_cache_size" not in options_dict:
            options_dict["block_cache_size"] = int(block_cache_mb) * 1024 * 1024

        if self._wal_path is not None:
            options_dict["wal_dir"] = str(self._wal_path)

        try:
            return rdbpy.Options(**options_dict)
        except Exception as e:
            raise StorageError(f"Invalid RocksDB options: {e}") from e

    def _build_txn_db_kwargs(self) -> dict[str, Any]:
        """Translate ``txn_db_options`` into ``rdbpy.TransactionDB`` kwargs."""
        return {_TXN_DB_OPTION_ALIASES.get(k, k): v for k, v in self._txn_db_options_dict.items()}

    def _build_begin_kwargs(self) -> dict[str, Any]:
        """Translate ``txn_options`` into ``TransactionDB.begin`` kwargs."""
        kwargs = {_TXN_OPTION_ALIASES.get(k, k): v for k, v in self._txn_options_dict.items()}
        kwargs.pop("set_snapshot", None)  # always on
        kwargs["set_snapshot"] = True
        kwargs["sync"] = self._sync_writes
        kwargs["disable_wal"] = self._disable_wal
        return kwargs

    def _open_transaction_db(self, options: rdbpy.Options) -> None:
        """Open database as TransactionDB for read-write access.

        Args:
            options: RocksDB options

        Raises:
            StorageError: If database cannot be opened
        """
        try:
            self._db = rdbpy.TransactionDB(str(self._path), options, **self._build_txn_db_kwargs())
        except TypeError as e:
            raise StorageError(f"Invalid TransactionDB options: {e}") from e
        except Exception as e:
            if _is_lock_file_error(e):
                raise StorageError(
                    "RocksDB is single-process and this database is already open, in this "
                    "process or another. Open it once and share the storage, or open a "
                    f"read-only secondary (read_only=True, secondary_path=...): {e}"
                ) from e
            raise StorageError(f"Failed to open RocksDB TransactionDB: {e}") from e

    def _open_read_only_db(self, options: rdbpy.Options) -> None:
        """Open an existing database read-only (no secondary instance).

        Args:
            options: RocksDB options

        Raises:
            StorageError: If database cannot be opened
        """
        try:
            self._db = rdbpy.DB(str(self._path), options, read_only=True)
        except Exception as e:
            raise StorageError(f"Failed to open RocksDB read-only: {e}") from e

    def _open_secondary_db(self, options: rdbpy.Options) -> None:
        """Open database as secondary for parallel reads.

        Args:
            options: RocksDB options

        Raises:
            StorageError: If database cannot be opened
        """
        if not self._is_secondary or self._secondary_path is None:
            raise StorageError("Trying to open regular db as secondary db")

        # Secondary DB needs a separate path for its logs
        try:
            self._secondary_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise StorageError(f"Failed to create secondary path: {e}") from e

        try:
            self._db = rdbpy.DB(
                str(self._path.resolve()),
                options,
                read_only=True,
                secondary_path=str(self._secondary_path.resolve()),
            )
        except Exception as e:
            raise StorageError(f"Failed to open RocksDB as secondary: {e}") from e

    def open(self) -> None:
        """Open RocksDB database and initialize resources.

        Raises:
            StorageError: If database cannot be opened
        """
        if self._opened:
            return

        # Create directories, only for an open that may create the database:
        # a read-only open of a missing path must fail without leaving an
        # empty directory behind for a later read-write open to adopt.
        creates = not self._read_only and self._options_dict.get(
            "create_if_missing", self._create_if_missing
        )
        existed = creates or self._path.exists()
        if creates:
            try:
                self._path.mkdir(parents=True, exist_ok=True)
                if self._wal_path is not None:
                    self._wal_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise StorageError(f"Failed to create database directories: {e}") from e

        options = self._build_options()

        # Open in appropriate mode
        try:
            if self._is_secondary:
                self._open_secondary_db(options)
            elif self._read_only:
                self._open_read_only_db(options)
            else:
                self._open_transaction_db(options)
        except StorageError:
            if not existed:
                # rocksdb makes the directory before it finds no database
                # in it; an empty one left behind would be adopted silently
                # by the next read-write open.
                with contextlib.suppress(OSError):
                    self._path.rmdir()
            raise

        with self._state_lock:
            self._pid = os.getpid()
            self._opened = True

    def close(self) -> None:
        """Close database and release all resources.

        Closes the database first: rdbpy refuses while another thread is
        inside a call on it or on one of its transactions, snapshots or
        iterators, and then nothing is touched and the storage stays open.
        Once the database is closed (rdbpy rolls back its transactions and
        releases its snapshots), open transactions, snapshots and batches
        are marked ended and the writer slot is replaced: a slot reserved
        but never taken, or held by a context still finishing elsewhere,
        does not survive into a reopen, and a writer waiting on it wakes
        with ``StorageClosedError``.

        Raises:
            StorageError: If close fails, e.g. a call is still in flight on
                another thread
        """
        if not self._opened:
            return
        if self._pid != os.getpid():
            self._forget_inherited()
            return

        db = self._db
        if db is not None:
            try:
                db.close()
            except Exception as e:
                raise StorageError(f"Failed to close database: {e}") from e

        with self._state_lock:
            transactions = list(self._active_transactions)
            snapshots = list(self._active_snapshots)
            write_batches = list(self._active_write_batches)
            self._active_transactions.clear()
            self._active_write_batches.clear()
            self._active_snapshots.clear()
            self._db = None
            self._opened = False

        # Native state is gone; these only mark the contexts ended and give
        # back the writer slot. A waiter woken by that sees a closed storage.
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

        self._writer.close()

    def _forget_inherited(self) -> None:
        """Close in a forked child: forget the parent's handles, free nothing native."""
        with self._state_lock:
            contexts = [
                *self._active_transactions,
                *self._active_snapshots,
                *self._active_write_batches,
            ]
            self._active_transactions.clear()
            self._active_write_batches.clear()
            self._active_snapshots.clear()
            _INHERITED_HANDLES.append((self._db, contexts))
            self._db = None
            self._opened = False
            self._writer.reset()
        for context in contexts:
            # Ended without touching its rdbpy handle, which stays referenced.
            context._holds_write_slot = False
            context._closed = True

    def __enter__(self) -> RocksDBStorage:
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
        """Begin transaction or snapshot.

        Args:
            read_only: If True, creates a read-only snapshot
            write_only: If True, creates a write-only batch

        Returns:
            SnapshotProtocol if read_only=True
            WriteBatchProtocol if write_only=True
            TransactionProtocol otherwise

        Raises:
            StorageError: If begin fails
            StorageClosedError: If storage is not open
        """
        if read_only:
            return self.begin_snapshot()
        elif write_only:
            return self.begin_write_batch()
        else:
            return self.begin_transaction()

    def begin_snapshot(self) -> RocksDBSnapshot:
        """Begin read-only snapshot, pinned at this moment.

        Returns:
            New snapshot instance

        Raises:
            StorageError: If snapshot creation fails
            StorageClosedError: If storage is not open
        """
        self._require_open()
        db = self._db

        if self._is_secondary:
            self._catch_up_if_due()
            rdbpy_snapshot = None  # rocksdb: snapshots not supported in secondary mode
        else:
            try:
                rdbpy_snapshot = db.snapshot()
            except Exception as e:
                raise classify_rdbpy(e, "Failed to begin snapshot") from e

        snapshot = RocksDBSnapshot(self, db, rdbpy_snapshot)
        with self._state_lock:
            self._active_snapshots.add(snapshot)
        return snapshot

    def _catch_up_if_due(self) -> None:
        """Throttled catch-up of a secondary with the primary, before a snapshot."""
        interval = self._secondary_refresh_interval
        if interval is None:
            return
        now = time.monotonic()
        if now - self._last_catchup_monotonic < interval:
            return
        try:
            self._db.try_catch_up_with_primary()
        except Exception as e:
            raise StorageError(f"Failed to catch up with primary: {e}") from e
        self._last_catchup_monotonic = now

    def force_catch_up_with_primary(self) -> None:
        """Catch up to the primary's current manifest immediately.

        Unlike the throttled catch-up in `begin_snapshot`, this bypasses
        `secondary_refresh_interval`. It exists to recover from a read that
        hit an SST file the primary compacted away: the secondary is pinned
        to an old manifest version that still references the deleted file,
        and catching up to the current manifest drops that reference. No-op
        on a primary (non-secondary) storage.
        """
        if not self._is_secondary:
            return
        self._require_open()
        try:
            self._db.try_catch_up_with_primary()
        except Exception as e:
            raise StorageError(f"Failed to catch up with primary: {e}") from e
        self._last_catchup_monotonic = time.monotonic()

    def begin_transaction(self, *, write_slot: WriteSlot | None = None) -> RocksDBTransaction:
        """Begin read-write transaction, taking the writer slot now.

        Args:
            write_slot: A slot from ``areserve_write_slot``. The transaction
                takes it over instead of acquiring the slot itself, then
                releases it on commit or abort. None acquires as usual.

        Returns:
            New transaction instance

        Raises:
            StorageNestedTransactionError: If this task or thread already
                holds the writer slot
            StorageLockTimeoutError: If the slot is not free within
                ``busy_timeout``, or is held by another task on this loop
            StorageError: If transaction creation fails
            StorageClosedError: If storage is not open
        """
        if self._read_only:
            raise StorageError("Cannot start transaction in read only mode.")
        self._require_open()

        lock, give_back_slot = self._writer.take_slot(self, write_slot, self._busy_timeout)
        try:
            db = self._db
            if db is None:
                raise StorageClosedError("Storage is not open")
            rdbpy_txn = db.begin(**self._begin_kwargs)
        except StorageClosedError:
            give_back_slot()
            raise
        except TypeError as e:
            give_back_slot()
            raise StorageError(f"Invalid transaction options: {e}") from e
        except Exception as e:
            give_back_slot()
            # A database closed under us is StorageClosedError, not a generic failure.
            raise classify_rdbpy(e, "Failed to begin transaction") from e

        transaction = RocksDBTransaction(self, rdbpy_txn, lock)
        with self._state_lock:
            self._active_transactions.add(transaction)
        return transaction

    async def areserve_write_slot(self) -> WriteSlot:
        """Wait for the writer slot without blocking the event loop.

        Polls the slot with a short, growing sleep between tries, so other
        coroutines on the loop keep running, including the one that holds
        it. Waits behind other threads too.

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
        lock = await self._writer.areserve(self._busy_timeout)
        return RocksDBWriteSlot(self, lock)

    def begin_write_batch(self) -> RocksDBWriteBatch:
        """Begin write-only batch. Takes the writer slot only in ``write``.

        Returns:
            New write batch instance

        Raises:
            StorageError: If batch creation fails in read only mode
            StorageClosedError: If storage is closed
        """
        if self._read_only:
            raise StorageError("Cannot start write batch in read only mode.")
        self._require_open()

        write_batch = RocksDBWriteBatch(self, rdbpy.WriteBatch())
        with self._state_lock:
            self._active_write_batches.add(write_batch)
        return write_batch

    def _write_batch(self, batch: Any) -> None:  # noqa: ANN401
        """Apply an rdbpy write batch atomically, under the writer slot.

        Raises:
            StorageNestedTransactionError: If this task or thread holds a
                write transaction on the storage
            StorageLockTimeoutError: If the slot is not free in time, or
                rocksdb timed out on a row lock
            StorageTransactionConflictError: If rocksdb refused the write
            StorageTransactionError: If the write fails otherwise
        """
        give_back_slot = self._writer.take(self, None, self._busy_timeout)
        try:
            self._require_open()
            self._db.write(batch, sync=self._sync_writes, disable_wal=self._disable_wal)
        except StorageError:
            raise
        except Exception as e:
            mapped = classify_rdbpy(e, "Failed to write batch")
            if isinstance(mapped, (StorageTransactionError, StorageClosedError)):
                raise mapped from e  # retryable (conflict, lock timeout) or closed
            raise StorageTransactionError(f"Failed to write batch: {e}") from e
        finally:
            give_back_slot()

    def _release_write_slot(self, lock: Any = None) -> None:  # noqa: ANN401
        """Release the in-process writer slot.

        Args:
            lock: The lock the caller acquired, if known. Nothing is
                released when it is no longer the storage's lock (a close
                or a fork replaced the slot since).
        """
        self._writer.release(lock)

    @contextmanager
    def transaction(self) -> Iterator[RocksDBTransaction]:
        """Context manager for a read-write transaction.

        Commits on successful exit, aborts on any exception (including
        KeyboardInterrupt, cancellation and GeneratorExit) so the writer
        slot is never left held.
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
    def snapshot(self) -> Iterator[RocksDBSnapshot]:
        """Context manager for a read-only snapshot.

        Ensures snapshot is closed on exit.
        """
        snap = self.begin_snapshot()
        try:
            yield snap
        finally:
            try:
                snap.close()
            except Exception as e:
                logger.error(f"Snapshot close failed: {e}")

    @contextmanager
    def batch_write(self) -> Iterator[RocksDBWriteBatch]:
        """Context manager for a write batch.

        Writes on successful exit, aborts on any exception.
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

        Args:
            keys: Keys that changed
        """
        if self._publisher is not None and keys:
            try:
                self._publisher.notify(keys)
            except Exception as e:
                logger.error(f"Publisher notification failed: {e}")

    def _untrack_transaction(self, transaction: RocksDBTransaction) -> None:
        """Remove transaction from active set."""
        with self._state_lock:
            self._active_transactions.discard(transaction)

    def _untrack_snapshot(self, snapshot: RocksDBSnapshot) -> None:
        """Remove snapshot from active set."""
        with self._state_lock:
            self._active_snapshots.discard(snapshot)

    def _untrack_write_batch(self, write_batch: RocksDBWriteBatch) -> None:
        """Remove write batch from active set."""
        with self._state_lock:
            self._active_write_batches.discard(write_batch)

    def _require_open(self) -> None:
        """Validate storage is open, in the process that opened it.

        Raises:
            StorageClosedError: If storage is not open
            StorageError: If this is a forked child of the opening process:
                the rocksdb handle (background threads, directory lock)
                does not survive a fork, so the child must open its own
        """
        if not self._opened or self._db is None:
            raise StorageClosedError("Storage is not open")
        if self._pid != os.getpid():
            raise StorageError(
                "RocksDB storage was opened in another process; a forked child "
                "cannot use its handle. RocksDB is single-process: open the "
                "storage in this process instead"
            )


if TYPE_CHECKING:
    _: type[TransactionProtocol] = RocksDBTransaction
    __: type[SnapshotProtocol] = RocksDBSnapshot
