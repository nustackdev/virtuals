"""RocksDB storage backend implementation.

Provides persistent key-value storage with transactions, snapshots,
and optional change notifications via an observer.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, overload

from virtuals.tkv.storage import (
    SnapshotProtocol,
    StorageClosedError,
    StorageError,
    StorageOperationError,
    TransactionProtocol,
    WriteBatchProtocol,
)

from .snapshot import RocksDBSnapshot
from .transaction import RocksDBTransaction
from .write_batch import RocksDBWriteBatch


if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import TracebackType

    from virtuals.tkv.codec import CodecProtocol
    from virtuals.tkv.observer import ObserverProtocol, Subscription, SubscriptionOptions
    from virtuals.tkv.types import Key

try:
    import rdbpy
except ImportError as e:
    raise ImportError(
        "RocksDB python bindings (rdbpy) are required for RocksDBStorage. "
        "Install via: [uv] pip install rdbpython"
    ) from e


__all__ = ["RocksDBStorage"]


logger = getLogger(__name__)


class RocksDBStorage:
    """RocksDB storage implementation conforming to StorageProtocol.

    Provides persistent key-value storage with transactions, snapshots,
    and optional change notifications via an observer.
    """

    def __init__(
        self,
        path: Path | str,
        codec: CodecProtocol[bytes, bytes],
        observer: ObserverProtocol | None = None,
        *,
        read_only: bool = False,
        secondary_path: Path | str | None = None,
        wal_path: Path | str | None = None,
        options: dict[str, Any] | None = None,
        txn_db_options: dict[str, Any] | None = None,
        txn_options: dict[str, Any] | None = None,
        create_if_missing: bool = True,
        sync_writes: bool = False,
        disable_wal: bool = False,
    ) -> None:
        """Initialize RocksDB storage.

        Args:
            path: Database directory path
            codec: Codec for key/value encoding
            observer: Optional observer for change notifications
            read_only: Open database in read-only mode
            secondary_path: Path to open db via "rocksdb::DB::OpenAsSecondary"
                (allows multiple parallel readers)
            wal_path: Optional separate WAL directory
            options: RocksDB options dict
            txn_db_options: TransactionDB options dict
            txn_options: Transaction options dict
            create_if_missing: Create database if it doesn't exist
            sync_writes: Sync writes to disk
            disable_wal: Disable write-ahead log
        """
        # Core dependencies
        self._codec = codec
        self._observer = observer

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

        # State
        self._db: rdbpy.TransactionDB | rdbpy.DB | None = None
        self._db_lock = threading.RLock()
        self._active_transactions: set[RocksDBTransaction] = set()
        self._active_write_batches: set[RocksDBWriteBatch] = set()
        self._active_snapshots: set[RocksDBSnapshot] = set()
        self._opened = False

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

        try:
            options = rdbpy.Options(**options_dict)
        except Exception as e:
            raise StorageError(f"Invalid RocksDB options: {e}") from e

        if self._wal_path is not None:
            options.wal_dir = str(self._wal_path)

        return options

    def _open_transaction_db(self, options: rdbpy.Options) -> None:
        """Open database as TransactionDB for read-write access.

        Args:
            options: RocksDB options

        Raises:
            StorageError: If database cannot be opened
        """
        txn_db_options = None
        if self._txn_db_options_dict:
            try:
                txn_db_options = rdbpy.TransactionDBOptions(**self._txn_db_options_dict)
            except Exception as e:
                raise StorageError(f"Invalid TransactionDB options: {e}") from e

        try:
            self._db = rdbpy.TransactionDB(
                str(self._path),
                options,
                txn_db_options,
            )
        except Exception as e:
            raise StorageError(f"Failed to open RocksDB TransactionDB: {e}") from e

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

        with self._db_lock:
            # Create directories
            try:
                self._path.mkdir(parents=True, exist_ok=True)
                if self._wal_path is not None:
                    self._wal_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise StorageError(f"Failed to create database directories: {e}") from e

            options = self._build_options()

            # Open in appropriate mode
            if self._is_secondary:
                self._open_secondary_db(options)
            else:
                self._open_transaction_db(options)

            self._opened = True

    def close(self) -> None:
        """Close database and release all resources.

        Raises:
            StorageError: If close fails
        """
        if not self._opened:
            return

        with self._db_lock:
            # Abort all active transactions
            for transaction in list(self._active_transactions):
                try:
                    transaction.abort()
                except Exception as e:
                    logger.error(f"Transaction abort failed during close: {e}")

            # Close all active snapshots
            for snapshot in list(self._active_snapshots):
                try:
                    snapshot.close()
                except Exception as e:
                    logger.error(f"Snapshot close failed during close: {e}")

            # Abort all active write batches
            for write_batch in list(self._active_write_batches):
                try:
                    write_batch.abort()
                except Exception as e:
                    logger.error(f"Write batch abort failed during close: {e}")

            # Clear tracking sets
            self._active_transactions.clear()
            self._active_write_batches.clear()
            self._active_snapshots.clear()

            # Close database
            if self._db is not None:
                try:
                    if not self._is_secondary:
                        self._db.close()
                except Exception as e:
                    raise StorageError(f"Failed to close database: {e}") from e
                finally:
                    self._db = None

            self._opened = False

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
    # Subscriptions
    # =========================================================================

    def subscribe(self, options: SubscriptionOptions) -> Subscription:
        """Subscribe to key changes with flexible filtering.

        Args:
            options: Subscription options including filter specification

        Returns:
            Subscription object for binding callbacks and managing lifecycle.

        Raises:
            StorageOperationError: If subscription fails or observer not configured.
        """
        if self._observer is None:
            raise StorageOperationError("Observer not configured for this storage")

        try:
            return self._observer.subscribe(options)
        except Exception as e:
            raise StorageOperationError(f"Failed to subscribe: {e}") from e

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
        """Begin read-only snapshot.

        Returns:
            New snapshot instance

        Raises:
            StorageError: If snapshot creation fails
            StorageClosedError: If storage is not open
        """
        self._require_open()

        if self._is_secondary:
            return self._begin_snapshot_on_secdb()
        else:
            return self._begin_snapshot_on_txdb()

    def _begin_snapshot_on_secdb(self) -> RocksDBSnapshot:
        """Begin read-only snapshot on a secondary db instance.

        Returns:
            New snapshot instance

        Raises:
            StorageError: If snapshot creation fails
        """
        self._require_open()

        if not self._is_secondary:
            raise StorageError("Invalid snapshot creation")

        with self._db_lock:
            try:
                self._db.try_catch_up_with_primary()
            except Exception as e:
                raise StorageError(f"Failed to catch up with primary: {e}") from e

            try:
                snapshot = RocksDBSnapshot(self, self._db)
                return snapshot
            except Exception as e:
                raise StorageError(f"Failed to begin snapshot: {e}") from e

    def _begin_snapshot_on_txdb(self) -> RocksDBSnapshot:
        """Begin read-only snapshot on TransactionDB instance.

        Returns:
            New snapshot instance

        Raises:
            StorageError: If snapshot creation fails
        """
        self._require_open()

        if self._is_secondary:
            raise StorageError("Invalid snapshot creation")

        with self._db_lock:
            # Create transaction options with snapshot
            txn_options_dict = dict(self._txn_options_dict)
            txn_options_dict["set_snapshot"] = True

            try:
                txn_options = rdbpy.TransactionOptions(**txn_options_dict)
            except Exception as e:
                raise StorageError(f"Invalid snapshot options: {e}") from e

            # Begin transaction with snapshot
            try:
                rdbpy_txn = self._db.begin_transaction(txn_options)
                rdbpy_txn.set_snapshot()
            except Exception as e:
                raise StorageError(f"Failed to begin snapshot: {e}") from e

            snapshot = RocksDBSnapshot(self, rdbpy_txn)
            self._active_snapshots.add(snapshot)
            return snapshot

    def begin_transaction(self) -> RocksDBTransaction:
        """Begin read-write transaction.

        Returns:
            New transaction instance

        Raises:
            StorageError: If transaction creation fails
            StorageClosedError: If storage is not open
        """
        if self._read_only:
            raise StorageError("Cannot start transaction in read only mode.")

        self._require_open()

        with self._db_lock:
            # Create transaction options
            txn_options = None
            if self._txn_options_dict:
                try:
                    txn_options = rdbpy.TransactionOptions(**self._txn_options_dict)
                except Exception as e:
                    raise StorageError(f"Invalid transaction options: {e}") from e

            # Write options (e.g. disable WAL for bulk writes)
            write_options = None
            if self._disable_wal:
                write_options = {"disable_wal": True}

            # Begin transaction
            try:
                if txn_options is not None:
                    rdbpy_txn = self._db.begin_transaction(txn_options, write_options=write_options)
                else:
                    rdbpy_txn = self._db.begin_transaction(write_options=write_options)
            except Exception as e:
                raise StorageError(f"Failed to begin transaction: {e}") from e

            transaction = RocksDBTransaction(self, rdbpy_txn)
            self._active_transactions.add(transaction)
            return transaction

    def begin_write_batch(self) -> RocksDBWriteBatch:
        """Begin write-only batch.

        Returns:
            New write batch instance

        Raises:
            StorageOperationError: If batch creation fails
            StorageClosedError: If storage is closed
        """
        if self._read_only:
            raise StorageError("Cannot start write batch in read only mode.")

        self._require_open()

        with self._db_lock:
            rdbpy_batch = rdbpy.WriteBatch()
            write_batch = RocksDBWriteBatch(self, rdbpy_batch)
            self._active_write_batches.add(write_batch)
            return write_batch

    @contextmanager
    def transaction(self) -> Iterator[RocksDBTransaction]:
        """Context manager for a read-write transaction.

        Commits on successful exit, aborts on exception.
        """
        txn = self.begin_transaction()
        try:
            yield txn
            if not txn._committed and not txn._aborted:
                txn.commit()
        except Exception:
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

        Writes on successful exit, aborts on exception.
        """
        batch = self.begin_write_batch()
        try:
            yield batch
            if not batch._written and not batch._aborted:
                batch.write()
        except Exception:
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
        """Notify observer of key changes (batch).

        Args:
            keys: Keys that changed
        """
        if self._observer is not None and keys:
            try:
                self._observer.notify(keys)
            except Exception as e:
                logger.error(f"Observer notification failed: {e}")

    def _remove_transaction(self, transaction: RocksDBTransaction) -> None:
        """Remove transaction from active set.

        Args:
            transaction: Transaction to remove
        """
        with self._db_lock:
            self._active_transactions.discard(transaction)

    def _remove_snapshot(self, snapshot: RocksDBSnapshot) -> None:
        """Remove snapshot from active set.

        Args:
            snapshot: Snapshot to remove
        """
        with self._db_lock:
            self._active_snapshots.discard(snapshot)

    def _remove_write_batch(self, write_batch: RocksDBWriteBatch) -> None:
        """Remove write batch from active set.

        Args:
            write_batch: Write batch to remove
        """
        with self._db_lock:
            self._active_write_batches.discard(write_batch)

    def _require_open(self) -> None:
        """Validate storage is open.

        Raises:
            StorageClosedError: If storage is not open
        """
        if not self._opened or self._db is None:
            raise StorageClosedError("Storage is not open")


if TYPE_CHECKING:
    _: type[TransactionProtocol] = RocksDBTransaction
    __: type[SnapshotProtocol] = RocksDBSnapshot
