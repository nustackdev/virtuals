"""LMDB storage backend implementation.

Provides persistent key-value storage over LMDB with transactions,
MVCC snapshots, and optional change notifications via an observer.
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

from .snapshot import LMDBSnapshot
from .transaction import LMDBTransaction
from .write_batch import LMDBWriteBatch


if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import TracebackType

    from virtuals.tkv.codec import CodecProtocol
    from virtuals.tkv.observer import ObserverProtocol, Subscription, SubscriptionOptions
    from virtuals.tkv.types import Key


try:
    import lmdb
except ImportError as e:
    raise ImportError(
        "LMDB python bindings are required for LMDBStorage. "
        "Install via: [uv] pip install lmdb"
    ) from e


__all__ = ["LMDBStorage"]


logger = getLogger(__name__)


DEFAULT_MAP_SIZE = 10 * 1024 * 1024 * 1024  # 10 GiB
DEFAULT_MAX_READERS = 126


class LMDBStorage:
    """LMDB storage implementation conforming to StorageProtocol.

    Provides persistent key-value storage with MVCC snapshots and
    single-writer serialized transactions.

    LMDB semantics worth knowing:
      - Only one write transaction may be active per environment at a
        time; concurrent writers block inside LMDB. This adapter does
        not add its own locking on top.
      - Read transactions are MVCC snapshots and never block writers.
      - `map_size` sets the maximum on-disk size; the mmap is grown to
        `map_size` even on empty databases.
    """

    def __init__(
        self,
        path: Path | str,
        codec: CodecProtocol[bytes, bytes],
        observer: ObserverProtocol | None = None,
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
    ) -> None:
        """Initialize LMDB storage.

        Args:
            path: Database directory (subdir=True) or file (subdir=False).
            codec: Codec for key/value encoding.
            observer: Optional observer for change notifications.
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
            lock: If True, use LMDB's file locking.
            create_if_missing: Create database if it doesn't exist.
            env_options: Extra kwargs passed through to `lmdb.open()`.
        """
        self._codec = codec
        self._observer = observer

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

        self._env: lmdb.Environment | None = None
        self._env_lock = threading.RLock()
        self._active_transactions: set[LMDBTransaction] = set()
        self._active_write_batches: set[LMDBWriteBatch] = set()
        self._active_snapshots: set[LMDBSnapshot] = set()
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

            self._opened = True

    def close(self) -> None:
        """Close database and release all resources."""
        if not self._opened:
            return

        with self._env_lock:
            for transaction in list(self._active_transactions):
                try:
                    transaction.abort()
                except Exception as e:
                    logger.error(f"Transaction abort failed during close: {e}")

            for snapshot in list(self._active_snapshots):
                try:
                    snapshot.close()
                except Exception as e:
                    logger.error(f"Snapshot close failed during close: {e}")

            for write_batch in list(self._active_write_batches):
                try:
                    write_batch.abort()
                except Exception as e:
                    logger.error(f"Write batch abort failed during close: {e}")

            self._active_transactions.clear()
            self._active_write_batches.clear()
            self._active_snapshots.clear()

            if self._env is not None:
                try:
                    self._env.close()
                except Exception as e:
                    raise StorageError(f"Failed to close database: {e}") from e
                finally:
                    self._env = None

            self._opened = False

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
    # Subscriptions
    # =========================================================================

    def subscribe(self, options: SubscriptionOptions) -> Subscription:
        """Subscribe to key changes with flexible filtering."""
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

        with self._env_lock:
            try:
                lmdb_txn = self._env.begin(write=False)
            except Exception as e:
                raise StorageError(f"Failed to begin snapshot: {e}") from e

            snapshot = LMDBSnapshot(self, lmdb_txn)
            self._active_snapshots.add(snapshot)
            return snapshot

    def begin_transaction(self) -> LMDBTransaction:
        """Begin read-write transaction."""
        if self._read_only:
            raise StorageError("Cannot start transaction in read only mode.")

        self._require_open()

        with self._env_lock:
            try:
                lmdb_txn = self._env.begin(write=True)
            except Exception as e:
                raise StorageError(f"Failed to begin transaction: {e}") from e

            transaction = LMDBTransaction(self, lmdb_txn)
            self._active_transactions.add(transaction)
            return transaction

    def begin_write_batch(self) -> LMDBWriteBatch:
        """Begin write-only batch."""
        if self._read_only:
            raise StorageError("Cannot start write batch in read only mode.")

        self._require_open()

        with self._env_lock:
            try:
                lmdb_txn = self._env.begin(write=True)
            except Exception as e:
                raise StorageOperationError(f"Failed to begin write batch: {e}") from e

            write_batch = LMDBWriteBatch(self, lmdb_txn)
            self._active_write_batches.add(write_batch)
            return write_batch

    @contextmanager
    def transaction(self) -> Iterator[LMDBTransaction]:
        """Context manager for a read-write transaction."""
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
        """Context manager for a write batch."""
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

        Fire-and-forget: writer enqueues and returns. Callers that need a
        delivery barrier call observer.flush() explicitly.
        """
        if self._observer is not None and keys:
            try:
                self._observer.notify(keys)
            except Exception as e:
                logger.error(f"Observer notification failed: {e}")

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
