"""LMDB storage backend implementation.

Provides persistent key-value storage over LMDB with transactions,
MVCC snapshots, and optional change notifications via a publisher.
"""

from __future__ import annotations

import os
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

from .._write_slot import WriterLock, WriteSlot
from .snapshot import LMDBSnapshot
from .transaction import LMDBTransaction
from .write_batch import LMDBWriteBatch


if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import TracebackType

    from virtuals.tkv.codec import CodecProtocol
    from virtuals.tkv.publisher import PublisherProtocol
    from virtuals.tkv.types import Key


try:
    import lmdb
except ImportError as e:
    raise ImportError(
        "LMDB python bindings are required for LMDBStorage. "
        "Install via: [uv] pip install lmdb"
    ) from e


__all__ = ["LMDBStorage", "LMDBWriteSlot"]


logger = getLogger(__name__)


DEFAULT_MAP_SIZE = 10 * 1024 * 1024 * 1024  # 10 GiB
DEFAULT_MAX_READERS = 126
DEFAULT_BUSY_TIMEOUT = 60.0  # seconds

LMDBWriteSlot = WriteSlot
"""Returned by ``LMDBStorage.areserve_write_slot``; see ``WriteSlot``."""


class LMDBStorage:
    """LMDB storage implementation conforming to StorageProtocol.

    Provides persistent key-value storage with MVCC snapshots and
    single-writer serialized transactions.

    LMDB semantics worth knowing:
      - Only one write transaction may be active per environment at a
        time. Writers in other processes block inside LMDB.
      - Read transactions are MVCC snapshots and never block writers.
      - `map_size` sets the maximum on-disk size; the mmap is grown to
        `map_size` even on empty databases.

    Writers in this process (transactions and write batches) first take an
    in-process writer slot, the same one SQLite uses, so threads queue on
    it for up to ``busy_timeout`` and then raise ``StorageLockTimeoutError``
    instead of LMDB refusing a second writer. Snapshots never touch the
    slot, so a waiting writer never delays a reader.

    Coroutines: the slot belongs to a thread and, on an event loop, to the
    task that took it. Coroutines sharing one loop wait for it with
    ``areserve_write_slot``, which polls without blocking the loop, and
    hand it to ``begin_transaction(write_slot=...)``. A second writer from
    the task or thread that already holds the slot raises
    ``StorageNestedTransactionError``; a sync begin while another task on
    the same loop holds it raises ``StorageLockTimeoutError`` at once.
    An LMDB write transaction must finish on the thread that began it.

    Multiple processes: open the storage in each process (e.g. with the
    ``spawn`` start method); that is the supported path. Reusing an open
    storage across ``os.fork`` is best effort. If no write was in progress
    at the fork, the child keeps using it from any thread, but its readers
    stay registered under the parent's pid, so a reader check elsewhere
    can misjudge them. If the parent held a write transaction or the
    writer slot at the fork, LMDB's writer state in the child is unusable
    and py-lmdb cannot reopen the same files there, so every write in the
    child raises a non-retryable ``StorageError`` saying so.
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
            lock: If True, use LMDB's file locking.
            create_if_missing: Create database if it doesn't exist.
            env_options: Extra kwargs passed through to `lmdb.open()`.
            busy_timeout: Seconds an in-process writer waits for the
                writer slot before ``StorageLockTimeoutError``.
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

        self._env: lmdb.Environment | None = None
        self._env_lock = threading.RLock()
        self._active_transactions: set[LMDBTransaction] = set()
        self._active_write_batches: set[LMDBWriteBatch] = set()
        self._active_snapshots: set[LMDBSnapshot] = set()
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

            self._pid = os.getpid()
            self._forked_during_write = False
            self._opened = True

    def _check_fork(self) -> None:
        """Drop writer state inherited across a fork.

        A writer slot held by a parent thread would never be released in
        the child. If this process is not the one that opened the storage,
        the slot is reset and the parent's open contexts are forgotten, not
        ended: aborting them here would write to the lock file the parent
        still uses (its reader slots and writer mutex). py-lmdb frees
        nothing for objects inherited across a fork, and it refuses a
        second open of the same files in one process, so the child keeps
        the inherited environment.

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
            )
            self._active_transactions = set()
            self._active_write_batches = set()
            self._active_snapshots = set()
            self._writer.reset()
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
        """Close database and release all resources."""
        if not self._opened:
            return
        self._check_fork()  # never end contexts inherited from the parent

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

            if self._env is not None and self._forked_during_write:
                # Closing an environment inherited mid-write can hang in
                # LMDB; drop it instead (py-lmdb frees nothing inherited).
                self._env = None
            elif self._env is not None:
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
            try:
                lmdb_txn = self._env.begin(write=False)
            except Exception as e:
                raise StorageError(f"Failed to begin snapshot: {e}") from e

            snapshot = LMDBSnapshot(self, lmdb_txn)
            self._active_snapshots.add(snapshot)
            return snapshot

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

        lmdb_txn = self._begin_write(write_slot, StorageError, "transaction")
        transaction = LMDBTransaction(self, lmdb_txn)
        with self._env_lock:
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
        self._check_fork()
        self._require_writable_after_fork()
        lock = await self._writer.areserve(self._busy_timeout)
        return LMDBWriteSlot(self, lock)

    def begin_write_batch(self) -> LMDBWriteBatch:
        """Begin write-only batch, taking the writer slot now."""
        if self._read_only:
            raise StorageError("Cannot start write batch in read only mode.")

        self._require_open()
        self._check_fork()
        self._require_writable_after_fork()

        lmdb_txn = self._begin_write(None, StorageOperationError, "write batch")
        write_batch = LMDBWriteBatch(self, lmdb_txn)
        with self._env_lock:
            self._active_write_batches.add(write_batch)
        return write_batch

    def _begin_write(
        self, write_slot: WriteSlot | None, error: type[StorageError], what: str
    ) -> Any:  # noqa: ANN401
        """Take the writer slot, then begin an LMDB write transaction.

        ``env.begin(write=True)`` runs outside ``_env_lock``: it can still
        block on a writer in another process, and snapshots must not wait
        behind it. On failure the slot is released, or returned to
        ``write_slot``'s holder.
        """
        give_back_slot = self._writer.take(self, write_slot, self._busy_timeout)
        try:
            env = self._env
            if env is None:
                raise StorageClosedError("Storage is not open")
            return env.begin(write=True)
        except StorageClosedError:
            give_back_slot()
            raise
        except Exception as e:
            give_back_slot()
            raise error(f"Failed to begin {what}: {e}") from e

    def _release_write_slot(self, lock: Any = None) -> None:  # noqa: ANN401
        """Release the in-process writer slot."""
        self._writer.release(lock)

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
