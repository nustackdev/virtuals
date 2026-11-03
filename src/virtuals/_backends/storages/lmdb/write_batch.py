"""Write-only batch for LMDB storage."""

from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING

from virtuals.tkv.storage import (
    StorageClosedError,
    StorageTransactionError,
    WriteBatchProtocol,
)

from .context import ContextBase, WriteOperationsMixin


if TYPE_CHECKING:
    import threading
    from types import TracebackType

    from virtuals.tkv.types import Key

    from .storage import LMDBStorage


__all__ = ["LMDBWriteBatch"]


logger = getLogger(__name__)


class LMDBWriteBatch(ContextBase, WriteOperationsMixin, WriteBatchProtocol):
    """Write-only batch backed by an LMDB write transaction.

    LMDB has no distinct batch API - a write transaction is already the
    batch primitive. This class exposes it as a write-only surface
    (no read methods) matching the WriteBatchProtocol contract.
    """

    __slots__ = ("_aborted", "_modified_keys", "_written")

    def __init__(
        self, storage: LMDBStorage, lmdb_txn: object, write_lock: threading.Lock | None = None
    ) -> None:
        """Initialize write batch."""
        super().__init__(storage, lmdb_txn, holds_write_slot=True, write_lock=write_lock)
        self._modified_keys: set[Key] = set()
        self._written = False
        self._aborted = False

    def __del__(self) -> None:
        """Abort a batch dropped unended, so it cannot hold the writer lock."""
        try:
            if not self._closed:
                self._aborted = True
                self._finalize_unended_write("write batch")
        except Exception:  # noqa: S110 (interpreter shutdown, half-built object)
            pass

    @property
    def writable(self) -> bool:
        """Always True for write batches."""
        return True

    def write(self) -> None:
        """Write batch and make changes permanent.

        LMDB's commit runs on the storage's writer thread, the one that
        began the batch, so this may be called from any thread.
        """
        with self._call_lock:
            if self._closed:
                logger.error("Cannot write, batch is closed")
                raise StorageClosedError("Write batch is closed")
            if self._written:
                logger.error("Cannot write, batch already written")
                raise StorageTransactionError("Write batch already written")
            if self._aborted:
                logger.error("Cannot write, batch already aborted")
                raise StorageTransactionError("Write batch already aborted")

            txn = self._require_active()
            try:
                self._storage._end_write(txn, commit=True)
            except Exception as e:
                # LMDB frees a write transaction whose commit fails, so close
                # this batch too and give back the writer slot.
                self._aborted = True
                self._mark_closed()
                failure: Exception | None = e
            else:
                failure = None
                self._written = True
                self._mark_closed()

        if failure is not None:
            self._storage._untrack_write_batch(self)
            logger.error("Write batch write failed")
            raise StorageTransactionError(f"Failed to write batch: {failure}") from failure

        logger.info("Write batch written")
        self._storage._notify_batch(self._modified_keys)
        self._storage._untrack_write_batch(self)

    def abort(self) -> None:
        """Abort write batch and discard changes.

        Safe from any thread, including while the owner is inside a call:
        the abort waits for that call, then ends the batch on the writer
        thread. A second abort, or one racing a write, is a no-op.
        """
        if self._closed:
            # Unlocked on purpose, see LMDBTransaction.abort.
            logger.debug("Abort called on closed write batch")
            return
        with self._call_lock:
            if self._closed:
                return
            txn = self._require_active()
            failure: Exception | None = None
            try:
                self._storage._end_write(txn, commit=False)
                logger.info("Write batch aborted")
            except Exception as e:
                logger.error("Write batch abort failed")
                failure = e
            finally:
                self._aborted = True
                self._mark_closed()

        self._storage._untrack_write_batch(self)
        if failure is not None:
            raise StorageTransactionError(f"Failed to abort write batch: {failure}") from failure

    def __enter__(self) -> LMDBWriteBatch:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager - auto write or abort."""
        if exc_type is not None:
            try:
                self.abort()
            except Exception:
                logger.error("Write batch abort failed")
        else:
            if not self._written and not self._aborted:
                self.write()
