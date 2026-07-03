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

    def __init__(self, storage: LMDBStorage, lmdb_txn: object) -> None:
        """Initialize write batch."""
        super().__init__(storage, lmdb_txn)
        self._modified_keys: set[Key] = set()
        self._written = False
        self._aborted = False

    @property
    def writable(self) -> bool:
        """Always True for write batches."""
        return True

    def write(self) -> None:
        """Write batch and make changes permanent."""
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
            txn.commit()
        except Exception as e:
            logger.error("Write batch write failed")
            raise StorageTransactionError(f"Failed to write batch: {e}") from e

        logger.info("Write batch written")

        self._written = True
        self._mark_closed()

        self._storage._notify_batch(self._modified_keys)
        self._storage._untrack_write_batch(self)

    def abort(self) -> None:
        """Abort write batch and discard changes."""
        if self._closed:
            logger.debug("Abort called on closed write batch")
            return

        txn = self._require_active()

        try:
            txn.abort()
            logger.info("Write batch aborted")
        except Exception as e:
            logger.error("Write batch abort failed")
            raise StorageTransactionError(f"Failed to abort write batch: {e}") from e
        finally:
            self._aborted = True
            self._mark_closed()
            self._storage._untrack_write_batch(self)

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
