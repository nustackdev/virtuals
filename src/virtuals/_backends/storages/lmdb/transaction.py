"""Read-write transaction for LMDB storage."""

from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING

from virtuals.tkv.storage import (
    StorageClosedError,
    StorageTransactionAbortedError,
    StorageTransactionError,
    TransactionProtocol,
)

from .context import ContextBase, ReadOperationsMixin, WriteOperationsMixin


if TYPE_CHECKING:
    import threading
    from types import TracebackType

    from virtuals.tkv.types import Key

    from .storage import LMDBStorage


__all__ = ["LMDBTransaction"]


logger = getLogger(__name__)


class LMDBTransaction(ContextBase, ReadOperationsMixin, WriteOperationsMixin, TransactionProtocol):
    """Read-write transaction backed by an LMDB write transaction.

    LMDB serializes writers at the environment level: only one write
    transaction may be active per environment at a time. In-process
    writers queue on the storage's writer slot first; the transaction
    holds it from begin and releases it on commit or abort.
    """

    __slots__ = ("_aborted", "_committed", "_modified_keys")

    def __init__(
        self, storage: LMDBStorage, lmdb_txn: object, write_lock: threading.Lock | None = None
    ) -> None:
        """Initialize transaction."""
        super().__init__(storage, lmdb_txn, holds_write_slot=True, write_lock=write_lock)
        self._modified_keys: set[Key] = set()
        self._committed = False
        self._aborted = False

    def __del__(self) -> None:
        """Abort a transaction dropped unended, so it cannot hold the writer lock."""
        try:
            if not self._closed:
                self._aborted = True
                self._finalize_unended_write("transaction")
        except Exception:  # noqa: S110 (interpreter shutdown, half-built object)
            pass

    @property
    def writable(self) -> bool:
        """Always True for transactions."""
        return True

    def commit(self) -> None:
        """Commit all changes in the transaction.

        LMDB's commit runs on the storage's writer thread, the one that
        began the transaction, so this may be called from any thread.
        """
        with self._call_lock:
            if self._closed:
                logger.error("Cannot commit, transaction is closed")
                raise StorageClosedError("Transaction is closed")
            if self._committed:
                logger.error("Cannot commit, transaction already committed")
                raise StorageTransactionError("Transaction already committed")
            if self._aborted:
                logger.error("Cannot commit, transaction already aborted")
                raise StorageTransactionError("Transaction already aborted")

            txn = self._require_active()
            try:
                self._storage._end_write(txn, commit=True)
            except Exception as e:
                # LMDB frees a write transaction whose commit fails, so close
                # this one too and give back the writer slot.
                self._aborted = True
                self._mark_closed()
                failure: Exception | None = e
            else:
                failure = None
                self._committed = True
                self._mark_closed()

        if failure is not None:
            self._storage._untrack_transaction(self)
            logger.error("Transaction commit failed")
            raise StorageTransactionError(f"Failed to commit transaction: {failure}") from failure

        logger.info("Transaction committed")
        self._storage._notify_batch(self._modified_keys)
        self._storage._untrack_transaction(self)

    def abort(self) -> None:
        """Abort transaction and discard all changes.

        Safe from any thread, including while the owner is inside a call:
        the abort waits for that call, then ends the transaction on the
        writer thread. A second abort, or one racing a commit, is a no-op.
        """
        if self._closed:
            # Unlocked on purpose: the flag only ever turns True, after the
            # LMDB transaction has ended, so this abort has nothing to do.
            # close() calls here while the owner may be inside its commit's
            # slot release; waiting for that lock would wait on close itself.
            logger.debug("Abort called on closed transaction")
            return
        with self._call_lock:
            if self._closed:
                return
            txn = self._require_active()
            failure: Exception | None = None
            try:
                self._storage._end_write(txn, commit=False)
                logger.info("Transaction aborted")
            except Exception as e:
                logger.error("Transaction abort failed")
                failure = e
            finally:
                self._aborted = True
                self._mark_closed()

        self._storage._untrack_transaction(self)
        if failure is not None:
            raise StorageTransactionAbortedError(
                f"Failed to abort transaction: {failure}"
            ) from failure

    def __enter__(self) -> LMDBTransaction:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager - auto commit or abort."""
        if exc_type is not None:
            try:
                self.abort()
            except Exception:
                logger.error("Transaction abort failed")
        else:
            if not self._committed and not self._aborted:
                self.commit()
