"""Read-write transaction for SQLite storage."""

from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING

from virtuals.tkv.storage import (
    StorageClosedError,
    StorageLockTimeoutError,
    StorageTransactionAbortedError,
    StorageTransactionError,
    TransactionProtocol,
)

from .context import ContextBase, ReadOperationsMixin, WriteOperationsMixin, is_busy_error


if TYPE_CHECKING:
    import sqlite3
    import threading
    from types import TracebackType

    from virtuals.tkv.types import Key

    from .storage import SQLiteStorage


__all__ = ["SQLiteTransaction"]


logger = getLogger(__name__)


class SQLiteTransaction(
    ContextBase, ReadOperationsMixin, WriteOperationsMixin, TransactionProtocol
):
    """Read-write transaction backed by a ``BEGIN IMMEDIATE`` SQLite transaction.

    The write lock is taken when the transaction begins, not at its first
    write, so a transaction never starts as a reader and then fails to
    upgrade. Writers are serialized: one per database file across every
    process, waiting on each other up to the storage's ``busy_timeout``.
    Reads inside see the transaction's own pending writes.
    """

    __slots__ = ("_aborted", "_committed", "_modified_keys")

    def __init__(
        self, storage: SQLiteStorage, conn: sqlite3.Connection, writer_lock: threading.Lock
    ) -> None:
        """Initialize transaction.

        Args:
            storage: Parent storage.
            conn: Pooled connection with ``BEGIN IMMEDIATE`` already run.
            writer_lock: The storage's writer lock held for this transaction.
        """
        super().__init__(storage, conn, writer_lock=writer_lock)
        self._modified_keys: set[Key] = set()
        self._committed = False
        self._aborted = False

    @property
    def writable(self) -> bool:
        """Always True for transactions."""
        return True

    def commit(self) -> None:
        """Commit all changes in the transaction."""
        if self._closed:
            logger.error("Cannot commit, transaction is closed")
            raise StorageClosedError("Transaction is closed")
        if self._committed:
            logger.error("Cannot commit, transaction already committed")
            raise StorageTransactionError("Transaction already committed")
        if self._aborted:
            logger.error("Cannot commit, transaction already aborted")
            raise StorageTransactionError("Transaction already aborted")

        self._require_active()

        try:
            if not self._end("COMMIT"):
                raise StorageClosedError("Transaction was closed while committing")
        except Exception as e:
            # The connection went back to the pool rolled back: treat as aborted.
            self._aborted = True
            self._storage._untrack_transaction(self)
            logger.error("Transaction commit failed")
            if isinstance(e, StorageClosedError):
                raise  # the storage's close ended it first: say so, as every backend does
            if is_busy_error(e):
                raise StorageLockTimeoutError(f"Timed out committing transaction: {e}") from e
            raise StorageTransactionError(f"Failed to commit transaction: {e}") from e

        logger.debug("Transaction committed")

        self._committed = True
        self._storage._notify_batch(self._modified_keys)
        self._storage._untrack_transaction(self)

    def abort(self) -> None:
        """Abort transaction and discard all changes."""
        if self._closed:
            logger.debug("Abort called on closed transaction")
            return

        # Only the call that takes the connection ends the transaction. If a
        # racing commit took it, this abort rolled nothing back and the
        # transaction keeps the commit's outcome.
        rolled_back = True
        try:
            rolled_back = self._end("ROLLBACK")
            if rolled_back:
                logger.debug("Transaction aborted")
        except Exception as e:
            # The connection went back to the pool, which rolls it back.
            logger.error("Transaction abort failed")
            raise StorageTransactionAbortedError(f"Failed to abort transaction: {e}") from e
        finally:
            if rolled_back:
                self._aborted = True
            self._storage._untrack_transaction(self)

    def __del__(self) -> None:
        """Last resort: end a transaction dropped without commit or abort.

        Without this a forgotten transaction would hold the write lock,
        in this process and across processes, until the storage closed.
        """
        if getattr(self, "_closed", True):
            return
        self._aborted = True
        try:
            self._finalize()
        except Exception:  # noqa: S110 (interpreter shutdown, half-built object)
            pass

    def __enter__(self) -> SQLiteTransaction:
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
