"""Read-write transaction for LMDB storage."""

from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING, Any

import lmdb

from virtuals.tkv.storage import (
    StorageClosedError,
    StorageError,
    StorageLockTimeoutError,
    StorageTransactionAbortedError,
    StorageTransactionConflictError,
    StorageTransactionError,
    TransactionProtocol,
)

from ._writer import NOT_RUN
from .context import ContextBase, ReadOperationsMixin, WriteOperationsMixin


if TYPE_CHECKING:
    import threading
    from types import TracebackType

    from virtuals.tkv.types import Key

    from .context import Pending
    from .storage import LMDBStorage


__all__ = ["LMDBTransaction"]


logger = getLogger(__name__)


class LMDBTransaction(ContextBase, ReadOperationsMixin, WriteOperationsMixin, TransactionProtocol):
    """Read-write transaction backed by an LMDB write transaction.

    LMDB serializes writers at the environment level: only one write
    transaction may be active per environment at a time. In-process
    writers queue on the storage's writer slot first; the transaction
    holds it, and the cross-process gate, from begin and releases both on
    commit or abort.

    The LMDB transaction itself is begun lazily. Holding the slot and the
    gate is what fixes the view (no peer commits until this one ends), so
    puts and deletes collect in Python until the first read, which begins
    the LMDB transaction on the writer thread and replays them; from then
    on every call goes to the handle. A transaction that never reads is
    begun, replayed and committed in one writer-thread job at ``commit``,
    and one that never reads nor commits is aborted without touching LMDB.
    A transaction from ``areserve_write_slot`` starts begun.
    """

    __slots__ = ("_aborted", "_committed", "_modified_keys", "_pending")

    def __init__(
        self, storage: LMDBStorage, lmdb_txn: object, write_lock: threading.Lock | None = None
    ) -> None:
        """Initialize transaction, pending if ``lmdb_txn`` is None."""
        super().__init__(storage, lmdb_txn, holds_write_slot=True, write_lock=write_lock)
        self._pending: Pending | None = {} if lmdb_txn is None else None
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

    def _require_active(self) -> Any:  # noqa: ANN401
        """The LMDB handle, begun now if the transaction is still pending.

        Runs under ``_call_lock`` (every caller holds it). The begin waits
        on the writer thread, bounded; if it fails the transaction stays
        pending and the caller may retry or abort.
        """
        if self._closed:
            raise StorageClosedError("Context is closed")
        txn = self._lmdb_txn
        if txn is None:
            pending = self._pending
            if pending is None:
                raise StorageClosedError("Context handle is invalid")
            storage = self._storage
            if not pending:  # read first: just the begin, nothing to replay
                txn = storage._begin_on_writer()
            else:
                txn, missing = storage._begin_pending(pending)
                for encoded_key in missing:
                    self._modified_keys.discard(pending[encoded_key][0])
            self._lmdb_txn = txn
            self._pending = None
        return txn

    def commit(self) -> None:
        """Commit all changes in the transaction.

        LMDB's commit runs on the storage's writer thread, the one that
        began the transaction, so this may be called from any thread. A
        transaction that never read is begun, replayed and committed in
        one job there.

        An interrupt (a signal handler raising, ``KeyboardInterrupt``)
        while the commit is in flight never changes its outcome or how it
        is reported: the commit is waited out, the transaction ends
        committed, notified, slot and gate given back, and the interrupt
        is then raised as it is. One that arrives before the commit
        started (a write-only transaction whose begin had not landed)
        commits nothing: the transaction is aborted and the interrupt
        raised as it is. An interrupt in the few bytecodes right after
        the commit can cost the commit its notification; the data is
        committed either way.
        """
        deferred: list[BaseException] = []
        # ``deferred`` collects the interrupts that land in the commit's
        # wait; the commit is reported first, then the first of them is
        # raised as it is.
        with self._call_lock:
            if self._closed:
                logger.error("Cannot commit, transaction is closed")
                raise StorageClosedError("Transaction is closed")

            pending = self._pending
            try:
                if pending is not None:
                    missing = self._storage._commit_pending(pending, deferred)
                else:
                    txn = self._require_active()
                    self._storage._end_write(txn, commit=True, deferred=deferred)
                    missing = ()
            except Exception as e:
                # LMDB frees a write transaction whose commit fails, so close
                # this one too and give back the writer slot.
                self._aborted = True
                self._pending = None
                self._mark_closed()
                self._storage._untrack_transaction(self)
                logger.error("Transaction commit failed")
                if deferred:
                    raise deferred[0] from e  # the interrupt, with the failure as its cause
                if isinstance(e, (StorageLockTimeoutError, StorageTransactionConflictError)):
                    raise  # retryable as it is: the whole transaction is to be re-run
                if not isinstance(e, (lmdb.Error, StorageError)):
                    raise  # an interrupt raised as a plain exception: as it is
                raise StorageTransactionError(f"Failed to commit transaction: {e}") from e
            if missing is NOT_RUN:
                # Nothing ran: aborted, as on sqlite, and the interrupt as it is.
                self._pending = None
                self._aborted = True
                self._mark_closed()
                self._storage._untrack_transaction(self)
                raise deferred[0]
            for encoded_key in missing:
                self._modified_keys.discard(pending[encoded_key][0])  # type: ignore[index]
            self._pending = None
            self._committed = True
            self._mark_closed()
            logger.debug("Transaction committed")
            try:
                self._storage._notify_batch(self._modified_keys)
            finally:
                self._storage._untrack_transaction(self)
            if deferred:
                raise deferred[0]

    def abort(self) -> None:
        """Abort transaction and discard all changes.

        Safe from any thread, including while the owner is inside a call:
        the abort waits for that call, then ends the transaction on the
        writer thread (a pending one touches no LMDB state at all). A
        second abort, or one racing a commit, is a no-op.
        """
        if self._closed:
            # Unlocked on purpose. The flag only ever turns True, and once it
            # is set the LMDB transaction has ended, or its end is queued
            # on the writer thread (a finalizer), and close() ends every
            # live write itself before closing the environment. close()
            # calls here while the owner may be inside its commit's slot
            # release; waiting for that lock would wait on close itself.
            logger.debug("Abort called on closed transaction")
            return
        with self._call_lock:
            if self._closed:
                return
            failure: Exception | None = None
            try:
                if self._pending is None:
                    txn = self._require_active()
                    self._storage._end_write(txn, commit=False)
                logger.debug("Transaction aborted")
            except Exception as e:
                logger.error("Transaction abort failed")
                failure = e
            finally:
                self._pending = None
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
