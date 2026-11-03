"""Read-only snapshot for LMDB storage."""

from __future__ import annotations

from typing import TYPE_CHECKING

from virtuals.tkv.storage import StorageError

from .context import ContextBase, ReadOperationsMixin


if TYPE_CHECKING:
    from types import TracebackType

    from .storage import LMDBStorage


__all__ = ["LMDBSnapshot"]


class LMDBSnapshot(ContextBase, ReadOperationsMixin):
    """Read-only snapshot backed by an LMDB read transaction.

    LMDB transactions are already MVCC-snapshotted, so a read-only
    transaction gives a consistent point-in-time view for free.
    """

    __slots__ = ()

    def __init__(self, storage: LMDBStorage, lmdb_txn: object) -> None:
        """Initialize snapshot."""
        super().__init__(storage, lmdb_txn)

    @property
    def writable(self) -> bool:
        """Always False for snapshots."""
        return False

    def close(self) -> None:
        """Close snapshot and release its reader slot.

        Safe from any thread: a close racing the owner's call waits for
        that call, then ends the read transaction. Read transactions have
        no thread affinity in LMDB, so no writer thread is involved.
        """
        with self._call_lock:
            if self._closed:
                return
            txn = self._lmdb_txn
            failure: Exception | None = None
            try:
                if txn is not None:
                    txn.abort()
            except Exception as e:
                failure = e
            finally:
                self._mark_closed()

        self._storage._untrack_snapshot(self)
        if failure is not None:
            raise StorageError(f"Failed to close snapshot: {failure}") from failure

    def __enter__(self) -> LMDBSnapshot:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager - auto close."""
        self.close()
