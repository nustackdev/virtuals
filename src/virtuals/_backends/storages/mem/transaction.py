"""Read-write transaction for the in-memory storage."""

from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING, Any

from virtuals.tkv.storage import (
    StorageClosedError,
    StorageTransactionError,
    TransactionProtocol,
)

from .context import ContextBase, ReadOperationsMixin, WriteOperationsMixin
from .sorted_keys import SortedKeys
from .store import ABSENT, DELETED, merge_pages


if TYPE_CHECKING:
    import threading
    from types import TracebackType

    from virtuals.tkv.types import Key

    from .storage import InMemoryStorage
    from .store import Encoded, Entry, Marker


__all__ = ["InMemoryTransaction"]


logger = getLogger(__name__)


class InMemoryTransaction(
    ContextBase, ReadOperationsMixin, WriteOperationsMixin, TransactionProtocol
):
    """A serializable read-write transaction.

    Holds the storage's writer slot from begin to commit or abort, so
    writers run one after another and nothing commits underneath it. Reads
    see committed state plus the transaction's own pending writes, point
    reads and scans alike. Commit applies the write set atomically.

    ``_pending_index`` orders the pending keys for scans; it is built on
    the first scan and kept in step from then on, so transactions that
    never scan pay nothing for it.
    """

    __slots__ = ("_aborted", "_committed", "_modified_keys", "_pending", "_pending_index", "_slot")

    def __init__(self, storage: InMemoryStorage, slot: threading.Lock) -> None:
        """Begin over an already acquired writer slot ``slot``."""
        super().__init__(storage)
        self._slot = slot
        self._pending: dict[Encoded, Entry] = {}
        self._pending_index: SortedKeys[Encoded] | None = None
        self._modified_keys: set[Key] = set()
        self._committed = False
        self._aborted = False

    @property
    def writable(self) -> bool:
        """Always True for transactions."""
        return True

    def _lookup(self, enc: Encoded) -> Entry | Marker:
        entry = self._pending.get(enc)
        if entry is None:
            return self._storage._store.get(enc)
        return ABSENT if entry[1] is DELETED else entry

    def _page(
        self, bound: Encoded | None, *, inclusive: bool, reverse: bool, n: int, stop: Encoded | None
    ) -> tuple[list[tuple[Any, Entry]], Any, bool]:
        base = self._storage._store.page(
            bound, inclusive=inclusive, reverse=reverse, n=n, stop=stop
        )
        if not self._pending:
            return base, (base[-1][0] if len(base) == n else None), len(base) < n
        index = self._pending_index
        if index is None:
            index = self._pending_index = SortedKeys(self._pending)
        pending = self._pending
        overlay: list[tuple[Any, Any]] = []
        for enc in index.page(bound, inclusive=inclusive, reverse=reverse, n=n, stop=stop):
            entry = pending[enc]
            overlay.append((enc, ABSENT if entry[1] is DELETED else entry))
        return merge_pages(base, overlay, n, reverse)

    def _stage(self, enc: Encoded, key: Key, value: object) -> None:
        pending = self._pending
        if enc not in pending and self._pending_index is not None:
            self._pending_index.add(enc)
        pending[enc] = (key, value)

    def _end(self) -> None:
        """Give the writer slot back and drop out of the active set. Runs once."""
        self._storage._release_write_slot(self._slot)
        self._storage._untrack_transaction(self)

    def commit(self) -> None:
        """Apply the write set atomically, release the slot and notify.

        The write set lands as a whole or not at all, and the slot is
        released on every outcome, including KeyboardInterrupt or
        SystemExit raised while applying.
        """
        if not self._claim_end():
            logger.error("Cannot commit, transaction is closed")
            raise StorageClosedError("Transaction is closed")

        modified = self._modified_keys
        try:
            if self._pending:
                self._storage._store.apply(self._pending, modified)
        except BaseException as e:
            self._aborted = True
            self._end()
            logger.error("Transaction commit failed")
            if isinstance(e, Exception):
                raise StorageTransactionError(f"Failed to commit transaction: {e}") from e
            raise

        self._committed = True
        self._end()
        logger.debug("Transaction committed")
        self._storage._notify_batch(modified)

    def abort(self) -> None:
        """Discard the write set and release the slot. No-op once closed."""
        if not self._claim_end():
            logger.debug("Abort called on closed transaction")
            return
        self._aborted = True
        self._pending.clear()
        self._end()
        logger.debug("Transaction aborted")

    def __enter__(self) -> InMemoryTransaction:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Commit on a clean exit, abort on an exception."""
        if exc_type is not None:
            self.abort()
        elif not self._closed:
            self.commit()
