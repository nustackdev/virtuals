"""Read-only snapshot for the in-memory storage."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from virtuals.tkv.storage import SnapshotProtocol

from .context import ContextBase, ReadOperationsMixin


if TYPE_CHECKING:
    from types import TracebackType

    from .storage import InMemoryStorage
    from .store import Encoded, Entry, Marker


__all__ = ["InMemorySnapshot"]


class InMemorySnapshot(ContextBase, ReadOperationsMixin, SnapshotProtocol):
    """A point-in-time view, pinned at the commit sequence it was begun at.

    Every read answers as the store stood at that sequence: repeatable,
    and never half of a later commit. Takes no lock and copies nothing;
    the store keeps the versions this snapshot may still need until it
    closes.
    """

    __slots__ = ("_seq",)

    def __init__(self, storage: InMemoryStorage, seq: int) -> None:
        """Bind the snapshot to ``storage`` at pinned sequence ``seq``."""
        super().__init__(storage)
        self._seq = seq

    @property
    def writable(self) -> bool:
        """Always False for snapshots."""
        return False

    def _lookup(self, enc: Encoded) -> Entry | Marker:
        return self._storage._store.get_at(enc, self._seq)

    def _page(
        self, bound: Encoded | None, *, inclusive: bool, reverse: bool, n: int, stop: Encoded | None
    ) -> tuple[list[tuple[Any, Entry]], Any, bool]:
        return self._storage._store.page_at(
            self._seq, bound, inclusive=inclusive, reverse=reverse, n=n, stop=stop
        )

    def close(self) -> None:
        """Close the snapshot and let the store drop versions only it needed."""
        if self._closed:
            return
        self._closed = True
        self._storage._store.unpin(self._seq)
        self._storage._untrack_snapshot(self)

    def __enter__(self) -> InMemorySnapshot:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager."""
        self.close()
