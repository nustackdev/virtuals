"""Read-only snapshot for text storage."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from virtuals.tkv.storage import SnapshotProtocol

from .context import ContextBase, ReadOperationsMixin


if TYPE_CHECKING:
    from types import TracebackType

    from .storage import TextStorage


__all__ = ["TextSnapshot"]


class TextSnapshot(ContextBase, ReadOperationsMixin, SnapshotProtocol):
    """Read-only snapshot for text storage.

    Provides point-in-time view of storage state.
    """

    __slots__ = ()

    def __init__(self, storage: TextStorage, state: dict[str, Any]) -> None:
        """Initialize snapshot with copied state.

        Args:
            storage: Parent storage instance
            state: Snapshot of state (copied, not shared)
        """
        super().__init__(storage, state)

    @property
    def writable(self) -> bool:
        """Check if snapshot is writable.

        Returns:
            Always False for snapshots.
        """
        return False

    def close(self) -> None:
        """Close snapshot and release resources."""
        if not self._closed:
            self._mark_closed()
            self._storage._untrack_snapshot(self)

    def __enter__(self) -> TextSnapshot:
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
