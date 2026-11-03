"""Read-only snapshot for RocksDB storage."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from virtuals.tkv.storage import (
    StorageError,
)

from .context import ContextBase, ReadOperationsMixin


if TYPE_CHECKING:
    from types import TracebackType

    from .storage import RocksDBStorage


__all__ = ["RocksDBSnapshot"]


class RocksDBSnapshot(ContextBase, ReadOperationsMixin):
    """Read-only snapshot: a rocksdb DB snapshot taken when it begins.

    Every get and scan reads at that snapshot, so the view is repeatable
    and never shows half of a commit. It takes no lock and never waits on
    a writer.

    On a read-only secondary rocksdb does not support snapshots: reads
    there see the secondary's state as of its last catch-up with the
    primary, which the storage runs when the snapshot begins (throttled by
    ``secondary_refresh_interval``).
    """

    __slots__ = ("_db", "_snapshot")

    def __init__(self, storage: RocksDBStorage, db: Any, snapshot: Any) -> None:  # noqa: ANN401
        """Initialize snapshot.

        Args:
            storage: Parent storage instance
            db: rdbpy DB handle to read through
            snapshot: rdbpy Snapshot pinning the view, or None on a secondary
        """
        super().__init__(storage)
        self._db: Any | None = db
        self._snapshot: Any | None = snapshot

    @property
    def writable(self) -> bool:
        """Always False for snapshots."""
        return False

    def _drop_handles(self) -> None:
        self._db = None
        self._snapshot = None

    def _get_raw(self, encoded_key: bytes) -> bytes | None:
        """Read at the pinned snapshot."""
        return self._db.get(encoded_key, snapshot=self._snapshot)  # type: ignore[union-attr]

    def _iterator(self, *, keys_only: bool, reverse: bool) -> Any:  # noqa: ANN401
        """Cursor at the pinned snapshot."""
        return self._db.iterator(  # type: ignore[union-attr]
            snapshot=self._snapshot, keys_only=keys_only, reverse=reverse
        )

    def close(self) -> None:
        """Close snapshot: close its scans, then release the rocksdb snapshot.

        Raises:
            StorageError: If close fails
        """
        if self._closed:
            return

        self._close_scans()  # their iterators pin files past the snapshot otherwise
        snapshot = self._snapshot
        try:
            if snapshot is not None:
                snapshot.release()
        except Exception as e:
            raise StorageError(f"Failed to close snapshot: {e}") from e
        finally:
            self._mark_closed()
            self._storage._untrack_snapshot(self)

    def __enter__(self) -> RocksDBSnapshot:
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
