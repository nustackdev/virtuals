"""Read-only snapshot for RocksDB storage."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from virtuals.tkv.storage import (
    StorageClosedError,
    StorageError,
    StorageTransactionConflictError,
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
    a writer. Dropped without ``close``, it releases the rocksdb snapshot
    once collected; ``close`` releases it at once.

    On a read-only secondary rocksdb does not support snapshots: the view
    is the secondary's state as of its last catch-up with the primary,
    and the storage does not catch up while this snapshot is open, so the
    view is fixed for its life. It is lost when the storage is forced to
    catch up anyway (``force_catch_up_with_primary``, or a read that hit a
    file the primary compacted away): from then on every read here raises
    the retryable ``StorageTransactionConflictError``, and the caller
    opens a new snapshot.
    """

    __slots__ = ("_db", "_epoch", "_snapshot")

    def __init__(
        self,
        storage: RocksDBStorage,
        db: Any,  # noqa: ANN401
        snapshot: Any,  # noqa: ANN401
        *,
        epoch: int | None = None,
    ) -> None:
        """Initialize snapshot.

        Args:
            storage: Parent storage instance
            db: rdbpy DB handle to read through
            snapshot: rdbpy Snapshot pinning the view, or None on a secondary
            epoch: On a secondary, the storage's catch-up epoch this view
                belongs to; a later catch-up retires it
        """
        super().__init__(storage)
        self._db: Any | None = db
        self._snapshot: Any | None = snapshot
        self._epoch = epoch

    def _require_active(self) -> None:
        """Raise if closed, or if a secondary catch-up retired this view."""
        if self._closed:
            raise StorageClosedError("Context is closed")
        if self._epoch is not None and self._epoch != self._storage._secondary_epoch:
            raise StorageTransactionConflictError(
                "Secondary snapshot lost its view: the storage caught up with the primary "
                "since it began. Open a new snapshot and re-read"
            )

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

    def __del__(self) -> None:
        """Last resort: release the rocksdb snapshot of one dropped without ``close``.

        The object is unreachable, so nothing else can end it, and the
        storage's weak active set forgets it on its own. The storage's
        state lock is not taken here, since a collection may run on a
        thread that holds it. A storage closed since has released the
        snapshot already, so a failing release is ignored.
        """
        if getattr(self, "_closed", True):
            return
        snapshot = self._snapshot
        try:
            self._mark_closed()
            if snapshot is not None:
                snapshot.release()
        except Exception:  # noqa: S110 (interpreter shutdown, half-built object)
            pass

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
