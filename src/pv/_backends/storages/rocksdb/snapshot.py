"""Read-only snapshot for RocksDB storage."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tkv.tkv.storage import (
    StorageError,
)

from .context import ContextBase, ReadOperationsMixin


if TYPE_CHECKING:
    from types import TracebackType

    from .storage import RocksDBStorage


__all__ = ["RocksDBSnapshot"]


class RocksDBSnapshot(ContextBase, ReadOperationsMixin):
    """Read-only snapshot implementation.

    Provides consistent read-only view of the database at a point in time.
    Composed from base context and read operations.
    """

    __slots__ = ()

    def __init__(self, storage: RocksDBStorage, rdbpy_txn: object) -> None:
        """Initialize snapshot.

        Args:
            storage: Parent storage instance
            rdbpy_txn: RocksDB transaction handle with snapshot
        """
        super().__init__(storage, rdbpy_txn)

    @property
    def writable(self) -> bool:
        """Always False for snapshots."""
        return False

    def close(self) -> None:
        """Close snapshot and release resources.

        Raises:
            StorageError: If close fails
        """
        if self._closed:
            return

        if self._storage._is_secondary:
            # Secondary DB snapshots don't need rollback
            return
            # FIXME

        if self._rdbpy_txn is not None:
            try:
                # Rollback to release the snapshot
                self._rdbpy_txn.rollback()
            except Exception as e:
                raise StorageError(f"Failed to close snapshot: {e}") from e
            finally:
                self._mark_closed()
                self._storage._remove_snapshot(self)
        else:
            self._mark_closed()
            self._storage._remove_snapshot(self)

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
