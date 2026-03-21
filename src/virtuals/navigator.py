"""Navigator - high-level entrypoint for virtuals storage.

Wraps storage + views into a convenient API. Manages storage lifecycle,
transaction/snapshot contexts, and provides path-based access.

Usage:
    nav = Navigator(storage)

    with nav.transaction() as tx:
        tx.root["users"] = {"alice": {"name": "Alice"}}

    with nav.snapshot() as snap:
        print(snap.root["users"]["alice"])
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

from virtuals._views import LazyDictView
from virtuals.loc.path_nav import navigate_view


if TYPE_CHECKING:
    from collections.abc import Iterator

    from virtuals.tkv.storage import StorageContextType
    from virtuals.view import View


__all__ = [
    "Navigator",
    "ViewScope",
]


class ViewScope:
    """A scoped view context (transaction or snapshot) with a root view.

    Provides path-based access to the storage tree.
    """

    def __init__(self, ctx: StorageContextType, root_view_cls: type[View] = LazyDictView) -> None:
        """Initialize scope with storage context and root view class."""
        self._ctx = ctx
        self._root = root_view_cls.open_root(ctx)

    @property
    def root(self) -> View:
        """The root view of this scope."""
        return self._root

    @property
    def ctx(self) -> StorageContextType:
        """The underlying storage context (transaction/snapshot)."""
        return self._ctx

    def view(self, *path: tuple[str, type[View]]) -> View:
        """Navigate to a view at the given path.

        Args:
            *path: sequence of (address, ViewType) tuples

        Returns:
            View at the target path

        Example:
            tx.view(("users", DictView), ("alice", DictView))
        """
        return navigate_view(self._root, path)


class Navigator:
    """High-level entrypoint for virtuals storage.

    Holds a storage reference and root view class.
    Used as base for NavigatorResource in composables integration.

    Args:
        storage: a virtuals storage backend (InMemoryStorage, RocksDBStorage, etc.)
        root_view: view class for the root (default: LazyDictView)
    """

    def __init__(
        self,
        storage: object,
        root_view: type[View] = LazyDictView,
    ) -> None:
        """Initialize navigator with storage and root view class."""
        self._storage = storage
        self._root_view = root_view

    @property
    def storage(self) -> object:
        """The underlying storage backend."""
        return self._storage

    @property
    def root_view(self) -> type[View]:
        """The root view class."""
        return self._root_view

    @contextmanager
    def transaction(self) -> Iterator[ViewScope]:
        """Open a read-write transaction scope."""
        with self._storage.transaction() as tx:
            yield ViewScope(tx, self._root_view)

    @contextmanager
    def snapshot(self) -> Iterator[ViewScope]:
        """Open a read-only snapshot scope."""
        with self._storage.snapshot() as snap:
            yield ViewScope(snap, self._root_view)
