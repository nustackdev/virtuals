"""Navigator - high-level entrypoint for virtuals storage.

Central coordinator for storage access. Owns the ViewRegistry, creates
root views, provides path-based access. All views flow from Navigator.

Usage:
    nav = Navigator(InMemoryStorage(codec=NoOpCodec()))

    with nav.storage as storage:
        with storage.transaction() as tx:
            root = nav.root(tx)
            root["users"] = {"alice": {"name": "Alice"}}

        with storage.snapshot() as snap:
            root = nav.root(snap)
            print(root["users"]["alice"])
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar

from virtuals.container import Container
from virtuals.loc import DATA_ROOT, path

from .view.registry import ViewRegistry


if TYPE_CHECKING:
    from virtuals.loc import site as site_
    from virtuals.tkv.storage import StorageContextType, StorageProtocol
    from virtuals.view import View


__all__ = [
    "Navigator",
]

ViewT = TypeVar("ViewT", bound="View")


def _default_views() -> tuple[type[View], ...]:
    """Standard views registered by default."""
    from virtuals._views.bytearray_view import ByteArrayView
    from virtuals._views.dict_view import EagerDictView
    from virtuals._views.frozenset_view import FrozenSetView
    from virtuals._views.list_view import EagerListView
    from virtuals._views.set_view import SetView
    from virtuals._views.tuple_view import TupleView

    return (ByteArrayView, EagerDictView, FrozenSetView, EagerListView, SetView, TupleView)


class Navigator(Generic[ViewT]):
    """High-level entrypoint for virtuals storage.

    Owns the ViewRegistry and provides view creation methods.
    All views should be created through Navigator.

    Args:
        storage: a virtuals storage backend (InMemoryStorage, RocksDBStorage, etc.)
        root_view: view class for the root (default: EagerDictView)
        views: tuple of view classes to register. None = standard views.
        site: root site for this navigator (default: DATA_ROOT)
    """

    def __init__(
        self,
        storage: StorageProtocol,
        root_view: type[ViewT] | None = None,
        views: tuple[type[View], ...] | None = None,
        site: site_.Site | None = None,
    ) -> None:
        """Initialize Navigator with storage, root view class, and optional views."""
        if root_view is None:
            from virtuals._views.dict_view import EagerDictView

            root_view = EagerDictView  # type: ignore[assignment]

        self._storage = storage
        self._root_view_cls = root_view
        self._site: site_.Site = site if site is not None else (DATA_ROOT,)
        self._registry = self._build_registry(views)

    @property
    def storage(self) -> StorageProtocol:
        """The underlying storage backend."""
        return self._storage

    @property
    def registry(self) -> ViewRegistry:
        """The view registry."""
        return self._registry

    @property
    def site(self) -> site_.Site:
        """The root site for this navigator."""
        return self._site

    def root(self, ctx: StorageContextType) -> ViewT:
        """Open root view on a storage context (transaction/snapshot).

        This is the primary entry point for accessing storage.

        Args:
            ctx: storage context (transaction, snapshot, or write batch)

        Returns:
            Root view with navigator's registry attached.
        """
        if self._root_view_cls is None:
            raise ValueError("Root view class is not specified")

        container = Container(ctx=ctx, site=self._site)
        return self._root_view_cls(container, self._registry)

    def open_at_path(self, view_path: path.PathToView, ctx: StorageContextType) -> View:
        """Navigate to a view given a path.

        Creates the root view and runs the full navigation server-side.
        Path is a tuple of (address, view_type) segments — pure immutable data.

        Pure navigation — safe on read-only storage contexts (snapshots,
        RO secondaries). Does NOT create any containers along the way. Use
        ``open_at_path_and_ensure`` when the caller is a writer and needs
        each level materialized with its declared view type.

        Args:
            view_path: Path tuple, e.g. (("users", DictView), ("alice", DictView))
            ctx: storage context (transaction, snapshot, or write batch)

        Returns:
            View at the end of the path.
        """
        from virtuals.loc.path_nav import navigate_view

        root = self.root(ctx)
        return navigate_view(root, view_path)

    def open_at_path_and_ensure(
        self,
        view_path: path.PathToView,
        ctx: StorageContextType,
    ) -> View:
        """Write-side sibling of ``open_at_path``.

        Walks the path and calls ``ensure_created()`` at each level, so every
        intermediate container is stamped with its declared view type's
        marker (and its ``_ensure_internal_layout`` hook runs). This is the
        entry point ref-write code uses so that custom-layout views along a
        path (``LogIndexedDictView``, ``IndexedDictView``, etc.) never get
        auto-created with the default marker by the container layer's
        view-blind parent-fill.

        Fast path: single existence probe on the deepest site. If already
        present, skips the walk (invariant: prior walks through this helper
        stamped every ancestor correctly).

        Requires a write-capable context.

        Args:
            view_path: Path tuple.
            ctx: Write-capable storage context.

        Returns:
            View at the end of the path, guaranteed materialized.
        """
        from virtuals.loc.path_nav import navigate_and_ensure

        root = self.root(ctx)
        return navigate_and_ensure(root, view_path)

    def root_at(self, site: site_.Site, ctx: StorageContextType) -> ViewT:
        """Open a view at a specific site.

        Args:
            site: target site tuple
            ctx: storage context

        Returns:
            View at the given site with navigator's registry.
        """
        if self._root_view_cls is None:
            raise ValueError("Root view class is not specified")

        container = Container(ctx=ctx, site=site)
        return self._root_view_cls(container, self._registry)

    def _build_registry(self, views: tuple[type[View], ...] | None = None) -> ViewRegistry:
        """Build registry from provided views or defaults."""
        registry = ViewRegistry()
        for view in views or _default_views():
            registry.register(view)
        return registry
