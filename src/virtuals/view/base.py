"""Base View implementation for Layer 3.

Views are thin wrappers over Container providing protocol-based capabilities.
Views are created through Navigator, not directly.
"""

from __future__ import annotations

from abc import ABC
from logging import getLogger
from typing import TYPE_CHECKING, ClassVar

import attrs

from virtuals.container import (
    Container,
    ContainerProtocol,
    ContainerStructure,
    ContainerTypeError,
    NodeType,
    node_ops,
)

from .exceptions import ViewRegistryError


if TYPE_CHECKING:
    from collections.abc import Generator

    from virtuals.loc import site as site_

    from .registry import ViewRegistry
    from .view import View

__all__ = [
    "ViewBase",
]

logger = getLogger(__name__)


@attrs.frozen
class ViewBase(ABC):
    """Base class for all views.

    Views are thin wrappers over Container that provide familiar Python
    interfaces. All storage operations are delegated to the Container API.

    Design:
    - Stateless: No cached data, always delegates to container
    - Immutable: View instances don't change (frozen attrs)
    - Registry-aware: Can create nested views automatically
    - Parent-aware: A child view knows the view it was opened from, so
      creating it goes through that parent and the parent's bookkeeping
    - Protocol-based: Subclasses implement Convertible/Initializable/Nestable as needed

    Attributes:
        container: Container instance for storage operations
        registry: Registry for nested view creation
        parent: View this view was opened from (None for a root). Pure
            in-memory navigation state, never stored.
    """

    container: Container
    registry: ViewRegistry
    parent: View | None = attrs.field(default=None, eq=False, repr=False)

    # =========================================================================
    # STRUCTURE & PROTOCOL
    # =========================================================================

    STRUCTURE: ClassVar[ContainerStructure]
    PROTOCOL: ClassVar[ContainerProtocol] = ContainerProtocol.NONE
    CONTAINER_CLS: ClassVar[type | None] = None

    @classmethod
    def get_structure(cls) -> ContainerStructure:
        """Get view structure."""
        if cls.STRUCTURE is None:
            raise
        return cls.STRUCTURE

    @classmethod
    def get_protocol(cls) -> ContainerProtocol:
        """Get view protocol hints."""
        return cls.PROTOCOL

    @classmethod
    def get_container_cls(cls) -> type | None:
        """Get container type, associated with this view."""
        return cls.CONTAINER_CLS

    # =========================================================================
    # WRITE SUPPORT
    # =========================================================================

    def ensure_created(self) -> None:
        """Ensure this view's container marker exists, then its internal layout.

        Runs the view-specific layout setup (``_ensure_internal_layout``).

        Call before any write operation. Idempotent - safe to call
        multiple times (an existing compatible marker short-circuits).

        A child view never stamps itself: it asks its parent view to create
        it (``_ensure_child_view``), which recurses up the parent chain for
        missing ancestors and lets every parent record the new child in its
        own bookkeeping (``_on_child_created``). A root view (no parent)
        creates itself; its storage parent must already exist, since the
        container layer never creates parents.

        Raises:
            ContainerNotFoundError: If a root view's storage parent is missing
            ContainerExistsError: If a container exists with an incompatible type
        """
        if self.parent is not None:
            self.parent._ensure_child_view(self)  # type: ignore[attr-defined]
            return
        Container.create(
            self.container.site,
            self.container.ctx,
            self.get_structure(),
            self.get_protocol(),
        )
        self._ensure_internal_layout()

    def _ensure_internal_layout(self) -> None:  # noqa: B027
        """View-specific layout setup, called at the end of ``ensure_created``.

        Default is a no-op. Views with an internal container layout
        (``LogIndexedDictView``'s ``__keys__/`` + ``__data__/``, etc.) override
        this to materialize their sub-containers with the correct structure.

        Runs AFTER the view's own marker is stamped, so ``self.container``
        is guaranteed to exist. Idempotent.
        """

    def ensure_child(self, address: object, view_class: type[View]) -> View:
        """Open the child at ``address`` as ``view_class``, creating it if missing.

        The single door for creating a child container. Idempotent: an
        existing compatible child is opened, a missing one is created at the
        site this view's child layout puts it (``open_child``; e.g. under
        ``__data__/`` for indexed dicts), its internal layout is set up, and
        ``_on_child_created`` lets this view record it. This view is created
        first if it is missing, recursively up the parent chain.

        Args:
            address: Child address in this view's address space
            view_class: View class for the child

        Returns:
            Child view, guaranteed materialized

        Raises:
            TypeError: If this view has no child navigation
            ContainerExistsError: If the child exists with an incompatible type
        """
        open_child = getattr(self, "open_child", None)
        if open_child is None:
            raise TypeError(f"{type(self).__name__} is not Nestable. Cannot create children.")
        child = open_child(address, view_class)
        self._ensure_child_view(child)
        return child

    def _ensure_child_view(self, child: View) -> None:
        """Materialize ``child``, a view this view opened at one of its child sites.

        Existing child: validates the marker against the child's view type
        and runs its internal layout. Missing child: ensures this view
        exists, stamps the child's marker, runs its internal layout, then
        fires ``_on_child_created`` exactly once.

        Args:
            child: Child view whose site this view's child layout chose
        """
        site = child.container.site
        ctx = child.container.ctx
        info = node_ops.get_node_info(site, ctx)
        if not info.exists:
            self.ensure_created()
        Container.create(
            site,
            ctx,
            child.get_structure(),
            child.get_protocol(),
            node_info=info,
        )
        child._ensure_internal_layout()  # type: ignore[attr-defined]
        if not info.exists:
            self._on_child_created(site[-1])

    def _put_child_primitive(self, address: site_.SiteSegment, value: object) -> None:
        """Write a primitive child, firing ``_on_child_created`` if the key is new.

        One node info read covers both the new-key check and the guard
        against overwriting a container child. The caller must have
        ensured this view exists (``ensure_created``).

        Args:
            address: Child address (storage segment under ``_children_container``)
            value: Primitive value to store

        Raises:
            ContainerTypeError: If the child exists as a container
        """
        children = self._children_container()
        child_site = (*children.site, address)
        info = node_ops.get_node_info(child_site, children.ctx)
        if info.exists and info.node_type != NodeType.PRIMITIVE:
            raise ContainerTypeError(f"Site is not a primitive: {child_site}")
        children.put_child_primitive(
            address,
            value,  # type: ignore[arg-type]
            validate=False,
            validate_parent=False,
        )
        if not info.exists:
            self._on_child_created(address)

    def _children_container(self) -> Container:
        """Container this view's children live under.

        Default is the view's own container. Views with an internal layout
        override it (indexed dicts keep children under ``__data__/``).
        """
        return self.container

    # =========================================================================
    # BOOKKEEPING HOOKS
    # =========================================================================

    def _on_child_created(self, address: site_.SiteSegment) -> None:  # noqa: B027
        """Hook: a new child appeared at ``address`` through this view.

        Fired once per new child, whether a container (``_ensure_child_view``)
        or a primitive (``_put_child_primitive``). Default is a no-op. Views
        that keep their own counts or indexes override it: ``DictView``
        bumps its length, indexed dicts record the key. Views that count
        live or by position leave it alone.

        Args:
            address: Child address as stored (the normalized segment)
        """

    def _rebuild_bookkeeping(self) -> None:  # noqa: B027
        """Hook: rebuild this view's own counter or index from storage.

        Default is a no-op. Views that keep bookkeeping override it so data
        written around them (unsafe writes, older versions) can be repaired.
        """

    def rebuild_bookkeeping(self) -> None:
        """Rebuild counters and indexes for this view and its whole subtree.

        Walks container children depth-first and runs each view's
        ``_rebuild_bookkeeping``, children before parents. A child's view
        type comes from its stored structure, via the registry or the
        built-in views. Children with an unknown structure are skipped.
        Silent if this view does not exist in storage.
        """
        if not self.container.exists():
            return
        for child in self._iter_child_views():
            child.rebuild_bookkeeping()  # type: ignore[attr-defined]
        self._rebuild_bookkeeping()

    def _iter_child_views(self) -> Generator[View, None, None]:
        """Yield a view for every container child, typed by its stored structure."""
        children = self._children_container()
        for address, info in children.iter_children(validate=False):
            if info.node_type != NodeType.CONTAINER or info.structure is None:
                continue
            view_class = _view_for_structure(self.registry, info.structure)
            if view_class is None:
                logger.debug(
                    "Skipping child with unknown structure",
                    extra={"site": children.site, "address": address, "structure": info.structure},
                )
                continue
            container = Container(ctx=children.ctx, site=(*children.site, address))
            yield view_class(container, self.registry, parent=self)  # type: ignore[call-arg]

    # =========================================================================
    # NAVIGATION HELPERS
    # =========================================================================

    def open_parent(self) -> View:
        """Navigate to parent container.

        Returns the view this one was opened from when known, otherwise
        resolves the storage parent's view type from its marker.

        Returns:
            View instance for parent container

        Raises:
            ValueError: If already at root (no parent)
        """
        if self.parent is not None:
            return self.parent
        parent_site = self.container.site[:-1] if self.container.site else None
        if parent_site is None:
            raise ValueError("Cannot navigate to parent - already at root")

        # Create parent container
        parent_container = Container(ctx=self.container.ctx, site=parent_site)

        # Get parent's structure ID to find correct view type
        parent_info = parent_container.info()
        if parent_info.structure is None:
            raise ValueError(f"Parent container at {parent_site} has no structure ID")

        # Use registry to create appropriate view
        view_class = self.registry.get_view_for_structure(parent_info.structure)
        return view_class(container=parent_container, registry=self.registry)  # type: ignore


def _view_for_structure(registry: ViewRegistry, structure: ContainerStructure) -> type[View] | None:
    """View class for a stored structure: the registry's, else the built-in one."""
    try:
        return registry.get_view_for_structure(structure)
    except ViewRegistryError:
        from virtuals import _views

        builtins = (
            _views.ByteArrayView,
            _views.EagerDictView,
            _views.EagerIndexedDictView,
            _views.EagerKh57View,
            _views.EagerListView,
            _views.EagerLogIndexedDictView,
            _views.FlatDictView,
            _views.FlatListView,
            _views.FrozenSetView,
            _views.LightDictView,
            _views.SetView,
            _views.TupleView,
        )
        for view_class in builtins:
            if view_class.STRUCTURE == structure:
                return view_class  # type: ignore[return-value]
        return None
