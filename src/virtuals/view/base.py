"""Base View implementation for Layer 3.

Views are thin wrappers over Container providing protocol-based capabilities.
Views are created through Navigator, not directly.
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, ClassVar

import attrs

from virtuals.container import Container, ContainerProtocol, ContainerStructure
from virtuals.container.types import DEFAULT_PARENT_PROTOCOL, DEFAULT_PARENT_STRUCTURE


if TYPE_CHECKING:
    from .registry import ViewRegistry
    from .view import View

__all__ = [
    "ViewBase",
]


@attrs.frozen
class ViewBase(ABC):
    """Base class for all views.

    Views are thin wrappers over Container that provide familiar Python
    interfaces. All storage operations are delegated to the Container API.

    Design:
    - Stateless: No cached data, always delegates to container
    - Immutable: View instances don't change (frozen attrs)
    - Registry-aware: Can create nested views automatically
    - Protocol-based: Subclasses implement Convertible/Initializable/Nestable as needed

    Attributes:
        container: Container instance for storage operations
        registry: Registry for nested view creation
    """

    container: Container
    registry: ViewRegistry

    # =========================================================================
    # STRUCTURE & PROTOCOL
    # =========================================================================

    STRUCTURE: ClassVar[ContainerStructure]
    PROTOCOL: ClassVar[ContainerProtocol] = ContainerProtocol.NONE
    CONTAINER_CLS: ClassVar[type | None] = None

    @classmethod
    def get_default_parent_view(cls) -> type[View] | None:
        """Returns view used to create missing parents.

        Defaults to EagerDictView for standard views.
        """
        from virtuals._views.dict_view import EagerDictView

        return EagerDictView

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
        """Ensure this view's container marker exists in storage, and run any
        view-specific internal layout setup (``_ensure_internal_layout``).

        Call before any write operation. Idempotent - safe to call
        multiple times (Container.create short-circuits when the marker
        already exists).

        Creates the full parent chain via ensure_healthy_parents=True,
        so even deeply navigated containers get materialized on first write.
        Note: when the write path routed through the ref layer's
        walk-and-ensure helper (``navigate_and_ensure``), every ancestor
        has already been stamped with its declared view type, so the
        default-parent-structure fallback here is a redundant safety net
        for direct container-API callers, not the hot path.
        """
        dpv = self.get_default_parent_view()
        Container.create(
            self.container.site,
            self.container.ctx,
            self.get_structure(),
            self.get_protocol(),
            default_parent_structure=dpv.get_structure() if dpv else DEFAULT_PARENT_STRUCTURE,
            default_parent_protocol=dpv.get_protocol() if dpv else DEFAULT_PARENT_PROTOCOL,
            ensure_healthy_parents=True,
        )
        self._ensure_internal_layout()

    def _ensure_internal_layout(self) -> None:
        """View-specific layout setup, called at the end of ``ensure_created``.

        Default is a no-op. Views with an internal container layout
        (``LogIndexedDictView``'s ``__keys__/`` + ``__data__/``, etc.) override
        this to materialize their sub-containers with the correct structure.

        Runs AFTER the view's own marker is stamped, so ``self.container``
        is guaranteed to exist. Idempotent.
        """

    # =========================================================================
    # NAVIGATION HELPERS
    # =========================================================================

    def open_parent(self) -> View:
        """Navigate to parent container.

        Returns:
            View instance for parent container

        Raises:
            ValueError: If already at root (no parent)
        """
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
