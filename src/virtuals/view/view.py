"""View protocol for Layer 3.

Views are thin wrappers over Container providing protocol-based capabilities.
Views are created through Navigator, not directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable


if TYPE_CHECKING:
    from virtuals.container import Container, ContainerProtocol, ContainerStructure

    from .registry import ViewRegistry

__all__ = [
    "View",
]


@runtime_checkable
class View(Protocol):
    """View Protocol.

    Views are thin wrappers over Container that provide familiar Python
    interfaces. All storage operations are delegated to the Container API.

    Views are created through Navigator.root(ctx) or Navigator.root_at(site, ctx).
    Child views are created via open_child() (Nestable protocol).

    Attributes:
        container: Container instance for storage operations
        registry: Registry for nested view creation
    """

    @classmethod
    def get_default_parent_view(cls) -> type[View] | None:
        """Returns view used to create missing parents."""
        ...

    @classmethod
    def get_structure(cls) -> ContainerStructure:
        """Get view structure."""
        ...

    @classmethod
    def get_protocol(cls) -> ContainerProtocol:
        """Get view protocol hints."""
        ...

    @classmethod
    def get_container_cls(cls) -> type | None:
        """Get container type, associated with this view."""
        ...

    def open_parent(self) -> View:
        """Navigate to parent container.

        Returns:
            View instance for parent container

        Raises:
            ValueError: If already at root (no parent)
        """
        ...

    def __init__(self, container: Container, registry: ViewRegistry) -> None:
        """Initializes a new View with given container and registry."""
        pass
