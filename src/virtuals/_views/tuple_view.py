"""TupleView - Tuple-like view over container (immutable sequence)."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, TypeVar

from virtuals.collections import TupleMethods
from virtuals.container import (
    Container,
    ContainerNotFoundError,
    ContainerProtocol,
    ContainerStructure,
)
from virtuals.loc import key as key_
from virtuals.view import (
    ChildNavigationBase,
    ChildNestedGetBase,
    ChildNestedSetBase,
    ChildPrimitiveSetBase,
    IndexNotFoundError,
    MetadataBasedChildrenCountBase,
    UnsafePrimitiveOpsBase,
    View,
    ViewBase,
)

from ._index import require_int_index


if TYPE_CHECKING:
    from collections.abc import Iterable

    from virtuals.collections import (
        Containable,
        Convertible,
        Initializable,
        Nestable,
        Sizeable,
        Subscriptable,
    )
    from virtuals.types import Empty

__all__ = ["TupleView"]

ViewT = TypeVar("ViewT", bound=View)


class TupleView(
    MetadataBasedChildrenCountBase,
    ChildNavigationBase[int],
    ChildNestedGetBase,
    ChildNestedSetBase,
    ChildPrimitiveSetBase,
    UnsafePrimitiveOpsBase,
    ViewBase,
    TupleMethods[object],
):
    """Tuple-like view over container (immutable sequence).

    A read-only ``Sequence`` using integer keys: ``[i]`` and slices,
    ``len``, ``in``, iteration, ``reversed``, ``index`` and ``count``,
    plus ``+``, ``*`` and ordering as on ``tuple``. Not hashable, unlike
    ``tuple``: the stored items can be replaced under the view, so a hash
    would not hold for the view's lifetime.

    Type Parameters:
        V: Type of values (default: Value)

    Example:
        >>> coords: TupleView[int] = TupleView(container, registry)
        >>> # Must be initialized via store()
        >>> coords.store((10, 20, 30))
        >>> print(coords[0])  # 10
        >>> print(len(coords))  # 3
    """

    STRUCTURE: ClassVar[ContainerStructure] = ContainerStructure(3)
    PROTOCOL: ClassVar[ContainerProtocol] = ContainerProtocol.INDEXED | ContainerProtocol.SIZED
    CONTAINER_CLS: ClassVar[type] = tuple

    @classmethod
    def is_address_static(cls, address: object) -> bool:
        """Positive int indices are passthrough; negative need len() lookup."""
        return isinstance(address, int) and address >= 0

    def normalize_address(self, address: int) -> int:
        """Normalize index, handling negative indices.

        Args:
            address: Index to normalize

        Returns:
            Normalized positive index

        Raises:
            TypeError: If ``address`` is not an int, as ``tuple`` does
            IndexError: If index out of bounds (also on an absent tuple)
        """
        require_int_index(address, "tuple")
        length = self._len()

        if address < 0:
            address = length + address

        if address < 0 or address >= length:
            raise IndexNotFoundError("tuple index out of range")

        return address

    def __getitem__(self, address: int | slice) -> object | Empty | tuple[object, ...]:
        """Get item at index; a slice gives a plain tuple, as ``tuple`` does.

        Args:
            address: Index (supports negative) or slice

        Returns:
            Value at index

        Raises:
            IndexError: If index out of bounds
        """
        if isinstance(address, slice):
            return tuple(self[i] for i in range(*address.indices(len(self))))
        normalized = self.normalize_address(address)
        try:
            return self._get_child_value(normalized)
        except ContainerNotFoundError as e:
            raise IndexNotFoundError("tuple index out of range") from e

    # =========================================================================
    # VIEW INTERFACE
    # =========================================================================

    def extract(self) -> tuple[object, ...]:
        """Extract all items as tuple.

        Returns:
            Tuple of all items in order
        """
        return tuple(self)

    def store(self, value: Iterable[object], *, replace: bool = True) -> None:
        """Store tuple contents.

        Args:
            value: Sequence to store
            replace: If True, clear existing content first; if False, append
        """
        value = self._reject_empty_in(value)  # type: ignore[assignment]
        self.ensure_created()
        if replace:
            self.container.clear_children()
            count = 0
        else:
            count = self._len()
        for item in value:
            self._set_child_value(count, item)
            count += 1

        # Set final length metadata
        self._set_length(count)

    def open_child(self, address: int, view: type[ViewT]) -> ViewT:
        """Open child view at index.

        Pure navigation — does not write to storage.

        Args:
            address: Child container index
            view: View class for child

        Returns:
            View instance for child container
        """
        normalized = self.normalize_address(address)
        child_site = key_.join_segment(self.container.site, normalized)
        child_container = Container(ctx=self.container.ctx, site=child_site)
        return view(child_container, self.registry, parent=self)


if TYPE_CHECKING:
    # Verify protocol implementations
    _subscriptable: type[Subscriptable[int, object]] = TupleView
    _convertible: type[Convertible[object]] = TupleView
    _initializable: type[Initializable[Iterable[object]]] = TupleView
    _nestable: type[Nestable[int]] = TupleView
    _containable: type[Containable[object]] = TupleView
    _sizeable: type[Sizeable] = TupleView
    pass
