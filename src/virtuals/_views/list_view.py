"""ListView - List-like view over container.

Provides three classes following the eager/lazy facet pattern:
- ListViewBase: shared mutations, lifecycle
- EagerListView: reads return extracted Python values
- LazyListView: reads return child Views for containers, values for primitives
"""

from __future__ import annotations

from collections.abc import MutableSequence
from typing import TYPE_CHECKING, ClassVar

from virtuals.container import (
    ContainerNotFoundError,
    ContainerProtocol,
    ContainerStructure,
    NodeType,
)
from virtuals.view import (
    ChildNavigationBase,
    ChildNestedGetBase,
    ChildNestedSetBase,
    ChildObservableBase,
    ChildPrimitiveSetBase,
    LazyChildReadBase,
    MetadataBasedChildrenCountBase,
    ObservableBase,
    UnsafePrimitiveOpsBase,
    ViewBase,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable
    from collections.abc import Sequence as PySequence

    from virtuals.collections import (
        Assignable,
        ChildObservable,
        Clearable,
        Containable,
        Convertible,
        Deletable,
        Initializable,
        Nestable,
        Observable,
        ReactiveSequenceProtocol,
        Sizeable,
        Subscriptable,
    )
    from virtuals.types import Empty, Value


__all__ = [
    "EagerListView",
    "LazyListView",
    "ListViewBase",
]


# =============================================================================
# BASE — shared by eager and lazy facets
# =============================================================================


class ListViewBase(
    ObservableBase,
    ChildObservableBase[int],
    MetadataBasedChildrenCountBase,
    ChildNavigationBase[int],
    ChildNestedGetBase,
    ChildNestedSetBase,
    ChildPrimitiveSetBase,
    LazyChildReadBase,
    UnsafePrimitiveOpsBase,
    ViewBase,
):
    """List-like view base — shared by eager and lazy facets.

    Provides mutations, index normalization, lifecycle. Read operations
    that surface child data (__getitem__, __iter__) are defined by the
    EagerListView and LazyListView facets.

    Example:
        >>> tasks = nav.root(ctx)
        >>> tasks.append("Buy groceries")
        >>> tasks.append("Write code")
        >>> print(len(tasks))  # 2
    """

    STRUCTURE: ClassVar[ContainerStructure] = ContainerStructure(2)
    PROTOCOL: ClassVar[ContainerProtocol] = (
        ContainerProtocol.INDEXED | ContainerProtocol.MUTABLE | ContainerProtocol.SIZED
    )

    @classmethod
    def is_address_static(cls, address: object) -> bool:
        """Positive int indices are passthrough; negative need len() lookup."""
        return isinstance(address, int) and address >= 0

    def normalize_address(self, address: int) -> int:
        """Normalize index, handling negative indices."""
        length = len(self)

        if address < 0:
            address = length + address

        if address < 0 or address >= length:
            raise IndexError("list index out of range")

        return address

    # =========================================================================
    # MEMBERSHIP (same for both facets)
    # =========================================================================

    def __contains__(self, obj: object) -> bool:
        """Check if value exists in list."""
        for item in self:
            if item == obj:
                return True
        return False

    # =========================================================================
    # MUTATIONS (same for both facets)
    # =========================================================================

    def __setitem__(self, address: int, value: object) -> None:
        """Set item at index."""
        normalized = self.normalize_address(address)
        self._set_child_value(normalized, value)

    def __delitem__(self, address: int) -> None:
        """Delete item at index and shift remaining items."""
        self.ensure_created()
        normalized = self.normalize_address(address)
        length = len(self)

        self.container.delete_child(normalized)

        for i in range(normalized + 1, length):
            child_type = self.container.get_child_type(i)
            if child_type == NodeType.PRIMITIVE:
                value = self.container.get_child_primitive(i)
                self.container.put_child_primitive(i - 1, value)
                self.container.delete_child(i)
            elif child_type == NodeType.CONTAINER:
                raise NotImplementedError(
                    "Deleting list items with container children not yet supported"
                )

        self._set_length(length - 1)

    def append(self, value: object) -> None:
        """Append value to end."""
        index = len(self)
        self._set_child_value(index, value)
        self._set_length(index + 1)

    def insert(self, address: int, value: object) -> None:
        """Insert value at index, shifting later items."""
        self.ensure_created()
        length = len(self)

        if address < 0:
            address = max(0, length + address)
        else:
            address = min(address, length)

        for i in range(length - 1, address - 1, -1):
            child_type = self.container.get_child_type(i)
            if child_type == NodeType.PRIMITIVE:
                child_value = self.container.get_child_primitive(i)
                self.container.put_child_primitive(i + 1, child_value)
            elif child_type == NodeType.CONTAINER:
                raise NotImplementedError(
                    "Inserting into list with container children not yet supported"
                )

        self._set_child_value(address, value)
        self._set_length(length + 1)

    def clear(self) -> None:
        """Remove all items."""
        self.ensure_created()
        self.container.clear_children()
        self._set_length(0)

    def extend(self, values: Iterable[object]) -> None:
        """Extend list with items from iterable."""
        for value in values:
            self.append(value)

    def remove(self, value: object) -> None:
        """Remove first occurrence of value."""
        for i, item in enumerate(self):
            if item == value:
                del self[i]
                return
        raise ValueError(f"{value!r} is not in list")

    def store(self, value: Iterable[object]) -> None:
        """Store list contents."""
        self.clear()

        count = 0
        for item in value:
            self._set_child_value(count, item)
            count += 1

        self._set_length(count)


# =============================================================================
# EAGER FACET — reads return extracted Python values
# =============================================================================


class EagerListView(ListViewBase):
    """Eager list view — reads return extracted Python values.

    The default list experience: __getitem__ returns materialized values,
    iteration yields Python objects. Works naturally with sorted(), islice(), etc.

    Cross-navigate to lazy facet via .lazy property.
    """

    CONTAINER_CLS: ClassVar[type] = list

    def __getitem__(self, address: int) -> object | Empty:
        """Get item at index — returns extracted Python value."""
        normalized = self.normalize_address(address)
        try:
            return self._get_child_value(normalized)
        except ContainerNotFoundError as e:
            raise IndexError("list index out of bounds") from e

    def __iter__(self) -> Generator[Value, None, None]:
        """Iterate over items — yields extracted Python values."""
        for k, v in self.container.iter_children():
            if v.node_type == NodeType.PRIMITIVE:
                yield v.primitive_value
            elif v.node_type == NodeType.CONTAINER:
                yield self[int(k)]

    def __reversed__(self) -> Generator[object, None, None]:
        """Iterate in reverse order."""
        for i in range(len(self) - 1, -1, -1):
            yield self[i]

    def __add__(self, other: Iterable[object]) -> list[object]:
        """Concatenate with another iterable, returning a plain list."""
        return list(self) + list(other)

    def __radd__(self, other: Iterable[object]) -> list[object]:
        """Support ``other + list_view``."""
        return list(other) + list(self)

    def pop(self, address: int = -1) -> object | Empty:
        """Remove and return item at index."""
        if len(self) == 0:
            raise IndexError("pop from empty list")

        value = self[address]
        del self[address]
        return value

    def extract(self) -> Iterable[object]:
        """Extract all items as list."""
        return list(self)

    # =========================================================================
    # FACET NAVIGATION
    # =========================================================================

    @property
    def lazy(self) -> LazyListView:
        """Switch to lazy facet — reads return child Views."""
        return LazyListView(container=self.container, registry=self.registry)

    @property
    def eager(self) -> EagerListView:
        """Identity — already eager."""
        return self

    # =========================================================================
    # FUNCTIONAL OPERATIONS
    # =========================================================================

    def map(self, fn: Callable[[object], object]) -> list[object]:
        """Apply function to each element."""
        return [fn(item) for item in self]

    def filter(self, fn: Callable[[object], bool]) -> list[object]:
        """Filter elements by predicate."""
        return [item for item in self if fn(item)]

    def reduce(self, fn: Callable[[object, object], object], initial: object) -> object:
        """Reduce list to single value."""
        result = initial
        for item in self:
            result = fn(result, item)
        return result

    # =========================================================================
    # SEARCH OPERATIONS
    # =========================================================================

    def index(self, value: object) -> int:
        """Find index of first occurrence of value."""
        for i, item in enumerate(self):
            if item == value:
                return i
        raise ValueError(f"{value!r} is not in list")

    def count(self, value: object) -> int:
        """Count occurrences of value."""
        return sum(1 for item in self if item == value)

    def find(self, fn: Callable[[object], bool]) -> object:
        """Find first element matching predicate."""
        for item in self:
            if fn(item):
                return item
        raise ValueError("No matching element found")

    def find_index(self, fn: Callable[[object], bool]) -> int:
        """Find index of first element matching predicate."""
        for i, item in enumerate(self):
            if fn(item):
                return i
        raise ValueError("No matching element found")


# =============================================================================
# LAZY FACET — reads return child Views
# =============================================================================


class LazyListView(ListViewBase):
    """Lazy list view — reads return child Views for containers.

    Enables composition without materializing data. Iteration yields child
    Views for container children, values for primitives. Works naturally
    with Python tools: islice(lazy_list, 5) yields Views.

    Cross-navigate to eager facet via .eager property.
    """

    def __getitem__(self, address: int) -> object:
        """Get child — returns View for containers, value for primitives."""
        normalized = self.normalize_address(address)
        try:
            return self._get_child_view_or_value(normalized)
        except ContainerNotFoundError as e:
            raise IndexError("list index out of bounds") from e

    def __iter__(self) -> Generator[object, None, None]:
        """Iterate — yields Views for containers, values for primitives."""
        for k, v in self.container.iter_children():
            if v.node_type == NodeType.PRIMITIVE:
                yield v.primitive_value
            elif v.node_type == NodeType.CONTAINER:
                yield self._get_child_view_or_value(int(k), node_info=v)

    def __reversed__(self) -> Generator[object, None, None]:
        """Iterate in reverse order."""
        for i in range(len(self) - 1, -1, -1):
            yield self[i]

    # =========================================================================
    # FACET NAVIGATION
    # =========================================================================

    @property
    def eager(self) -> EagerListView:
        """Switch to eager facet — reads return extracted values."""
        return EagerListView(container=self.container, registry=self.registry)

    @property
    def lazy(self) -> LazyListView:
        """Identity — already lazy."""
        return self


MutableSequence.register(EagerListView)


if TYPE_CHECKING:
    # Verify protocol implementations
    _subscriptable: type[Subscriptable[int, object]] = EagerListView
    _convertible: type[Convertible[object]] = EagerListView
    _initializable: type[Initializable[Iterable[object]]] = EagerListView
    _assignable: type[Assignable[int, object]] = EagerListView
    _nestable: type[Nestable[int]] = EagerListView
    _containable: type[Containable[object]] = EagerListView
    _sizeable: type[Sizeable] = EagerListView
    _deletable: type[Deletable[int]] = EagerListView
    _clearable: type[Clearable] = EagerListView
    _reactive_sequence: type[ReactiveSequenceProtocol[object]] = EagerListView
    _observable: type[Observable] = EagerListView
    _observable_children: type[ChildObservable] = EagerListView
    # Python types
    _py_seq: type[PySequence[object]] = EagerListView
    _py_iter: type[Iterable] = EagerListView
