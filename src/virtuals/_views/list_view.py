"""ListView - List-like view over container.

Provides three classes following the eager/lazy facet pattern:
- ListViewBase: shared mutations, lifecycle
- EagerListView: reads return extracted Python values
- LazyListView: reads return child Views for containers, values for primitives
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from virtuals.collections import ListMethods, ReactiveSequenceBase
from virtuals.container import (
    ContainerNotFoundError,
    ContainerProtocol,
    ContainerStructure,
    NodeType,
)
from virtuals.tkv.filter import PrefixFilter
from virtuals.tkv.observer import SubscriptionOptions
from virtuals.view import (
    ChildNavigationBase,
    ChildNestedGetBase,
    ChildNestedSetBase,
    ChildObservableBase,
    ChildPrimitiveSetBase,
    DescendantsObservableBase,
    IndexNotFoundError,
    LazyChildReadBase,
    MetadataBasedChildrenCountBase,
    ObservableBase,
    UnsafePrimitiveOpsBase,
    ViewBase,
)

from ._index import require_int_index


if TYPE_CHECKING:
    from collections.abc import Generator, Iterable
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
    DescendantsObservableBase,
    MetadataBasedChildrenCountBase,
    ChildNavigationBase[int],
    ChildNestedGetBase,
    ChildNestedSetBase,
    ChildPrimitiveSetBase,
    LazyChildReadBase,
    UnsafePrimitiveOpsBase,
    ViewBase,
    ReactiveSequenceBase[object],
    ListMethods[object],
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
        """Normalize index, handling negative indices.

        Uses the stored length with no existence check: an index on an
        absent list is out of range, like on an empty one.

        Raises:
            TypeError: If ``address`` is not an int, as ``list`` does.
            IndexError: If ``address`` is out of range.
        """
        require_int_index(address)
        length = self._len()

        if address < 0:
            address = length + address

        if address < 0 or address >= length:
            raise IndexNotFoundError("list index out of range")

        return address

    def on_child_change(self, address: int) -> SubscriptionOptions:
        """Build subscription options for the item at ``address`` and its subtree.

        A watch is a filter, not a read: a non-negative index names a slot
        whether or not the list, or that slot, exists yet, so it is used as
        is and the watch works on an absent list (contract: subscriptions
        on absent containers work). A negative index is resolved against
        the current length and raises ``IndexError`` when out of range.
        """
        if address < 0:
            address = self.normalize_address(address)
        return SubscriptionOptions(filter=PrefixFilter(prefix=(*self.container.site, address)))

    # =========================================================================
    # MUTATIONS (same for both facets)
    # =========================================================================

    def __setitem__(self, address: int | slice, value: object) -> None:
        """Set the item at an index, or the items of a slice as ``list`` does.

        A slice write reads the items whole and stores the result back.
        """
        if isinstance(address, slice):
            self._set_slice(address, value)  # type: ignore[arg-type]
            return
        normalized = self.normalize_address(address)
        self._set_child_value(normalized, value)

    def set_child_container_as(
        self,
        address: int,
        value: object,
        view_class: type,
    ) -> None:
        """Peer of ``__setitem__`` with an explicit child view class.

        Mirrors ``__setitem__`` — normalize the index and populate the child
        — but takes ``view_class`` explicitly instead of dispatching by
        Python value type. Used by Refs that already know the child layout
        (``ShapesListRef[Shape]`` overwriting a slot, etc.). A value holding
        EMPTY anywhere raises ``ValueError`` before anything is written.
        """
        normalized = self.normalize_address(address)
        value = self._reject_empty_at(normalized, value)
        self.ensure_created()
        self._populate_child_container(normalized, value, view_class=view_class)

    def __delitem__(self, address: int | slice) -> None:
        """Delete item at index and shift remaining items.

        O(n) in the number of storage entries under the shifted tail. For
        large lists of container children where middle-position deletes are
        common, prefer a dict keyed by id. A slice is deleted as ``list``
        does: the items are read whole and the rest stored back.
        """
        if isinstance(address, slice):
            self._del_slice(address)
            return
        normalized = self.normalize_address(address)  # IndexError before any write
        self.ensure_created()
        length = self._len()

        self.container.delete_child(normalized)

        for i in range(normalized + 1, length):
            child_type = self.container.get_child_type(i)
            if child_type == NodeType.PRIMITIVE:
                value = self.container.get_child_primitive(i)
                self.container.put_child_primitive(i - 1, value)
                self.container.delete_child(i)
            elif child_type == NodeType.CONTAINER:
                self.container.move_child_subtree(i, i - 1)

        self._set_length(length - 1)

    def append(self, value: object) -> None:
        """Append value to end."""
        index = self._len()
        self._set_child_value(index, value)
        self._set_length(index + 1)

    def insert(self, address: int, value: object) -> None:
        """Insert value at index, shifting later items."""
        require_int_index(address)
        self._reject_empty(address, value)
        self.ensure_created()
        length = self._len()

        if address < 0:
            address = max(0, length + address)
        else:
            address = min(address, length)

        for i in range(length - 1, address - 1, -1):
            child_type = self.container.get_child_type(i)
            if child_type == NodeType.PRIMITIVE:
                child_value = self.container.get_child_primitive(i)
                self.container.put_child_primitive(i + 1, child_value)
                if i == address:
                    # Vacate the slot: a compound value over the stale
                    # primitive row would trip the kind rule after the
                    # tail was already shifted.
                    self.container.delete_child(i)
            elif child_type == NodeType.CONTAINER:
                self.container.move_child_subtree(i, i + 1)

        self._set_child_value(address, value)
        self._set_length(length + 1)

    def clear(self) -> None:
        """Remove all items."""
        if not self.container.exists():
            return  # clear of nothing is not a write: the view stays absent
        self.ensure_created()
        self.container.clear_children()
        self._set_length(0)

    def extend(self, values: Iterable[object]) -> None:
        """Extend list with items from iterable."""
        values = self._reject_empty_in(values)  # type: ignore[assignment]
        for value in values:
            self.append(value)

    def remove(self, value: object) -> None:
        """Remove first occurrence of value. On an absent list, as on an empty one."""
        if self._len() == 0:
            raise ValueError(f"{value!r} is not in list")
        for i, item in enumerate(self):
            if item == value:
                del self[i]
                return
        raise ValueError(f"{value!r} is not in list")

    def reverse(self) -> None:
        """Reverse in place, as ``list.reverse``: the items are read whole and stored back reversed.

        Not by swaps: a swap would write a compound item over a primitive
        one, which a list view refuses (an item keeps its kind).
        """
        if self._len() < 2:
            return
        self._replace_all(self._plain_values()[::-1])

    def pop(self, index: int = -1) -> object:
        """Remove and return the item at ``index``, read as a plain value, as ``list.pop``."""
        if self._len() == 0:
            raise IndexError("pop from empty list")
        return super().pop(index)

    def _plain_values(self) -> list[object]:
        """Every item as plain data, read through the eager facet in one scan."""
        return list(self.eager)  # type: ignore[call-overload]

    def _replace_all(self, values: list[object]) -> None:
        """Store ``values`` as the items: one bulk store.

        A view among them (``lazy[:] = lazy`` hands in this list's own
        children) is read to data first, before the store clears what it
        stands at.
        """
        self.store([v.eager.extract() if isinstance(v, ViewBase) else v for v in values])  # type: ignore[attr-defined]

    def store(self, value: Iterable[object], *, replace: bool = True) -> None:
        """Store list contents.

        Args:
            value: Iterable to store
            replace: If True, clear existing content first; if False, append
        """
        value = self._reject_empty_in(value)  # type: ignore[assignment]
        self.ensure_created()
        if replace:
            self.clear()
            count = 0
        else:
            count = self._len()
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

    def __getitem__(self, address: int | slice) -> object | Empty | list:
        """Get item at index — returns extracted Python value.

        With a slice, returns a plain list of extracted values (matches
        Python list semantics; negative/step handled via ``slice.indices``).
        """
        if isinstance(address, slice):
            return [self[i] for i in range(*address.indices(len(self)))]
        normalized = self.normalize_address(address)
        try:
            return self._get_child_value(normalized)
        except ContainerNotFoundError as e:
            raise IndexNotFoundError("list index out of bounds") from e

    def __iter__(self) -> Generator[Value, None, None]:
        """Iterate over items — yields extracted Python values."""
        for k, v in self.container.iter_children(strict=True):
            if v.node_type == NodeType.PRIMITIVE:
                yield v.primitive_value
            elif v.node_type == NodeType.CONTAINER:
                yield self[int(k)]

    def extract(self) -> Iterable[object]:
        """Extract all items as list."""
        return list(self)

    # =========================================================================
    # FACET NAVIGATION
    # =========================================================================

    @property
    def lazy(self) -> LazyListView:
        """Switch to lazy facet — reads return child Views."""
        return LazyListView(container=self.container, registry=self.registry, parent=self.parent)

    @property
    def eager(self) -> EagerListView:
        """Identity — already eager."""
        return self


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

    def __getitem__(self, address: int | slice) -> object | list:
        """Get child — returns View for containers, value for primitives.

        With a slice, returns a plain list of child views/values (matches
        Python list semantics; negative/step handled via ``slice.indices``).
        """
        if isinstance(address, slice):
            return [self[i] for i in range(*address.indices(len(self)))]
        normalized = self.normalize_address(address)
        try:
            return self._get_child_view_or_value(normalized)
        except ContainerNotFoundError as e:
            raise IndexNotFoundError("list index out of bounds") from e

    def __iter__(self) -> Generator[object, None, None]:
        """Iterate — yields Views for containers, values for primitives."""
        for k, v in self.container.iter_children(strict=True):
            if v.node_type == NodeType.PRIMITIVE:
                yield v.primitive_value
            elif v.node_type == NodeType.CONTAINER:
                yield self._get_child_view_or_value(int(k), node_info=v)

    # =========================================================================
    # FACET NAVIGATION
    # =========================================================================

    def extract(self) -> list[object]:
        """The whole subtree as plain Python values, read through the eager facet."""
        return self.eager.extract()

    @property
    def eager(self) -> EagerListView:
        """Switch to eager facet — reads return extracted values."""
        return EagerListView(container=self.container, registry=self.registry, parent=self.parent)

    @property
    def lazy(self) -> LazyListView:
        """Identity — already lazy."""
        return self


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
