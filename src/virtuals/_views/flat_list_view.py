"""FlatListView — primitives-only list with length tracking, no nesting overhead.

Lightweight list view optimized for primitive values (int, float, str, bool,
bytes, None).  No nested containers, no child navigation. Observable like
``ListView``: the whole list, one index, the indexes, a pattern.
Each element is a single ``put_child_primitive`` / ``get_child_primitive`` call.

Append is O(1) (one primitive write + one metadata write for length).
Iteration is a prefix scan over flat primitives only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, cast

from virtuals.collections import ListMethods, ReactiveSequenceBase
from virtuals.container import ContainerProtocol, ContainerStructure
from virtuals.tkv.filter import PrefixFilter
from virtuals.tkv.observer import SubscriptionOptions
from virtuals.types import Empty, Value
from virtuals.view import (
    ChildObservableBase,
    ChildPrimitiveSetBase,
    DescendantsObservableBase,
    IndexNotFoundError,
    MetadataBasedChildrenCountBase,
    ObservableBase,
    UnsafePrimitiveOpsBase,
    ViewBase,
)

from ._index import require_int_index


if TYPE_CHECKING:
    from collections.abc import Generator, Iterable


__all__ = [
    "FlatListView",
]


class FlatListView(
    ObservableBase,
    ChildObservableBase[int],
    DescendantsObservableBase,
    MetadataBasedChildrenCountBase,
    ChildPrimitiveSetBase,
    UnsafePrimitiveOpsBase,
    ViewBase,
    ReactiveSequenceBase[Value],
    ListMethods[Value],
):
    """Primitives-only list with O(1) append and length tracking.

    Stores only primitive values using integer keys (0, 1, 2, ...).
    Includes ``__len__`` via metadata.  No nested containers.

    Use when:
    - Elements are always primitives (str, int, etc.)
    - Append-heavy workload (key index, log, etc.)
    - You need fast ``len()`` and iteration without nesting overhead

    Example::

        >>> keys = FlatListView(container, registry)
        >>> keys.append("alice")
        >>> keys.append("bob")
        >>> print(len(keys))   # 2
        >>> print(keys[0])     # "alice"
        >>> print(list(keys))  # ["alice", "bob"]
    """

    STRUCTURE: ClassVar[ContainerStructure] = ContainerStructure(14)
    PROTOCOL: ClassVar[ContainerProtocol] = (
        ContainerProtocol.INDEXED | ContainerProtocol.MUTABLE | ContainerProtocol.SIZED
    )
    CONTAINER_CLS: ClassVar[type] = list

    @classmethod
    def is_address_static(cls, address: object) -> bool:
        """Positive int indices are passthrough."""
        return isinstance(address, int) and address >= 0

    def normalize_address(self, address: int) -> int:
        """Normalize index, handling negative indices. No existence check.

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
        """Build subscription options for the item at ``address``, as ``ListView`` does.

        A non-negative index names a slot whether or not the list, or that
        slot, exists yet, so the watch works on an absent list. A negative
        index is resolved against the current length and raises
        ``IndexError`` when out of range.
        """
        if address < 0:
            address = self.normalize_address(address)
        return SubscriptionOptions(filter=PrefixFilter(prefix=(*self.container.site, address)))

    # -- Read ---------------------------------------------------------------

    def __getitem__(self, address: int | slice) -> Value | list[Value]:
        """Get primitive value at index; a slice gives a plain list, as ``list`` does."""
        if isinstance(address, slice):
            start, stop, step = address.indices(len(self))
            if step == 1:
                return self.extract_range(start, stop)
            return [self[i] for i in range(start, stop, step)]
        normalized = self.normalize_address(address)
        value = self.container.get_child_primitive(normalized)
        if isinstance(value, Empty):
            raise IndexNotFoundError("list index out of range")
        return value

    def __iter__(self) -> Generator[Value, None, None]:
        """Iterate over values via prefix scan (flat primitives only)."""
        for _k, v in self.container.iter_children(validate=False, strict=True):
            yield cast("Value", v.primitive_value)

    # -- Write --------------------------------------------------------------

    def append(self, value: Value) -> None:
        """Append a primitive value.  O(1): one primitive write + one metadata write."""
        self._reject_empty_in(value)
        self.ensure_created()
        index = self._len()
        self.container.put_child_primitive(index, value)
        self._set_length(index + 1)

    def __setitem__(self, address: int | slice, value: Value) -> None:
        """Set the value at an existing index, or the items of a slice as ``list`` does.

        A slice write reads the items whole and stores the result back.
        """
        if isinstance(address, slice):
            self._set_slice(address, value)  # type: ignore[arg-type]
            return
        normalized = self.normalize_address(address)
        self._reject_empty_in(value)
        self.container.put_child_primitive(normalized, value)

    def __delitem__(self, address: int | slice) -> None:
        """Delete the item at ``address``, shifting later items down. O(n - address) rows.

        A slice is deleted as ``list`` does: the items are read whole and
        the rest stored back.
        """
        if isinstance(address, slice):
            self._del_slice(address)
            return
        normalized = self.normalize_address(address)  # IndexError before any write
        length = self._len()
        for i in range(normalized + 1, length):
            self.container.put_child_primitive(i - 1, self.container.get_child_primitive(i))
        self.container.delete_child(length - 1)
        self._set_length(length - 1)

    def insert(self, index: int, value: Value) -> None:
        """Insert ``value`` before ``index``, shifting later items up, as ``list.insert``.

        An index past either end clamps to it. O(n - index) rows.
        """
        if not isinstance(index, int):
            raise TypeError(f"'{type(index).__name__}' object cannot be interpreted as an integer")
        self._reject_empty_in(value)
        self.ensure_created()
        length = self._len()
        index = max(0, length + index) if index < 0 else min(index, length)
        for i in range(length - 1, index - 1, -1):
            self.container.put_child_primitive(i + 1, self.container.get_child_primitive(i))
        self.container.put_child_primitive(index, value)
        self._set_length(length + 1)

    def clear(self) -> None:
        """Remove all items."""
        if not self.container.exists():
            return  # clear of nothing is not a write: the view stays absent
        self.ensure_created()
        self.container.clear_children(validate=False)
        self._set_length(0)

    # -- Bulk ---------------------------------------------------------------

    def store(self, value: Iterable[Value], *, replace: bool = True) -> None:
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
            self.container.put_child_primitive(count, item)
            count += 1
        self._set_length(count)

    def extend(self, values: Iterable[Value]) -> None:
        """Append multiple values."""
        values = self._reject_empty_in(values)  # type: ignore[assignment]
        index = None
        for item in values:
            if index is None:  # created on the first item: extend([]) writes nothing
                self.ensure_created()
                index = self._len()
            self.container.put_child_primitive(index, item)
            index += 1
        if index is not None:
            self._set_length(index)

    # -- Extract ------------------------------------------------------------

    def extract(self) -> list[Value]:
        """Extract all items as a plain list."""
        return list(self)

    def __reversed__(self) -> Generator[Value, None, None]:
        """Values last first: one reverse scan."""
        for _k, v in self.container.iter_children(validate=False, strict=True, reverse=True):
            yield cast("Value", v.primitive_value)

    def _plain_values(self) -> list[Value]:
        """Every value, in order, in one scan."""
        return list(self)

    def _replace_all(self, values: list[Value]) -> None:
        """Store ``values`` as the items: one bulk store."""
        self.store(values)

    def extract_range(self, start: int, stop: int) -> list[Value]:
        """Extract a range of items by index.  O(stop-start) individual reads."""
        length = len(self)
        start = max(0, start)
        stop = min(stop, length)
        result = []
        for i in range(start, stop):
            value = self.container.get_child_primitive(i)
            if not isinstance(value, Empty):
                result.append(value)
        return result
