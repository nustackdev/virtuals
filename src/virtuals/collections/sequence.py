"""Sequence collection bases.

Mirror ``collections.abc.Sequence`` and ``MutableSequence``: the same
abstract core, the same mixin methods with the same semantics. Each base
is registered with its ABC, so every view built on it is a ``Sequence``
or ``MutableSequence`` to ``isinstance``.

Abstract (must implement):
    __getitem__, __len__                       (Sequence)
    __setitem__, __delitem__, insert           (MutableSequence)

Provided for free:
    __contains__, __iter__, __reversed__, index, count
    append, clear, reverse, extend, pop, remove, __iadd__

``ListMethods`` and ``TupleMethods`` add what ``list`` and ``tuple`` have
beyond the ABCs (``+``, ``*``, ordering, ``sort``, ``copy``, slice writes)
for a view that stands in for one of them.
"""

from __future__ import annotations

import operator
from abc import abstractmethod
from collections.abc import MutableSequence, Sequence
from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable

from .bases import CollectionBase


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable


__all__ = [
    "ListMethods",
    "MutableSequenceBase",
    "ReactiveSequenceBase",
    "ReactiveSequenceProtocol",
    "SequenceBase",
    "TupleMethods",
]

V = TypeVar("V")


class SequenceBase(CollectionBase[V]):
    """Read-only sequence base, as ``collections.abc.Sequence``.

    Inherits abstract __len__ from CollectionBase.
    Overrides __contains__ and __iter__ with index-based implementations.

    Abstract:
        __getitem__(index) -> value (an int, or a slice)
    """

    @abstractmethod
    def __getitem__(self, index: int) -> V: ...

    def __iter__(self) -> Generator[V, None, None]:
        """Iterate over items in order."""
        for i in range(len(self)):
            yield self[i]

    def __reversed__(self) -> Generator[V, None, None]:
        """Iterate in reverse order."""
        for i in range(len(self) - 1, -1, -1):
            yield self[i]

    def __contains__(self, value: object) -> bool:
        """True if an item is or equals ``value``."""
        return any(item is value or item == value for item in self)

    def index(self, value: object, start: int = 0, stop: int | None = None) -> int:
        """Position of the first item equal to ``value`` in ``[start:stop]``, as ``list.index``.

        Raises:
            ValueError: If no such item.
        """
        start, stop, _ = slice(start, stop).indices(len(self))
        for i, item in enumerate(self):
            if i >= stop:
                break
            if i >= start and (item is value or item == value):
                return i
        raise ValueError(f"{value!r} is not in sequence")

    def count(self, value: object) -> int:
        """Count occurrences of value."""
        return sum(1 for item in self if item is value or item == value)


class MutableSequenceBase(SequenceBase[V]):
    """Mutable sequence base, as ``collections.abc.MutableSequence``.

    Additional abstract:
        __setitem__(index, value) -> None
        __delitem__(index) -> None
        insert(index, value) -> None
    """

    @abstractmethod
    def __setitem__(self, index: int, value: V) -> None: ...

    @abstractmethod
    def __delitem__(self, index: int) -> None: ...

    @abstractmethod
    def insert(self, index: int, value: V) -> None:
        """Insert value at index, shifting later items."""
        ...

    def _plain_value(self, index: int) -> V:
        """An item as a plain value, for the methods that move or hand back items.

        The read by default. A sequence whose reads hand out live child
        views overrides it to read the plain value: a popped child is gone
        after the delete, and a moved one is written by value.
        """
        return self[index]

    def _plain_values(self) -> list[V]:
        """Every item as a plain value, in order: what a whole rewrite stores back.

        A list by default; a sequence standing in for another type (a
        bytearray) returns that type, so slice writes follow its rules.
        """
        return [self._plain_value(i) for i in range(len(self))]

    def _replace_all(self, values: list[V]) -> None:
        """Make the items ``values``, in order. Override for a bulk store."""
        self.clear()
        self.extend(values)

    def _set_slice(self, index: slice, values: Iterable[V]) -> None:
        """``s[a:b] = values`` by a whole rewrite: the plain type's rules and errors.

        Nothing is written when the plain type refuses the write.
        """
        items = self._plain_values()
        items[index] = values
        self._replace_all(items)

    def _del_slice(self, index: slice) -> None:
        """``del s[a:b]`` by a whole rewrite."""
        items = self._plain_values()
        del items[index]
        self._replace_all(items)

    def append(self, value: V) -> None:
        """Append value to end."""
        self.insert(len(self), value)

    def clear(self) -> None:
        """Remove all items. Override for bulk delete."""
        while len(self) > 0:
            del self[-1]

    def reverse(self) -> None:
        """Reverse the items in place: ``n // 2`` swaps."""
        n = len(self)
        for i in range(n // 2):
            j = n - i - 1
            self[i], self[j] = self._plain_value(j), self._plain_value(i)

    def extend(self, values: Iterable[V]) -> None:
        """Extend sequence with items from iterable. ``s.extend(s)`` doubles ``s``."""
        if values is self:
            values = list(values)
        for value in values:
            self.append(value)

    def pop(self, index: int = -1) -> V:
        """Remove and return item at index.

        Raises:
            IndexError: If sequence empty or index out of bounds.
        """
        value = self._plain_value(index)
        del self[index]
        return value

    def remove(self, value: V) -> None:
        """Remove first occurrence of value.

        Raises:
            ValueError: If value not found.
        """
        del self[self.index(value)]

    def __iadd__(self, values: Iterable[V]) -> MutableSequenceBase[V]:
        """``s += values``: extend in place."""
        self.extend(values)
        return self


def _repeat_count(count: object) -> int | None:
    """``count`` as an int for ``*``, or None when a sequence cannot be repeated by it."""
    try:
        return operator.index(count)
    except TypeError:
        return None


class TupleMethods(SequenceBase[V]):
    """What ``tuple`` has beyond ``Sequence``: ``+``, ``*``, ordering and ``==`` against tuples.

    New values come back as a plain ``tuple``. ``+`` and the comparisons
    take a tuple or another tuple view, as ``tuple`` does; a list, say,
    gives ``TypeError``.
    """

    def __add__(self, other: object) -> tuple[V, ...]:
        if not isinstance(other, tuple | TupleMethods):
            return NotImplemented
        return (*self, *other)

    def __radd__(self, other: object) -> tuple[V, ...]:
        if not isinstance(other, tuple):
            return NotImplemented
        return (*other, *self)

    def __mul__(self, count: object) -> tuple[V, ...]:
        n = _repeat_count(count)
        return NotImplemented if n is None else tuple(self) * n

    __rmul__ = __mul__

    def _cmp(self, other: object, op: Callable[[object, object], bool]) -> bool:
        if not isinstance(other, tuple | TupleMethods):
            return NotImplemented
        return op(tuple(self), tuple(other))

    def __eq__(self, other: object) -> bool:
        return self._cmp(other, operator.eq)

    def __lt__(self, other: object) -> bool:
        return self._cmp(other, operator.lt)

    def __le__(self, other: object) -> bool:
        return self._cmp(other, operator.le)

    def __gt__(self, other: object) -> bool:
        return self._cmp(other, operator.gt)

    def __ge__(self, other: object) -> bool:
        return self._cmp(other, operator.ge)

    __hash__ = None  # type: ignore[assignment]


class ListMethods(MutableSequenceBase[V]):
    """What ``list`` has beyond ``MutableSequence``.

    ``copy``, ``sort``, ``+``, ``*``, ``*=``, ordering and ``==`` against
    lists, and the slice writes (``s[a:b] = ...``, ``del s[a:b]``). A view
    built on it is a list in full: every ``list`` method with the same
    semantics. New values come back as a plain ``list`` holding the items
    the view reads (a shallow copy, as ``list.copy``). ``+`` and the
    comparisons take a list or another list view, as ``list`` does.

    ``sort`` and the slice writes read the items whole and store the
    result back: O(n) rows, whatever the slice. The view's own
    ``__setitem__`` and ``__delitem__`` hand a slice to ``_set_slice`` and
    ``_del_slice``.
    """

    def copy(self) -> list[V]:
        """A plain list of the items, as ``list.copy``."""
        return list(self)

    def sort(self, *, key: Callable[[V], object] | None = None, reverse: bool = False) -> None:
        """Sort in place, stably, as ``list.sort``: ``key`` sees the items as the view reads them.

        A comparison that fails raises before anything is written.
        """
        items = list(self)
        keys = items if key is None else [key(item) for item in items]
        order = sorted(range(len(items)), key=keys.__getitem__, reverse=reverse)
        if order != list(range(len(items))):
            plain = self._plain_values()
            self._replace_all([plain[i] for i in order])

    def __add__(self, other: object) -> list[V]:
        if not isinstance(other, list | ListMethods):
            return NotImplemented
        return [*self, *other]

    def __radd__(self, other: object) -> list[V]:
        if not isinstance(other, list):
            return NotImplemented
        return [*other, *self]

    def __mul__(self, count: object) -> list[V]:
        n = _repeat_count(count)
        return NotImplemented if n is None else list(self) * n

    __rmul__ = __mul__

    def __imul__(self, count: object) -> ListMethods[V]:
        n = _repeat_count(count)
        if n is None:
            return NotImplemented
        if n <= 0:
            self.clear()
        elif n > 1:
            self.extend(self._plain_values() * (n - 1))
        return self

    def _cmp(self, other: object, op: Callable[[object, object], bool]) -> bool:
        if not isinstance(other, list | ListMethods):
            return NotImplemented
        return op(list(self), list(other))

    def __eq__(self, other: object) -> bool:
        return self._cmp(other, operator.eq)

    def __lt__(self, other: object) -> bool:
        return self._cmp(other, operator.lt)

    def __le__(self, other: object) -> bool:
        return self._cmp(other, operator.le)

    def __gt__(self, other: object) -> bool:
        return self._cmp(other, operator.gt)

    def __ge__(self, other: object) -> bool:
        return self._cmp(other, operator.ge)

    __hash__ = None  # type: ignore[assignment]


class ReactiveSequenceBase(MutableSequenceBase[V]):
    """Reactive sequence base. Mutable sequence with change observation.

    Additional abstract:
        on_change() -> Subscription
        on_child_change(index) -> Subscription
        on_children_change() -> Subscription
    """

    @abstractmethod
    def on_change(self) -> object:
        """Subscribe to all changes in this sequence."""
        ...

    @abstractmethod
    def on_child_change(self, address: int) -> object:
        """Subscribe to changes at a specific index."""
        ...

    @abstractmethod
    def on_children_change(self) -> object:
        """Subscribe to changes on any index."""
        ...


Sequence.register(SequenceBase)
MutableSequence.register(MutableSequenceBase)


@runtime_checkable
class ReactiveSequenceProtocol(Protocol[V]):
    """Protocol: MutableSequence + Observable + ChildObservable.

    Use for type-checking views that support both mutation and change subscription.
    """

    def __getitem__(self, index: int) -> V: ...
    def __setitem__(self, index: int, value: V) -> None: ...
    def __delitem__(self, index: int) -> None: ...
    def __len__(self) -> int: ...
    def insert(self, index: int, value: V) -> None: ...  # noqa: D102
    def on_change(self) -> object: ...  # noqa: D102
    def on_child_change(self, address: int) -> object: ...  # noqa: D102
    def on_children_change(self) -> object: ...  # noqa: D102
