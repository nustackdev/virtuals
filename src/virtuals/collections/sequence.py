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

List-only methods (``+``, ``*``, ``sort``, ``copy``) are not here: a view
standing in for a list adds them itself.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import MutableSequence, Sequence
from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable

from .bases import CollectionBase


if TYPE_CHECKING:
    from collections.abc import Generator, Iterable


__all__ = [
    "MutableSequenceBase",
    "ReactiveSequenceBase",
    "ReactiveSequenceProtocol",
    "SequenceBase",
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
