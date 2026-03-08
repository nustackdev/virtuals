"""Sequence collection bases.

Provides default implementations following collections.abc.MutableSequence pattern.
Subclasses implement the abstract core; the base derives everything else.

Abstract (must implement):
    __getitem__, __setitem__, __delitem__, __len__, insert

Provided for free:
    append, pop, extend, remove, clear, __contains__, __iter__,
    __reversed__, index, count, __add__, __radd__
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Generator, Iterable


__all__ = [
    "MutableSequenceBase",
    "SequenceBase",
]


class SequenceBase[V]:
    """Read-only sequence base. Provides iteration/search from getitem/len.

    Abstract:
        __getitem__(index) -> value
        __len__() -> int
    """

    @abstractmethod
    def __getitem__(self, index: int) -> V: ...

    @abstractmethod
    def __len__(self) -> int: ...

    def __iter__(self) -> Generator[V, None, None]:
        """Iterate over items in order."""
        for i in range(len(self)):
            yield self[i]

    def __reversed__(self) -> Generator[V, None, None]:
        """Iterate in reverse order."""
        for i in range(len(self) - 1, -1, -1):
            yield self[i]

    def __contains__(self, value: object) -> bool:
        """Check if value exists in sequence."""
        for item in self:
            if item == value:
                return True
        return False

    def index(self, value: object) -> int:
        """Find index of first occurrence.

        Raises:
            ValueError: If value not found.
        """
        for i, item in enumerate(self):
            if item == value:
                return i
        raise ValueError(f"{value!r} is not in sequence")

    def count(self, value: object) -> int:
        """Count occurrences of value."""
        return sum(1 for item in self if item == value)


class MutableSequenceBase[V](SequenceBase[V]):
    """Mutable sequence base. Adds append/pop/extend/remove/clear from insert/delitem.

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

    def append(self, value: V) -> None:
        """Append value to end."""
        self.insert(len(self), value)

    def pop(self, index: int = -1) -> V:
        """Remove and return item at index.

        Raises:
            IndexError: If sequence empty or index out of bounds.
        """
        value = self[index]
        del self[index]
        return value

    def extend(self, values: Iterable[V]) -> None:
        """Extend sequence with items from iterable."""
        for value in values:
            self.append(value)

    def remove(self, value: V) -> None:
        """Remove first occurrence of value.

        Raises:
            ValueError: If value not found.
        """
        for i, item in enumerate(self):
            if item == value:
                del self[i]
                return
        raise ValueError(f"{value!r} is not in sequence")

    def clear(self) -> None:
        """Remove all items. Override for bulk delete."""
        while len(self) > 0:
            del self[-1]

    def __add__(self, other: Iterable[V]) -> list[V]:
        """Concatenate with iterable, returning plain list."""
        return list(self) + list(other)

    def __radd__(self, other: Iterable[V]) -> list[V]:
        """Support other + sequence."""
        return list(other) + list(self)
