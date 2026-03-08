"""Set collection bases.

Provides default implementations following collections.abc.MutableSet pattern.
Subclasses implement the abstract core; the base derives everything else.

Abstract (must implement):
    __contains__, __iter__, __len__, add, discard

Provided for free:
    remove, clear, isdisjoint, issubset, issuperset,
    __or__, __and__, __sub__, __xor__, __le__, __ge__
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .bases import CollectionBase


if TYPE_CHECKING:
    from collections.abc import Iterable


__all__ = [
    "MutableSetBase",
    "ReactiveSetBase",
    "ReactiveSetProtocol",
    "SetBase",
]


class SetBase[V](CollectionBase[V]):
    """Read-only set base. Provides set algebra from contains/iter/len.

    All three abstract methods (__contains__, __iter__, __len__) inherited
    from CollectionBase — subclasses must implement all of them.
    """

    def isdisjoint(self, other: Iterable[object]) -> bool:
        """Check if no elements in common with other."""
        return not any(value in self for value in other)

    def issubset(self, other: Iterable[object]) -> bool:
        """Check if all elements are in other."""
        other_set = other if isinstance(other, set) else set(other)
        return all(value in other_set for value in self)

    def issuperset(self, other: Iterable[object]) -> bool:
        """Check if all elements of other are in this set."""
        return all(value in self for value in other)

    def __le__(self, other: object) -> bool:
        """Test if subset: self <= other."""
        return self.issubset(other)  # type: ignore[arg-type]

    def __ge__(self, other: object) -> bool:
        """Test if superset: self >= other."""
        return self.issuperset(other)  # type: ignore[arg-type]

    def __or__(self, other: object) -> set[V]:
        """Set union: self | other."""
        return set(self) | set(other)  # type: ignore[arg-type]

    def __and__(self, other: object) -> set[V]:
        """Set intersection: self & other."""
        return set(self) & set(other)  # type: ignore[arg-type]

    def __sub__(self, other: object) -> set[V]:
        """Set difference: self - other."""
        return set(self) - set(other)  # type: ignore[arg-type]

    def __xor__(self, other: object) -> set[V]:
        """Set symmetric difference: self ^ other."""
        return set(self) ^ set(other)  # type: ignore[arg-type]


class MutableSetBase[V](SetBase[V]):
    """Mutable set base. Adds remove/clear from add/discard.

    Additional abstract:
        add(value) -> None
        discard(value) -> None
    """

    @abstractmethod
    def add(self, value: V) -> None:
        """Add value to set."""
        ...

    @abstractmethod
    def discard(self, value: V) -> None:
        """Remove value if present, no error if absent."""
        ...

    def remove(self, value: V) -> None:
        """Remove value. Raises KeyError if not present."""
        if value not in self:
            raise KeyError(value)
        self.discard(value)

    def clear(self) -> None:
        """Remove all items. Override for bulk delete."""
        for value in list(self):
            self.discard(value)


class ReactiveSetBase[V](MutableSetBase[V]):
    """Reactive set base. Mutable set with change observation.

    Additional abstract:
        on_change() -> Subscription
    """

    @abstractmethod
    def on_change(self) -> object:
        """Subscribe to all changes in this set."""
        ...


@runtime_checkable
class ReactiveSetProtocol[V](Protocol):
    """Protocol: MutableSet + Observable.

    Use for type-checking views that support both mutation and change subscription.
    """

    def __contains__(self, value: object) -> bool: ...
    def __iter__(self) -> object: ...
    def __len__(self) -> int: ...
    def add(self, value: V) -> None: ...  # noqa: D102
    def discard(self, value: V) -> None: ...  # noqa: D102
    def on_change(self) -> object: ...  # noqa: D102
