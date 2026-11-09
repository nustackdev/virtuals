"""Set collection bases.

Mirror ``collections.abc.Set`` and ``MutableSet``: the same abstract core,
the same mixin methods and operators with the same semantics. Each base is
registered with its ABC, so every view built on it is a ``Set`` or
``MutableSet`` to ``isinstance``.

Abstract (must implement):
    __contains__, __iter__, __len__            (Set)
    add, discard                               (MutableSet)

Provided for free:
    <=, <, ==, >=, >, &, |, -, ^ (and reflected), isdisjoint
    remove, pop, clear, |=, &=, -=, ^=

Operators return a plain Python set (``_from_iterable``); like a Python
set, a view is not hashable. Set-only methods (``union``, ``issubset``,
``update``, ...) are not here: a view standing in for a set adds them.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable, MutableSet, Set
from typing import Protocol, TypeVar, runtime_checkable

from .bases import CollectionBase


__all__ = [
    "MutableSetBase",
    "ReactiveSetBase",
    "ReactiveSetProtocol",
    "SetBase",
]

V = TypeVar("V")


class SetBase(CollectionBase[V]):
    """Read-only set base, as ``collections.abc.Set``.

    All three abstract methods (__contains__, __iter__, __len__) inherited
    from CollectionBase: subclasses implement all of them.
    """

    @classmethod
    def _from_iterable(cls, it: Iterable[V]) -> set[V]:
        """What an operator returns: a plain set. Override for another type."""
        return set(it)

    def __le__(self, other: object) -> bool:
        """Subset: every item of ``self`` is in ``other``."""
        if not isinstance(other, Set):
            return NotImplemented
        if len(self) > len(other):
            return False
        return all(elem in other for elem in self)

    def __lt__(self, other: object) -> bool:
        """Proper subset."""
        if not isinstance(other, Set):
            return NotImplemented
        return len(self) < len(other) and self.__le__(other)

    def __gt__(self, other: object) -> bool:
        """Proper superset."""
        if not isinstance(other, Set):
            return NotImplemented
        return len(self) > len(other) and self.__ge__(other)

    def __ge__(self, other: object) -> bool:
        """Superset: every item of ``other`` is in ``self``."""
        if not isinstance(other, Set):
            return NotImplemented
        if len(self) < len(other):
            return False
        return all(elem in self for elem in other)

    def __eq__(self, other: object) -> bool:
        """Equal to any set holding the same items, as ``Set.__eq__``."""
        if not isinstance(other, Set):
            return NotImplemented
        return len(self) == len(other) and self.__le__(other)

    __hash__ = None  # type: ignore[assignment]

    def __and__(self, other: object) -> set[V]:
        """Intersection: ``self & other``."""
        if not isinstance(other, Iterable):
            return NotImplemented
        return self._from_iterable(value for value in other if value in self)

    __rand__ = __and__

    def isdisjoint(self, other: Iterable[object]) -> bool:
        """True if no item of ``other`` is in ``self``."""
        return not any(value in self for value in other)

    def __or__(self, other: object) -> set[V]:
        """Union: ``self | other``."""
        if not isinstance(other, Iterable):
            return NotImplemented
        return self._from_iterable([*self, *other])

    __ror__ = __or__

    def __sub__(self, other: object) -> set[V]:
        """Difference: ``self - other``."""
        if not isinstance(other, Set):
            if not isinstance(other, Iterable):
                return NotImplemented
            other = self._from_iterable(other)
        return self._from_iterable(value for value in self if value not in other)

    def __rsub__(self, other: object) -> set[V]:
        """Reflected difference: ``other - self``."""
        if not isinstance(other, Set):
            if not isinstance(other, Iterable):
                return NotImplemented
            other = self._from_iterable(other)
        return self._from_iterable(value for value in other if value not in self)

    def __xor__(self, other: object) -> set[V]:
        """Symmetric difference: ``self ^ other``."""
        if not isinstance(other, Set):
            if not isinstance(other, Iterable):
                return NotImplemented
            other = self._from_iterable(other)
        return (self - other) | (other - self)  # type: ignore[operator]

    __rxor__ = __xor__


class MutableSetBase(SetBase[V]):
    """Mutable set base, as ``collections.abc.MutableSet``.

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

    def pop(self) -> V:
        """Remove and return an arbitrary item (the first iteration yields).

        Raises:
            KeyError: If the set is empty.
        """
        it = iter(self)
        try:
            value = next(it)
        except StopIteration:
            raise KeyError("pop from an empty set") from None
        self.discard(value)
        return value

    def clear(self) -> None:
        """Remove all items. Override for bulk delete."""
        for value in list(self):
            self.discard(value)

    def __ior__(self, it: Iterable[V]) -> MutableSetBase[V]:
        """``self |= it``: add every item."""
        for value in it:
            self.add(value)
        return self

    def __iand__(self, it: Iterable[V]) -> MutableSetBase[V]:
        """``self &= it``: keep only the items also in ``it``."""
        for value in self - it:  # type: ignore[operator]
            self.discard(value)
        return self

    def __ixor__(self, it: Iterable[V]) -> MutableSetBase[V]:
        """``self ^= it``: keep the items in exactly one of the two."""
        if it is self:
            self.clear()
            return self
        if not isinstance(it, Set):
            it = self._from_iterable(it)
        for value in it:
            if value in self:
                self.discard(value)
            else:
                self.add(value)
        return self

    def __isub__(self, it: Iterable[V]) -> MutableSetBase[V]:
        """``self -= it``: drop every item of ``it``."""
        if it is self:
            self.clear()
            return self
        for value in it:
            self.discard(value)
        return self


class ReactiveSetBase(MutableSetBase[V]):
    """Reactive set base. Mutable set with change observation.

    Additional abstract:
        on_change() -> Subscription
    """

    @abstractmethod
    def on_change(self) -> object:
        """Subscribe to all changes in this set."""
        ...


Set.register(SetBase)
MutableSet.register(MutableSetBase)


@runtime_checkable
class ReactiveSetProtocol(Protocol[V]):
    """Protocol: MutableSet + Observable.

    Use for type-checking views that support both mutation and change subscription.
    """

    def __contains__(self, value: object) -> bool: ...
    def __iter__(self) -> object: ...
    def __len__(self) -> int: ...
    def add(self, value: V) -> None: ...  # noqa: D102
    def discard(self, value: V) -> None: ...  # noqa: D102
    def on_change(self) -> object: ...  # noqa: D102
