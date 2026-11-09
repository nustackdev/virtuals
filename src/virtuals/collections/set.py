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
set, a view is not hashable.

``FrozenSetMethods`` and ``SetMethods`` add what ``frozenset`` and ``set``
have beyond the ABCs (``union``, ``issubset``, ``copy``, ``update``, ...)
for a view that stands in for one of them.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable, MutableSet, Set
from typing import Protocol, TypeVar, runtime_checkable

from .bases import CollectionBase


__all__ = [
    "FrozenSetMethods",
    "MutableSetBase",
    "ReactiveSetBase",
    "ReactiveSetProtocol",
    "SetBase",
    "SetMethods",
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


class FrozenSetMethods(SetBase[V]):
    """What ``frozenset`` has beyond ``Set``: the named methods, ``copy``, and its operator rules.

    ``union``, ``intersection``, ``difference``, ``symmetric_difference``,
    ``issubset``, ``issuperset`` take any iterables, as on ``frozenset``.
    The operators ``& | - ^`` take a set, a frozenset or another set view
    and give ``TypeError`` for anything else, as on ``frozenset`` (the
    ``Set`` mixins take any iterable). New values come back as the plain
    type ``_from_iterable`` names; a reflected operator (``{1} | view``)
    gives the type of the left operand, as ``set | frozenset`` does.

    Not hashable, unlike ``frozenset``: the stored members can be replaced
    under the view, so a hash would not hold for the view's lifetime.
    """

    def _is_set_operand(self, other: object) -> bool:
        return isinstance(other, set | frozenset | FrozenSetMethods)

    @staticmethod
    def _reflected(other: set[V] | frozenset[V], items: Iterable[V]) -> set[V] | frozenset[V]:
        """A reflected operator's result: the left operand's plain type."""
        return set(items) if isinstance(other, set) else frozenset(items)

    def copy(self) -> set[V] | frozenset[V]:
        """A plain copy of the members, as ``set.copy``."""
        return self._from_iterable(self)

    def union(self, *others: Iterable[V]) -> set[V] | frozenset[V]:
        """Members of the view or any of ``others``."""
        return self.copy().union(*others)

    def intersection(self, *others: Iterable[V]) -> set[V] | frozenset[V]:
        """Members of the view found in every one of ``others``."""
        return self.copy().intersection(*others)

    def difference(self, *others: Iterable[V]) -> set[V] | frozenset[V]:
        """Members of the view found in none of ``others``."""
        return self.copy().difference(*others)

    def symmetric_difference(self, other: Iterable[V]) -> set[V] | frozenset[V]:
        """Members of exactly one of the view and ``other``."""
        return self.copy().symmetric_difference(other)

    def issubset(self, other: Iterable[object]) -> bool:
        """True if every member is in ``other``, as ``set.issubset``."""
        if not isinstance(other, set | frozenset):
            other = set(other)
        return len(self) <= len(other) and all(value in other for value in self)

    def issuperset(self, other: Iterable[object]) -> bool:
        """True if every item of ``other`` is a member, as ``set.issuperset``."""
        return all(value in self for value in other)

    def __and__(self, other: object) -> set[V] | frozenset[V]:
        return super().__and__(other) if self._is_set_operand(other) else NotImplemented

    def __or__(self, other: object) -> set[V] | frozenset[V]:
        return super().__or__(other) if self._is_set_operand(other) else NotImplemented

    def __sub__(self, other: object) -> set[V] | frozenset[V]:
        return super().__sub__(other) if self._is_set_operand(other) else NotImplemented

    def __xor__(self, other: object) -> set[V] | frozenset[V]:
        return super().__xor__(other) if self._is_set_operand(other) else NotImplemented

    def __rand__(self, other: object) -> set[V] | frozenset[V]:
        if not isinstance(other, set | frozenset):
            return NotImplemented
        return self._reflected(other, (value for value in other if value in self))

    def __ror__(self, other: object) -> set[V] | frozenset[V]:
        if not isinstance(other, set | frozenset):
            return NotImplemented
        return self._reflected(other, [*other, *self])

    def __rsub__(self, other: object) -> set[V] | frozenset[V]:
        if not isinstance(other, set | frozenset):
            return NotImplemented
        return self._reflected(other, (value for value in other if value not in self))

    def __rxor__(self, other: object) -> set[V] | frozenset[V]:
        if not isinstance(other, set | frozenset):
            return NotImplemented
        return self._reflected(other, self.copy() ^ other)


class SetMethods(MutableSetBase[V], FrozenSetMethods[V]):
    """What ``set`` has beyond ``MutableSet``: the ``frozenset`` methods plus the in-place ones.

    ``update``, ``intersection_update``, ``difference_update``,
    ``symmetric_difference_update`` take any iterables, as on ``set``. The
    in-place operators ``|= &= -= ^=`` take a set, a frozenset or another
    set view, and give ``TypeError`` for anything else, as on ``set``.
    """

    def update(self, *others: Iterable[V]) -> None:
        """Add every item of every one of ``others``."""
        for other in others:
            for value in list(other) if other is self else other:
                self.add(value)

    def intersection_update(self, *others: Iterable[V]) -> None:
        """Keep only the members found in every one of ``others``."""
        keep = self.intersection(*others)
        for value in [value for value in self if value not in keep]:
            self.discard(value)

    def difference_update(self, *others: Iterable[V]) -> None:
        """Drop every member found in any of ``others``."""
        for other in others:
            if other is self:
                self.clear()
                return
            for value in other:
                self.discard(value)

    def symmetric_difference_update(self, other: Iterable[V]) -> None:
        """Keep the members of exactly one of the view and ``other``."""
        MutableSetBase.__ixor__(self, other if other is self else set(other))

    def __ior__(self, other: Iterable[V]) -> SetMethods[V]:
        return super().__ior__(other) if self._is_set_operand(other) else NotImplemented

    def __iand__(self, other: Iterable[V]) -> SetMethods[V]:
        return super().__iand__(other) if self._is_set_operand(other) else NotImplemented

    def __isub__(self, other: Iterable[V]) -> SetMethods[V]:
        return super().__isub__(other) if self._is_set_operand(other) else NotImplemented

    def __ixor__(self, other: Iterable[V]) -> SetMethods[V]:
        return super().__ixor__(other) if self._is_set_operand(other) else NotImplemented


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
