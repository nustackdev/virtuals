"""Mapping collection bases.

Mirror ``collections.abc.Mapping`` and ``MutableMapping``: the same
abstract core, the same mixin methods with the same semantics. Each base
is registered with its ABC, so every view built on it is a ``Mapping`` or
``MutableMapping`` to ``isinstance``.

Abstract (must implement):
    __getitem__, __iter__, __len__             (Mapping)
    __setitem__, __delitem__                   (MutableMapping)

Provided for free:
    __contains__, keys, values, items, get, __eq__
    pop, popitem, setdefault, update, clear

Like a Python mapping, a view equals any mapping with the same items and
is not hashable. Dict-only methods (``|``, ``|=``, ``copy``) are not here.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import ItemsView, KeysView, Mapping, MutableMapping, ValuesView
from typing import Generic, Protocol, TypeVar, runtime_checkable

from .bases import CollectionBase


__all__ = [
    "MappingBase",
    "MutableMappingBase",
    "ReactiveMappingBase",
    "ReactiveMappingProtocol",
]

K = TypeVar("K")
V = TypeVar("V")

_MISSING = object()


class MappingBase(CollectionBase[K], Generic[K, V]):
    """Read-only mapping base, as ``collections.abc.Mapping``.

    Inherits abstract __iter__, __len__ from CollectionBase.

    Abstract:
        __getitem__(key) -> value
    """

    @abstractmethod
    def __getitem__(self, key: K) -> V: ...

    def __contains__(self, key: object) -> bool:
        """Check if key exists. Override for O(1) lookup."""
        try:
            self[key]  # type: ignore[index]
        except KeyError:
            return False
        return True

    def keys(self) -> KeysView[K]:
        """Get all keys as a set-like view."""
        return KeysView(self)  # type: ignore[arg-type]

    def values(self) -> ValuesView[V]:
        """Get all values as a collection view."""
        return ValuesView(self)  # type: ignore[arg-type]

    def items(self) -> ItemsView[K, V]:
        """Get all items as a set-like view."""
        return ItemsView(self)  # type: ignore[arg-type]

    def get(self, key: K, default: object = None) -> V | object:
        """Value at ``key``, or ``default`` (None) when the key is absent."""
        try:
            return self[key]
        except KeyError:
            return default

    def __eq__(self, other: object) -> bool:
        """Equal to any mapping holding the same items, as ``Mapping.__eq__``."""
        if not isinstance(other, Mapping):
            return NotImplemented
        return dict(self.items()) == dict(other.items())

    __hash__ = None  # type: ignore[assignment]


class MutableMappingBase(MappingBase[K, V]):
    """Mutable mapping base, as ``collections.abc.MutableMapping``.

    Additional abstract:
        __setitem__(key, value) -> None
        __delitem__(key) -> None
    """

    @abstractmethod
    def __setitem__(self, key: K, value: V) -> None: ...

    @abstractmethod
    def __delitem__(self, key: K) -> None: ...

    def _plain_value(self, key: K) -> V:
        """The value a removal hands back: the read by default.

        A mapping whose reads hand out live child views overrides it to
        read the plain value, since the child is gone after the delete.
        """
        return self[key]

    def pop(self, key: K, default: object = _MISSING) -> V | object:
        """Remove ``key`` and return its value; ``default`` if absent, else KeyError."""
        try:
            value = self._plain_value(key)
        except KeyError:
            if default is _MISSING:
                raise
            return default
        del self[key]
        return value

    def popitem(self) -> tuple[K, V]:
        """Remove and return a ``(key, value)`` pair: the last in order, as ``dict.popitem``.

        A mapping with no order (no ``__reversed__``) gives the first its
        iteration yields, as ``MutableMapping.popitem``.

        Raises:
            KeyError: If the mapping is empty.
        """
        ordered = getattr(self, "__reversed__", None) is not None
        keys = reversed(self) if ordered else iter(self)  # type: ignore[call-overload]
        try:
            key = next(keys)
        except StopIteration:
            raise KeyError("popitem(): mapping is empty") from None
        value = self._plain_value(key)
        del self[key]
        return key, value

    def setdefault(self, key: K, default: object = None) -> V | object:
        """``self[key]`` if present, else store ``default`` there and return it read back."""
        try:
            return self[key]
        except KeyError:
            self[key] = default  # type: ignore[assignment]
        return self[key]

    def update(self, other: object = (), /, **kwargs: V) -> None:
        """Update from a mapping, an object with ``keys()``, or ``(key, value)`` pairs, then kwargs."""
        for key, value in self._update_items(other, kwargs).items():
            self[key] = value

    @staticmethod
    def _update_items(other: object, kwargs: dict[str, V]) -> dict[K, V]:
        """What ``update(other, **kwargs)`` writes, in order, as one dict: last write wins.

        ``other`` is read the way ``dict.update`` reads it: a mapping, an
        object with ``keys()`` and ``[]``, or an iterable of pairs.
        """
        if isinstance(other, Mapping):
            items = {key: other[key] for key in other}
        elif hasattr(other, "keys"):
            items = {key: other[key] for key in other.keys()}  # type: ignore[index]
        else:
            items = {}
            for pair in other:  # type: ignore[attr-defined]
                key, value = pair
                items[key] = value
        items.update(kwargs)
        return items  # type: ignore[return-value]

    def clear(self) -> None:
        """Remove all items. Override for bulk delete."""
        for key in list(self):
            del self[key]


class ReactiveMappingBase(MutableMappingBase[K, V]):
    """Reactive mapping base. Mutable mapping with change observation.

    Additional abstract:
        on_change() -> Subscription
        on_child_change(key) -> Subscription
        on_children_change() -> Subscription
    """

    @abstractmethod
    def on_change(self) -> object:
        """Subscribe to all changes in this mapping."""
        ...

    @abstractmethod
    def on_child_change(self, address: K) -> object:
        """Subscribe to changes on a specific key."""
        ...

    @abstractmethod
    def on_children_change(self) -> object:
        """Subscribe to changes on any key."""
        ...


Mapping.register(MappingBase)
MutableMapping.register(MutableMappingBase)


@runtime_checkable
class ReactiveMappingProtocol(Protocol[K, V]):
    """Protocol: MutableMapping + Observable + ChildObservable.

    Use for type-checking views that support both mutation and change subscription.
    """

    def __getitem__(self, key: K) -> V: ...
    def __setitem__(self, key: K, value: V) -> None: ...
    def __delitem__(self, key: K) -> None: ...
    def __iter__(self) -> object: ...
    def __len__(self) -> int: ...
    def on_change(self) -> object: ...  # noqa: D102
    def on_child_change(self, address: K) -> object: ...  # noqa: D102
    def on_children_change(self) -> object: ...  # noqa: D102
