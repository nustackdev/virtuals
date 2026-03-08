"""Mapping collection bases.

Provides default implementations following collections.abc.MutableMapping pattern.
Subclasses implement the abstract core; the base derives everything else.

Abstract (must implement):
    __getitem__, __setitem__, __delitem__, __iter__, __len__

Provided for free:
    keys, values, items, get, pop, update, clear, __contains__
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Generator, Iterator, Mapping


__all__ = [
    "MappingBase",
    "MutableMappingBase",
    "ReactiveMappingBase",
]


class MappingBase[K, V]:
    """Read-only mapping base. Provides keys/values/items/get/__contains__ from core methods.

    Abstract:
        __getitem__(key) -> value
        __iter__() -> Iterator[key]
        __len__() -> int
    """

    @abstractmethod
    def __getitem__(self, key: K) -> V: ...

    @abstractmethod
    def __iter__(self) -> Iterator[K]: ...

    @abstractmethod
    def __len__(self) -> int: ...

    def __contains__(self, key: object) -> bool:
        """Check if key exists. Override for O(1) lookup."""
        try:
            self[key]  # type: ignore[index]
        except KeyError:
            return False
        return True

    def keys(self) -> Generator[K, None, None]:
        """Yield all keys."""
        yield from self

    def values(self) -> Generator[V, None, None]:
        """Yield all values."""
        for key in self:
            yield self[key]

    def items(self) -> Generator[tuple[K, V], None, None]:
        """Yield all (key, value) pairs."""
        for key in self:
            yield key, self[key]

    def get(self, key: K, default: object = None) -> V | object:
        """Get value with default fallback."""
        try:
            return self[key]
        except KeyError:
            return default


class MutableMappingBase[K, V](MappingBase[K, V]):
    """Mutable mapping base. Adds pop/update/clear from setitem/delitem.

    Additional abstract:
        __setitem__(key, value) -> None
        __delitem__(key) -> None
    """

    @abstractmethod
    def __setitem__(self, key: K, value: V) -> None: ...

    @abstractmethod
    def __delitem__(self, key: K) -> None: ...

    def pop(self, key: K, *args: object) -> V | object:
        """Remove and return value. Accepts optional default."""
        try:
            value = self[key]
            del self[key]
            return value
        except KeyError:
            if args:
                return args[0]
            raise

    def update(self, other: Mapping[K, V] | None = None, **kwargs: V) -> None:  # type: ignore[override]
        """Update from mapping or kwargs."""
        if other is not None:
            if hasattr(other, "items"):
                for key, value in other.items():
                    self[key] = value
            else:
                for key, value in other:  # type: ignore[union-attr]
                    self[key] = value
        for key, value in kwargs.items():
            self[key] = value  # type: ignore[index]

    def clear(self) -> None:
        """Remove all items. Override for bulk delete."""
        for key in list(self):
            del self[key]


class ReactiveMappingBase[K, V](MutableMappingBase[K, V]):
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
