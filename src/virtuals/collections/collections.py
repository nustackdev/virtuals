"""Collection protocol hierarchy.

Composes atomic capabilities from abc into complete collection interfaces.
Follows Python's collections.abc hierarchy with a Reactive tier for observability.

Hierarchy:
    Container → Collection → Mapping    → MutableMapping    → ReactiveMapping
                           → Sequence   → MutableSequence   → ReactiveSequence
                           → Set        → MutableSet        → ReactiveSet
"""

from __future__ import annotations

from collections.abc import Iterable as PyIterable
from collections.abc import Mapping as PyMapping
from collections.abc import Set as PySet
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from virtuals.collections.abc import (
    Appendable,
    Assignable,
    ChildObservable,
    Clearable,
    Containable,
    Convertible,
    Deletable,
    Initializable,
    Observable,
    Sizeable,
    Subscriptable,
)


if TYPE_CHECKING:
    from virtuals.types import Empty


__all__ = [
    "CollectionView",
    # Base
    "ContainerView",
    # Mapping
    "MappingView",
    "MutableMappingView",
    "MutableSequenceView",
    "MutableSetView",
    "ReactiveMappingView",
    "ReactiveSequenceView",
    "ReactiveSetView",
    # Sequence
    "SequenceView",
    # Set
    "SetView",
]


# =============================================================================
# BASE
# =============================================================================


@runtime_checkable
class ContainerView[V](
    Containable[V],
    Protocol,
):
    """Base protocol for all collections. Supports membership testing.

    Mirrors collections.abc.Container.
    """

    pass


@runtime_checkable
class CollectionView[V](
    ContainerView[V],
    Sizeable,
    Protocol,
):
    """Sized container. Supports membership testing and size queries.

    Mirrors collections.abc.Collection (minus __iter__ which views handle directly).
    """

    pass


# =============================================================================
# MAPPING
# =============================================================================


@runtime_checkable
class MappingView[K, V](
    CollectionView[K],
    Subscriptable[K, V],
    Convertible[PyMapping[K, V]],
    Protocol,
):
    """Read-only mapping. Supports key lookup, iteration, and extraction.

    Mirrors collections.abc.Mapping.
    """

    def keys(self) -> PyIterable[K]:
        """Get all keys."""
        ...

    def values(self) -> PyIterable[V]:
        """Get all values."""
        ...

    def items(self) -> PyIterable[tuple[K, V]]:
        """Get all key-value pairs."""
        ...


@runtime_checkable
class MutableMappingView[K, V](
    MappingView[K, V],
    Assignable[K, V],
    Deletable[K],
    Clearable,
    Initializable[PyMapping[K, V]],
    Protocol,
):
    """Mutable mapping. Adds assignment, deletion, clearing, and bulk update.

    Mirrors collections.abc.MutableMapping.
    """

    def update(self, other: PyMapping[K, V]) -> None:
        """Update mapping with key-value pairs from other."""
        ...


@runtime_checkable
class ReactiveMappingView[K, V](
    MutableMappingView[K, V],
    Observable,
    ChildObservable[K],
    Protocol,
):
    """Reactive mapping. Adds change observation to mutable mapping.

    Use when consumers need to subscribe to key-level or whole-view changes.
    """

    pass


# =============================================================================
# SEQUENCE
# =============================================================================


@runtime_checkable
class SequenceView[V](
    CollectionView[int],
    Subscriptable[int, V],
    Convertible[PyIterable[V]],
    Protocol,
):
    """Read-only sequence. Supports indexed access and extraction.

    Mirrors collections.abc.Sequence.
    """

    pass


@runtime_checkable
class MutableSequenceView[V](
    SequenceView[V],
    Assignable[int, V],
    Deletable[int],
    Appendable[V],
    Clearable,
    Initializable[PyIterable[V]],
    Protocol,
):
    """Mutable sequence. Adds assignment, deletion, appending, insert, and pop.

    Mirrors collections.abc.MutableSequence.
    """

    def insert(self, address: int, value: V) -> None:
        """Insert value at index."""
        ...

    def pop(self, address: int = -1) -> V | Empty:
        """Remove and return item at index."""
        ...


@runtime_checkable
class ReactiveSequenceView[V](
    MutableSequenceView[V],
    Observable,
    ChildObservable[int],
    Protocol,
):
    """Reactive sequence. Adds change observation to mutable sequence.

    Use when consumers need to subscribe to index-level or whole-view changes.
    """

    pass


# =============================================================================
# SET
# =============================================================================


@runtime_checkable
class SetView[V](
    CollectionView[V],
    Convertible[PySet[V]],
    Protocol,
):
    """Read-only set. Supports membership testing, size, and set algebra.

    Mirrors collections.abc.Set.
    """

    def isdisjoint(self, other: PySet[V]) -> bool:
        """Check if no elements in common with other."""
        ...

    def issubset(self, other: PySet[V]) -> bool:
        """Check if all elements are in other."""
        ...

    def issuperset(self, other: PySet[V]) -> bool:
        """Check if all elements of other are in this set."""
        ...


@runtime_checkable
class MutableSetView[V](
    SetView[V],
    Clearable,
    Initializable[PySet[V]],
    Protocol,
):
    """Mutable set. Adds add, remove, and clearing.

    Mirrors collections.abc.MutableSet.
    """

    def add(self, value: V) -> None:
        """Add value to set."""
        ...

    def remove(self, value: V) -> None:
        """Remove value from set. Raises KeyError if not present."""
        ...


@runtime_checkable
class ReactiveSetView[V](
    MutableSetView[V],
    Observable,
    Protocol,
):
    """Reactive set. Adds change observation to mutable set.

    Use when consumers need to subscribe to set changes.
    """

    pass
