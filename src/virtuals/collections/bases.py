"""Fundamental collection bases.

Mirrors Python's collections.abc hierarchy at the most basic level:

    ContainerBase — abstract __contains__
    IterableBase  — abstract __iter__
    SizedBase     — abstract __len__
    CollectionBase(ContainerBase, IterableBase, SizedBase) — combines all three

Concrete collection types (Mapping, Sequence, Set) inherit from CollectionBase.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterator


__all__ = [
    "CollectionBase",
    "ContainerBase",
    "IterableBase",
    "SizedBase",
]


class ContainerBase:
    """Base for containers supporting membership testing via ``in``.

    Abstract:
        __contains__(value) -> bool
    """

    @abstractmethod
    def __contains__(self, value: object) -> bool: ...


class IterableBase[V]:
    """Base for containers supporting iteration.

    Abstract:
        __iter__() -> Iterator[V]
    """

    @abstractmethod
    def __iter__(self) -> Iterator[V]: ...


class SizedBase:
    """Base for containers supporting ``len()``.

    Abstract:
        __len__() -> int
    """

    @abstractmethod
    def __len__(self) -> int: ...


class CollectionBase[V](ContainerBase, IterableBase[V], SizedBase):
    """Base combining Container + Iterable + Sized.

    All concrete collection types (Mapping, Sequence, Set) inherit from this.
    Subclasses may override __contains__/__iter__ with concrete implementations
    derived from more specific abstract methods (e.g. __getitem__).
    """
