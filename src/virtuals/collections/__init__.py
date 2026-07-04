"""Virtual collection toolkit.

Fundamental bases (bases):
    ContainerBase — abstract __contains__
    IterableBase  — abstract __iter__
    SizedBase     — abstract __len__
    CollectionBase(ContainerBase, IterableBase, SizedBase)

Protocols (protocols):
    Convertible, Initializable, Nestable — conversion & navigation
    Subscriptable, Assignable, Containable, Sizeable, Deletable, Clearable — access
    Observable, ChildObservable, DescendantsObservable — observation
    + is_* type guard functions

Collection bases (mapping, sequence, set):
    MappingBase → MutableMappingBase → ReactiveMappingBase
    SequenceBase → MutableSequenceBase → ReactiveSequenceBase
    SetBase → MutableSetBase → ReactiveSetBase

Reactive protocols (mapping, sequence, set):
    ReactiveMappingProtocol, ReactiveSequenceProtocol, ReactiveSetProtocol
"""

from __future__ import annotations

from .bases import CollectionBase, ContainerBase, IterableBase, SizedBase
from .mapping import (
    MappingBase,
    MutableMappingBase,
    ReactiveMappingBase,
    ReactiveMappingProtocol,
)
from .protocols import (
    Assignable,
    ChildObservable,
    Clearable,
    Containable,
    Convertible,
    Deletable,
    DescendantsObservable,
    Initializable,
    Nestable,
    Observable,
    Sampleable,
    Sizeable,
    Subscriptable,
    is_assignable,
    is_child_observable,
    is_clearable,
    is_containable,
    is_convertible,
    is_deletable,
    is_descendants_observable,
    is_initializable,
    is_nestable,
    is_observable,
    is_sampleable,
    is_sizeable,
    is_subscriptable,
)
from .sequence import (
    MutableSequenceBase,
    ReactiveSequenceBase,
    ReactiveSequenceProtocol,
    SequenceBase,
)
from .set import MutableSetBase, ReactiveSetBase, ReactiveSetProtocol, SetBase


__all__ = [
    "Assignable",
    "ChildObservable",
    "Clearable",
    # Fundamental bases
    "CollectionBase",
    # Protocols
    "Containable",
    "ContainerBase",
    "Convertible",
    "Deletable",
    "DescendantsObservable",
    "Initializable",
    "IterableBase",
    # Collection bases
    "MappingBase",
    "MutableMappingBase",
    "MutableSequenceBase",
    "MutableSetBase",
    "Nestable",
    "Observable",
    # Reactive bases
    "ReactiveMappingBase",
    # Reactive protocols
    "ReactiveMappingProtocol",
    "ReactiveSequenceBase",
    "ReactiveSequenceProtocol",
    "ReactiveSetBase",
    "ReactiveSetProtocol",
    "Sampleable",
    "SequenceBase",
    "SetBase",
    "Sizeable",
    "SizedBase",
    "Subscriptable",
    # Type guards
    "is_assignable",
    "is_child_observable",
    "is_clearable",
    "is_containable",
    "is_convertible",
    "is_deletable",
    "is_descendants_observable",
    "is_initializable",
    "is_nestable",
    "is_observable",
    "is_sampleable",
    "is_sizeable",
    "is_subscriptable",
]
