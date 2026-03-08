"""Virtual collection toolkit.

Atomic protocols (abc):
    Containable, Sizeable, Subscriptable, Assignable, Deletable, ...
    + is_* type guard functions

Collection bases (mapping, sequence, set):
    MappingBase → MutableMappingBase
    SequenceBase → MutableSequenceBase
    SetBase → MutableSetBase

Reactive protocols (abc):
    ReactiveMappingView, ReactiveSequenceView, ReactiveSetView
"""

from __future__ import annotations

from .abc import (
    Addable,
    Appendable,
    Assignable,
    ChildObservable,
    Clearable,
    Containable,
    Convertible,
    Deletable,
    DescendantsObservable,
    Discardable,
    Initializable,
    Insertable,
    Nestable,
    Observable,
    Poppable,
    ReactiveMappingView,
    ReactiveSequenceView,
    ReactiveSetView,
    Removable,
    Sizeable,
    Subscriptable,
    is_addable,
    is_appendable,
    is_assignable,
    is_child_observable,
    is_clearable,
    is_containable,
    is_convertible,
    is_deletable,
    is_descendants_observable,
    is_discardable,
    is_initializable,
    is_insertable,
    is_nestable,
    is_observable,
    is_poppable,
    is_removable,
    is_sizeable,
    is_subscriptable,
)
from .mapping import MappingBase, MutableMappingBase, ReactiveMappingBase
from .sequence import MutableSequenceBase, ReactiveSequenceBase, SequenceBase
from .set import MutableSetBase, ReactiveSetBase, SetBase


__all__ = [
    # Atomic protocols
    "Addable",
    "Appendable",
    "Assignable",
    "ChildObservable",
    "Clearable",
    "Containable",
    "Convertible",
    "Deletable",
    "DescendantsObservable",
    "Discardable",
    "Initializable",
    "Insertable",
    # Collection bases
    "MappingBase",
    "MutableMappingBase",
    "MutableSequenceBase",
    "MutableSetBase",
    "Nestable",
    "Observable",
    "Poppable",
    # Reactive bases
    "ReactiveMappingBase",
    # Reactive protocols
    "ReactiveMappingView",
    "ReactiveSequenceBase",
    "ReactiveSequenceView",
    "ReactiveSetBase",
    "ReactiveSetView",
    "Removable",
    "SequenceBase",
    "SetBase",
    "Sizeable",
    "Subscriptable",
    # Type guards
    "is_addable",
    "is_appendable",
    "is_assignable",
    "is_child_observable",
    "is_clearable",
    "is_containable",
    "is_convertible",
    "is_deletable",
    "is_descendants_observable",
    "is_discardable",
    "is_initializable",
    "is_insertable",
    "is_nestable",
    "is_observable",
    "is_poppable",
    "is_removable",
    "is_sizeable",
    "is_subscriptable",
]
