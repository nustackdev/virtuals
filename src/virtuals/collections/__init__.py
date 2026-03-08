"""Virtual collection protocols.

Atomic capabilities (abc) compose into collection hierarchies:
    Container → Collection → Mapping/Sequence/Set → Mutable* → Reactive*

Usage:
    from virtuals.collections import MutableMappingView, ReactiveSequenceView
    from virtuals.collections import Convertible, is_convertible
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
from .collections import (
    CollectionView,
    ContainerView,
    MappingView,
    MutableMappingView,
    MutableSequenceView,
    MutableSetView,
    ReactiveMappingView,
    ReactiveSequenceView,
    ReactiveSetView,
    SequenceView,
    SetView,
)


__all__ = [
    # Atomic protocols
    "Addable",
    "Appendable",
    "Assignable",
    "ChildObservable",
    "Clearable",
    # Collection hierarchy
    "CollectionView",
    "Containable",
    "ContainerView",
    "Convertible",
    "Deletable",
    "DescendantsObservable",
    "Discardable",
    "Initializable",
    "Insertable",
    "MappingView",
    "MutableMappingView",
    "MutableSequenceView",
    "MutableSetView",
    "Nestable",
    "Observable",
    "Poppable",
    "ReactiveMappingView",
    "ReactiveSequenceView",
    "ReactiveSetView",
    "Removable",
    "SequenceView",
    "SetView",
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
