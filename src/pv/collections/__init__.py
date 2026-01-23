"""View capabilities.

- Definitions for common capabilities of pythonic containers
- Definitions for common collections of pythonic containers
"""

from __future__ import annotations

from .collections import (
    CollectionView,
    ContainerView,
    MappingView,
    MutableMappingView,
    MutableSequenceView,
    MutableSetView,
    SequenceView,
    SetView,
)


__all__ = [
    "CollectionView",
    "ContainerView",
    "MappingView",
    "MutableMappingView",
    "MutableSequenceView",
    "MutableSetView",
    "SequenceView",
    "SetView",
]
