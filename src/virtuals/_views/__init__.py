"""Collection of standard views."""

from __future__ import annotations

from .bytearray_view import ByteArrayView
from .dict_view import DictViewBase, EagerDictView, LazyDictView
from .flat_dict_view import FlatDictView
from .flat_list_view import FlatListView
from .fractional_indexed_dict_view import (
    EagerFractionalIndexedDictView,
    FractionalIndexedDictViewBase,
    LazyFractionalIndexedDictView,
)
from .frozenset_view import FrozenSetView
from .indexed_dict_view import EagerIndexedDictView, IndexedDictViewBase, LazyIndexedDictView
from .kh57_view import EagerKh57View, Kh57ViewBase, LazyKh57View
from .light_dict_view import LightDictView
from .list_view import EagerListView, LazyListView, ListViewBase
from .log_indexed_dict_view import (
    EagerLogIndexedDictView,
    LazyLogIndexedDictView,
    LogIndexedDictViewBase,
)
from .ring_view import RingView
from .set_view import SetView
from .tuple_view import TupleView


BUILTIN_VIEWS = (
    ByteArrayView,
    EagerDictView,
    EagerFractionalIndexedDictView,
    EagerIndexedDictView,
    EagerKh57View,
    EagerListView,
    EagerLogIndexedDictView,
    FlatDictView,
    FlatListView,
    FrozenSetView,
    LightDictView,
    RingView,
    SetView,
    TupleView,
)
"""Every built-in view, one per stored structure: what a stored container is read back with."""


__all__ = (
    "ByteArrayView",
    "DictViewBase",
    "EagerDictView",
    "EagerFractionalIndexedDictView",
    "EagerIndexedDictView",
    "EagerKh57View",
    "EagerListView",
    "EagerLogIndexedDictView",
    "FlatDictView",
    "FlatListView",
    "FractionalIndexedDictViewBase",
    "FrozenSetView",
    "IndexedDictViewBase",
    "Kh57ViewBase",
    "LazyDictView",
    "LazyFractionalIndexedDictView",
    "LazyIndexedDictView",
    "LazyKh57View",
    "LazyListView",
    "LazyLogIndexedDictView",
    "LightDictView",
    "ListViewBase",
    "LogIndexedDictViewBase",
    "RingView",
    "SetView",
    "TupleView",
)
