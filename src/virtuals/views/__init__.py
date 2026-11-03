"""Collection of standard views."""

from __future__ import annotations

from virtuals._views import (
    ByteArrayView,
    DictViewBase,
    EagerDictView,
    EagerFractionalIndexedDictView,
    EagerIndexedDictView,
    EagerKh57View,
    EagerListView,
    EagerLogIndexedDictView,
    FlatDictView,
    FlatListView,
    FractionalIndexedDictViewBase,
    FrozenSetView,
    IndexedDictViewBase,
    Kh57ViewBase,
    LazyDictView,
    LazyFractionalIndexedDictView,
    LazyIndexedDictView,
    LazyKh57View,
    LazyListView,
    LazyLogIndexedDictView,
    LightDictView,
    ListViewBase,
    LogIndexedDictViewBase,
    SetView,
    TupleView,
)


DictView = LazyDictView
FractionalIndexedDictView = LazyFractionalIndexedDictView
IndexedDictView = LazyIndexedDictView
Kh57View = LazyKh57View
LogIndexedDictView = LazyLogIndexedDictView
ListView = LazyListView

__all__ = (
    "ByteArrayView",
    "DictView",
    "DictViewBase",
    "EagerDictView",
    "EagerFractionalIndexedDictView",
    "EagerIndexedDictView",
    "EagerKh57View",
    "EagerListView",
    "EagerLogIndexedDictView",
    "FlatDictView",
    "FlatListView",
    "FractionalIndexedDictView",
    "FractionalIndexedDictViewBase",
    "FrozenSetView",
    "IndexedDictView",
    "IndexedDictViewBase",
    "Kh57View",
    "Kh57ViewBase",
    "LazyDictView",
    "LazyFractionalIndexedDictView",
    "LazyIndexedDictView",
    "LazyKh57View",
    "LazyListView",
    "LazyLogIndexedDictView",
    "LightDictView",
    "ListView",
    "ListViewBase",
    "LogIndexedDictView",
    "LogIndexedDictViewBase",
    "SetView",
    "TupleView",
)
