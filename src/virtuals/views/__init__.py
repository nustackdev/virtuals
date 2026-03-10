"""Collection of standard views."""

from __future__ import annotations

from virtuals._views import (
    ByteArrayView,
    DictViewBase,
    EagerDictView,
    EagerIndexedDictView,
    EagerListView,
    FlatDictView,
    FlatListView,
    FrozenSetView,
    IndexedDictViewBase,
    LazyDictView,
    LazyIndexedDictView,
    LazyListView,
    LightDictView,
    ListSliceView,
    ListViewBase,
    SetView,
    TupleView,
)


DictView = LazyDictView
IndexedDictView = LazyIndexedDictView
ListView = LazyListView

__all__ = (
    "ByteArrayView",
    "DictView",
    "DictViewBase",
    "EagerDictView",
    "EagerIndexedDictView",
    "EagerListView",
    "FlatDictView",
    "FlatListView",
    "FrozenSetView",
    "IndexedDictView",
    "IndexedDictViewBase",
    "LazyDictView",
    "LazyIndexedDictView",
    "LazyListView",
    "LightDictView",
    "ListSliceView",
    "ListView",
    "ListViewBase",
    "SetView",
    "TupleView",
)
