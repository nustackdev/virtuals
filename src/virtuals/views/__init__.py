"""Collection of standard views."""

from __future__ import annotations

from virtuals._views import (
    ByteArrayView,
    DictViewBase,
    EagerDictView,
    EagerIndexedDictView,
    EagerListView,
    EagerLogIndexedDictView,
    FlatDictView,
    FlatListView,
    FrozenSetView,
    IndexedDictViewBase,
    LazyDictView,
    LazyIndexedDictView,
    LazyListView,
    LazyLogIndexedDictView,
    LightDictView,
    ListViewBase,
    LogIndexedDictViewBase,
    SetView,
    TupleView,
)


DictView = LazyDictView
IndexedDictView = LazyIndexedDictView
LogIndexedDictView = LazyLogIndexedDictView
ListView = LazyListView

__all__ = (
    "ByteArrayView",
    "DictView",
    "DictViewBase",
    "EagerDictView",
    "EagerIndexedDictView",
    "EagerListView",
    "EagerLogIndexedDictView",
    "FlatDictView",
    "FlatListView",
    "FrozenSetView",
    "IndexedDictView",
    "IndexedDictViewBase",
    "LazyDictView",
    "LazyIndexedDictView",
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
