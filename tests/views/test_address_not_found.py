"""A view's own answer for a missing key or index: one narrow not-found error.

A point read or a delete that finds nothing at its address raises
``KeyNotFoundError`` on a mapping, ``IndexNotFoundError`` on a sequence. Both
are ``AddressNotFoundError``, and still the ``KeyError`` / ``IndexError``
Python code expects, so a caller can tell the view's absence apart from a
storage fault or a container that is gone, which are ``KeyError``s too.
"""

from __future__ import annotations

import pytest

from virtuals._views import (
    ByteArrayView,
    EagerDictView,
    EagerFractionalIndexedDictView,
    EagerIndexedDictView,
    EagerKh57View,
    EagerListView,
    EagerLogIndexedDictView,
    FlatDictView,
    FlatListView,
    LazyDictView,
    LazyFractionalIndexedDictView,
    LazyIndexedDictView,
    LazyKh57View,
    LazyListView,
    LazyLogIndexedDictView,
    LightDictView,
    RingView,
    TupleView,
)
from virtuals.container import ContainerNotFoundError
from virtuals.tkv.storage.exceptions import StorageError, StorageKeyError
from virtuals.view import AddressNotFoundError, IndexNotFoundError, KeyNotFoundError


MAPPINGS = (
    EagerDictView,
    LazyDictView,
    FlatDictView,
    LightDictView,
    EagerIndexedDictView,
    LazyIndexedDictView,
    EagerLogIndexedDictView,
    LazyLogIndexedDictView,
    EagerFractionalIndexedDictView,
    LazyFractionalIndexedDictView,
    EagerKh57View,
    LazyKh57View,
)
SEQUENCES = (EagerListView, LazyListView, FlatListView, ByteArrayView, TupleView, RingView.of(3))
KH57 = (EagerKh57View, LazyKh57View)


def _key(view_cls):
    return 1 if view_cls in KH57 else "k"


def _missing(view_cls):
    return 2 if view_cls in KH57 else "gone"


def _fill(view):
    cls = type(view)
    if cls in MAPPINGS:
        view[_key(cls)] = 1
    elif cls is TupleView:
        view.store([7])
    else:
        view.append(7)


def test_the_narrow_errors_are_python_lookups_not_storage_errors() -> None:
    assert issubclass(KeyNotFoundError, KeyError)
    assert issubclass(IndexNotFoundError, IndexError)
    for narrow in (KeyNotFoundError, IndexNotFoundError):
        assert issubclass(narrow, AddressNotFoundError)
        assert not issubclass(narrow, StorageError)
    assert not issubclass(StorageKeyError, AddressNotFoundError)
    assert not issubclass(ContainerNotFoundError, AddressNotFoundError)


@pytest.mark.parametrize("view_cls", MAPPINGS)
@pytest.mark.parametrize("written", (False, True))
def test_a_mapping_read_of_a_missing_key_raises_key_not_found(root_view, view_cls, written):
    view = root_view.open_child("m", view_cls)
    if written:
        _fill(view)
    with pytest.raises(KeyNotFoundError):
        view[_missing(view_cls)]


@pytest.mark.parametrize("view_cls", MAPPINGS)
@pytest.mark.parametrize("written", (False, True))
def test_a_mapping_delete_of_a_missing_key_raises_key_not_found(root_view, view_cls, written):
    view = root_view.open_child("m", view_cls)
    if written:
        _fill(view)
    with pytest.raises(KeyNotFoundError):
        del view[_missing(view_cls)]


@pytest.mark.parametrize("view_cls", SEQUENCES)
@pytest.mark.parametrize("written", (False, True))
def test_a_sequence_read_past_the_end_raises_index_not_found(root_view, view_cls, written):
    if not written and view_cls is SEQUENCES[-1]:
        pytest.skip("a ring finds its ends by scan, so an absent ring raises its own absence")
    view = root_view.open_child("s", view_cls)
    if written:
        _fill(view)
    with pytest.raises(IndexNotFoundError):
        view[5]


@pytest.mark.parametrize("view_cls", (EagerListView, LazyListView, ByteArrayView))
@pytest.mark.parametrize("written", (False, True))
def test_a_sequence_delete_past_the_end_raises_index_not_found(root_view, view_cls, written):
    view = root_view.open_child("s", view_cls)
    if written:
        _fill(view)
    with pytest.raises(IndexNotFoundError):
        del view[5]


@pytest.mark.parametrize("view_cls", (EagerDictView, LazyDictView))
def test_a_key_the_view_cannot_hold_is_not_found(root_view, view_cls):
    view = root_view.open_child("m", view_cls)
    _fill(view)
    with pytest.raises(KeyNotFoundError):
        view[1.5]
