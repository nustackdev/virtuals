"""Hardening of every dict view against storage, on every backend.

What the contract asks of a dict view (existence, EMPTY, cost) and the
edges that tend to break at the storage boundary: odd keys, deep nesting,
large values, erase then len. Each case runs on mem, sqlite, lmdb and
rocksdb; op counts run on the seeking backends only.
"""

from __future__ import annotations

import pytest
from tests.support.counting import (
    BACKENDS,
    SEEKING_BACKENDS,
    install_counter,
    measure,
    open_backend,
)

from virtuals._views import (
    EagerDictView,
    EagerFractionalIndexedDictView,
    EagerIndexedDictView,
    EagerKh57View,
    EagerLogIndexedDictView,
    FlatDictView,
    LazyDictView,
    LightDictView,
)
from virtuals.navigator import Navigator


NESTED_VIEWS = (
    EagerDictView,
    LazyDictView,
    EagerIndexedDictView,
    EagerLogIndexedDictView,
    EagerFractionalIndexedDictView,
)
"""Dict views whose values may be compound (a child container)."""

FLAT_VIEWS = (FlatDictView, LightDictView, EagerKh57View)
"""Dict views whose values are primitives only."""

DICT_VIEWS = (*NESTED_VIEWS, *FLAT_VIEWS)

SILENT_DELETE_VIEWS = (EagerDictView, LazyDictView, FlatDictView, LightDictView)
"""Views whose delete of a missing key is a no-op; the others raise KeyError."""


def _key(view_cls, i):
    """A key of the type the view takes (Kh57 takes ints only)."""
    return i if view_cls is EagerKh57View else f"k{i}"


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture(params=BACKENDS)
def backend(request):
    return request.param


@pytest.fixture
def storage(backend, tmp_path):
    storage = open_backend(backend, tmp_path)
    try:
        yield storage
    finally:
        storage.close()


@pytest.fixture
def root(storage):
    with storage.transaction() as tx:
        yield Navigator(storage).root(tx)


@pytest.fixture(params=SEEKING_BACKENDS)
def costed(request, tmp_path, monkeypatch):
    storage = open_backend(request.param, tmp_path)
    try:
        yield storage, install_counter(storage, monkeypatch)
    finally:
        storage.close()


# ============================================================================
# EXISTENCE: a delete never creates the view (contract 6, V8)
# ============================================================================


@pytest.mark.parametrize("view_cls", DICT_VIEWS)
def test_delete_on_absent_view_does_not_create_it(root, view_cls):
    view = root.open_child("absent", view_cls)
    key = _key(view_cls, 1)
    if view_cls in SILENT_DELETE_VIEWS:
        del view[key]
    else:
        with pytest.raises(KeyError):
            del view[key]
    assert not view.container.exists()
    assert len(view) == 0
    assert key not in view
    assert list(view) == []


@pytest.mark.parametrize("view_cls", (EagerDictView, FlatDictView, LightDictView))
def test_delete_on_absent_nested_view_leaves_the_chain_absent(root, view_cls):
    """``del a.b["k"]`` on an absent chain stamps neither ``a`` nor ``b``."""
    outer = root.open_child("a", EagerDictView)
    inner = outer.open_child("b", view_cls)
    del inner["k"]
    assert not inner.container.exists()
    assert not outer.container.exists()
    assert "a" not in root


@pytest.mark.parametrize("view_cls", SILENT_DELETE_VIEWS)
def test_delete_missing_key_on_existing_view_is_silent(root, view_cls):
    view = root.open_child("d", view_cls)
    view["a"] = 1
    del view["zz"]
    assert list(view) == ["a"]
    assert len(view) == 1


@pytest.mark.parametrize("view_cls", (EagerDictView, FlatDictView))
def test_delete_missing_key_cost_is_constant_and_writes_nothing(costed, view_cls):
    storage, counts = costed
    for present in (False, True):
        if present:
            with storage.transaction() as tx:
                Navigator(storage).root(tx).open_child("d", view_cls).store({"a": 1})
        cost = measure(
            storage, counts, lambda root: root.open_child("d", view_cls).__delitem__("zz")
        )
        assert cost["put"] == 0 and cost["delete"] == 0, (present, cost)
        assert cost["scan"] == 0, (present, cost)
        assert cost["get"] + cost["exists"] <= 3, (present, cost)
