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
from virtuals.types import EMPTY


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


# ============================================================================
# EMPTY IS NEVER STORED (contract 6, V6)
# ============================================================================


def _rows_under(view):
    """Every storage row under the view's site, as decoded keys."""
    from virtuals.tkv.filter import PrefixFilter
    from virtuals.tkv.storage import StorageScanOptions

    site = view.container.site
    prefix = PrefixFilter(prefix=site)
    opts = StorageScanOptions(start=site, break_filter=prefix, filter=prefix)
    return [k for k, _ in view.container.ctx.scan(opts).items()]


@pytest.mark.parametrize("view_cls", DICT_VIEWS)
def test_set_empty_raises_and_stores_nothing(root, view_cls):
    view = root.open_child("d", view_cls)
    view[_key(view_cls, 1)] = 1
    rows = _rows_under(view)
    with pytest.raises(ValueError, match="EMPTY"):
        view[_key(view_cls, 2)] = EMPTY
    assert _rows_under(view) == rows
    assert len(view) == 1
    assert list(view) == [_key(view_cls, 1)]
    assert _key(view_cls, 2) not in view
    assert list(view.items()) == [(_key(view_cls, 1), 1)]


@pytest.mark.parametrize("view_cls", DICT_VIEWS)
def test_set_empty_on_absent_view_leaves_it_absent(root, view_cls):
    view = root.open_child("absent", view_cls)
    with pytest.raises(ValueError, match="EMPTY"):
        view[_key(view_cls, 1)] = EMPTY
    assert not view.container.exists()


@pytest.mark.parametrize("view_cls", DICT_VIEWS)
def test_store_and_update_with_empty_raise(root, view_cls):
    view = root.open_child("d", view_cls)
    with pytest.raises(ValueError, match="EMPTY"):
        view.store({_key(view_cls, 1): 1, _key(view_cls, 2): EMPTY})
    with pytest.raises(ValueError, match="EMPTY"):
        view.update({_key(view_cls, 3): EMPTY})
    assert _key(view_cls, 2) not in view
    assert _key(view_cls, 3) not in view


@pytest.mark.parametrize("view_cls", NESTED_VIEWS)
def test_nested_empty_raises(root, view_cls):
    """EMPTY inside a compound value is rejected at the leaf."""
    view = root.open_child("d", view_cls)
    with pytest.raises(ValueError, match="EMPTY"):
        view["a"] = {"x": 1, "y": {"z": EMPTY}}
    with pytest.raises(ValueError, match="EMPTY"):
        view["b"] = [1, EMPTY]


@pytest.mark.parametrize("view_cls", (EagerLogIndexedDictView, EagerFractionalIndexedDictView))
def test_set_primitive_empty_raises(root, view_cls):
    view = root.open_child("d", view_cls)
    view.store({"a": 1})
    with pytest.raises(ValueError, match="EMPTY"):
        view.set_primitive("e", EMPTY)
    assert list(view) == ["a"]
    assert len(view) == 1


def test_fractional_insert_empty_leaves_no_trace(root):
    view = root.open_child("lst", EagerFractionalIndexedDictView)
    view.store({"a": 1})
    for op in (
        lambda: view.append("e", EMPTY),
        lambda: view.prepend("e", EMPTY),
        lambda: view.insert_after("a", "e", EMPTY),
    ):
        with pytest.raises(ValueError, match="EMPTY"):
            op()
        assert "e" not in view
        assert len(view) == 1
        assert list(view) == ["a"]
        assert view.get("e", "dflt") == "dflt"


def test_container_put_child_primitive_rejects_empty(root):
    view = root.open_child("d", EagerDictView)
    view["a"] = 1
    with pytest.raises(ValueError, match="EMPTY"):
        view.container.put_child_primitive("e", EMPTY)
    with pytest.raises(ValueError, match="EMPTY"):
        view.container.put_child_primitive("e", EMPTY, validate=False, validate_parent=False)
    assert list(view) == ["a"]
