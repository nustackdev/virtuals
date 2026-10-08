"""Children created through their parent view keep the parent's books.

Every child, however it is created (nested write, lazily opened child view,
``ensure_child``, ``__setitem__``, ``store``), goes through its parent view,
so views that keep their own count or index (``DictView`` ``__len__``,
indexed dicts' ``__keys__``) see it exactly once.
"""

from __future__ import annotations

import pytest

from virtuals._views import (
    EagerDictView,
    EagerIndexedDictView,
    EagerKh57View,
    EagerListView,
    EagerLogIndexedDictView,
    FlatDictView,
    LazyDictView,
    LazyIndexedDictView,
    LazyLogIndexedDictView,
    LightDictView,
    SetView,
)
from virtuals.container import ContainerNotFoundError, ContainerStructure
from virtuals.loc.path_nav import navigate_and_ensure, navigate_view


DICT_LIKE = (EagerDictView, EagerIndexedDictView, EagerLogIndexedDictView)
LAZY_DICT_LIKE = (LazyDictView, LazyIndexedDictView, LazyLogIndexedDictView)


def _write_nested(root, parent_type, keys):
    """Write ``root[parent][k]["cells"]["name"] = "x"`` per key, the nustd.kv way."""
    for key in keys:
        path = (("rows", parent_type), (key, EagerDictView), ("cells", EagerDictView))
        navigate_and_ensure(root, path)["name"] = "x"


# ============================================================================
# NESTED WRITES
# ============================================================================


@pytest.mark.parametrize("parent_type", DICT_LIKE + LAZY_DICT_LIKE)
def test_nested_write_counts_every_new_child(root_view, parent_type):
    _write_nested(root_view, parent_type, ("r1", "r2", "r3"))

    rows = root_view.open_child("rows", parent_type)
    assert len(rows) == 3
    assert list(rows.keys()) == ["r1", "r2", "r3"]
    assert list(rows) == ["r1", "r2", "r3"]
    assert len(list(rows.values())) == 3
    assert len(root_view) == 1


@pytest.mark.parametrize("parent_type", DICT_LIKE)
def test_nested_write_extracts_with_correct_shape(root_view, parent_type):
    _write_nested(root_view, parent_type, ("r1", "r2"))

    rows = root_view.open_child("rows", parent_type)
    assert rows.extract() == {"r1": {"cells": {"name": "x"}}, "r2": {"cells": {"name": "x"}}}
    assert len(rows.open_child("r1", EagerDictView)) == 1
    assert len(rows.open_child("r1", EagerDictView).open_child("cells", EagerDictView)) == 1


@pytest.mark.parametrize("parent_type", DICT_LIKE)
def test_nested_write_deep_chain(root_view, parent_type):
    path = (
        ("a", EagerDictView),
        ("b", parent_type),
        ("c", EagerDictView),
        ("d", parent_type),
        ("e", EagerDictView),
    )
    navigate_and_ensure(root_view, path)["leaf"] = 1

    view = root_view
    for address, view_type in path:
        assert list(view.keys()) == [address]
        assert len(view) == 1
        view = view.open_child(address, view_type)
    assert list(view.keys()) == ["leaf"]
    assert len(view) == 1


@pytest.mark.parametrize("parent_type", DICT_LIKE)
def test_nested_write_is_idempotent(root_view, parent_type):
    for _ in range(3):
        _write_nested(root_view, parent_type, ("r1", "r2"))

    rows = root_view.open_child("rows", parent_type)
    assert len(rows) == 2
    assert list(rows.keys()) == ["r1", "r2"]
    assert len(root_view) == 1


def test_nested_write_through_list_parent_keeps_its_length(root_view):
    root_view["order"] = [{"x": 1}, {"x": 2}]
    path = (("order", EagerListView), (1, EagerDictView))
    navigate_and_ensure(root_view, path)["y"] = 2

    order = root_view.open_child("order", EagerListView)
    assert len(order) == 2
    assert order.extract() == [{"x": 1}, {"x": 2, "y": 2}]


def test_nested_write_past_list_end_raises(root_view):
    root_view["order"] = [{"x": 1}]
    with pytest.raises(IndexError):
        navigate_and_ensure(root_view, (("order", EagerListView), (1, EagerDictView)))
    assert len(root_view.open_child("order", EagerListView)) == 1


def test_nested_write_through_kh57_parent_counts(root_view):
    for key in (100, 42, 100):
        navigate_and_ensure(root_view, (("events", EagerKh57View), (key, EagerDictView)))["ts"] = 1

    events = root_view.open_child("events", EagerKh57View)
    assert len(events) == 2
    assert list(events) == [42, 100]


# ============================================================================
# LAZILY OPENED CHILD VIEWS
# ============================================================================


@pytest.mark.parametrize("parent_type", DICT_LIKE)
def test_write_into_lazily_opened_child(root_view, parent_type):
    rows = root_view.open_child("rows", parent_type)
    row = rows.open_child("r1", EagerDictView)
    row["name"] = "x"
    row["done"] = False
    rows.open_child("r2", EagerDictView).open_child("cells", EagerDictView)["a"] = 1

    assert len(root_view) == 1
    assert len(rows) == 2
    assert list(rows.keys()) == ["r1", "r2"]
    assert len(row) == 2


@pytest.mark.parametrize("parent_type", DICT_LIKE)
def test_store_into_lazily_opened_child(root_view, parent_type):
    rows = root_view.open_child("rows", parent_type)
    rows.open_child("r1", EagerDictView).store({"a": 1, "b": 2})
    rows.open_child("r1", EagerDictView).store({"a": 3})

    assert len(rows) == 1
    assert rows.extract() == {"r1": {"a": 3}}


def test_facet_switch_keeps_parent(root_view):
    child = root_view.open_child("rows", EagerDictView).open_child("r1", EagerDictView)
    lazy = child.lazy
    assert lazy.parent is child.parent
    lazy["x"] = 1
    assert len(root_view.open_child("rows", EagerDictView)) == 1


# ============================================================================
# ENSURE_CHILD
# ============================================================================


@pytest.mark.parametrize("parent_type", DICT_LIKE)
def test_ensure_child_creates_once(root_view, parent_type):
    rows = root_view.open_child("rows", parent_type)
    first = rows.ensure_child("r1", EagerDictView)
    second = rows.ensure_child("r1", EagerDictView)

    assert first.container.site == second.container.site
    assert first.container.exists()
    assert first.parent is rows
    assert len(rows) == 1
    assert list(rows.keys()) == ["r1"]
    assert len(root_view) == 1


def test_ensure_child_lands_under_indexed_data(root_view):
    idx = root_view.open_child("idx", EagerIndexedDictView)
    child = idx.ensure_child("a", EagerDictView)
    assert child.container.site == (*idx.container.site, "__data__", "a")


def test_ensure_child_runs_child_internal_layout(root_view):
    log = root_view.ensure_child("log", EagerLogIndexedDictView)
    log["k"] = 1
    assert len(log) == 1
    assert len(root_view) == 1


def test_ensure_child_on_non_nestable_raises(root_view):
    light = root_view.ensure_child("light", LightDictView)
    with pytest.raises(TypeError):
        light.ensure_child("x", EagerDictView)


def test_setitem_then_nested_write_no_double_count(root_view):
    root_view["rows"] = {"r1": {"a": 1}}
    _write_nested(root_view, EagerDictView, ("r1", "r2"))
    root_view.open_child("rows", EagerDictView).set_child_container_as(
        "r2", {"b": 2}, EagerDictView
    )

    rows = root_view.open_child("rows", EagerDictView)
    assert len(rows) == 2
    assert len(root_view) == 1


@pytest.mark.parametrize("parent_type", DICT_LIKE)
def test_store_without_replace_counts_only_new_keys(root_view, parent_type):
    view = root_view.ensure_child("d", parent_type)
    view.store({"a": 1, "b": {"x": 1}})
    view.store({"b": {"x": 2}, "c": 3}, replace=False)

    assert len(view) == 3
    assert list(view.keys()) == ["a", "b", "c"]


def test_set_store_counts_unique_members(root_view):
    tags = root_view.ensure_child("tags", SetView)
    tags.store(["a", "b", "a"])
    tags.add("b")
    tags.add("c")
    assert len(tags) == 3


# ============================================================================
# PRIMITIVE WRITES
# ============================================================================


@pytest.mark.parametrize("parent_type", DICT_LIKE)
def test_primitive_write_is_counted(root_view, parent_type):
    view = root_view.open_child("d", parent_type)
    view._primitive_write("blob", {"a": [1, 2]})
    view._primitive_write("blob", {"a": [3]})

    assert len(view) == 1
    assert list(view.keys()) == ["blob"]
    assert view._primitive_read("blob") == {"a": [3]}


def test_flat_dict_primitive_set_is_counted(root_view):
    flat = root_view.open_child("flat", FlatDictView)
    flat._set_primitive("a", 1)
    flat["a"] = 2
    flat["b"] = 3
    assert len(flat) == 2
    assert len(root_view) == 1


def test_light_dict_counts_live(root_view):
    light = root_view.open_child("light", LightDictView)
    light["a"] = 1
    light["a"] = 2
    light._unsafe_primitive_write("b", 3)
    assert len(light) == 2
    assert len(root_view) == 1


def test_unsafe_write_skips_parent_bookkeeping(root_view):
    view = root_view.ensure_child("d", EagerDictView)
    view["a"] = 1
    view._unsafe_primitive_write("b", 2)

    # Documented: a new key written unsafely is not counted
    assert len(view) == 1
    assert set(view.keys()) == {"a", "b"}


# ============================================================================
# PARENT CHAIN
# ============================================================================


def test_navigate_view_builds_parent_chain(root_view):
    path = (("a", EagerDictView), ("b", EagerIndexedDictView), ("c", EagerDictView))
    leaf = navigate_view(root_view, path)

    assert leaf.parent.parent.parent is root_view
    assert isinstance(leaf.parent, EagerIndexedDictView)
    assert leaf.open_parent() is leaf.parent
    # Pure navigation, nothing stored
    assert not root_view.container.exists()


def test_root_has_no_parent_and_creates_itself(nav, tx):
    root = nav.root(tx)
    assert root.parent is None
    root.ensure_created()
    assert root.container.exists()


def test_root_at_missing_storage_parent_raises(nav, tx):
    root = nav.root_at(("/", "ns", "inner"), tx)
    with pytest.raises(ContainerNotFoundError):
        root["x"] = 1
    assert not nav.root(tx).container.exists()


# ============================================================================
# REBUILD
# ============================================================================


def test_rebuild_repairs_dict_counters_in_subtree(root_view):
    _write_nested(root_view, EagerDictView, ("r1", "r2", "r3"))
    rows = root_view.open_child("rows", EagerDictView)
    rows._set_length(0)
    rows.open_child("r1", EagerDictView)._set_length(7)
    root_view._set_length(0)

    root_view.rebuild_bookkeeping()

    assert len(root_view) == 1
    assert len(rows) == 3
    assert len(rows.open_child("r1", EagerDictView)) == 1


def test_rebuild_counts_unsafe_writes(root_view):
    view = root_view.ensure_child("d", EagerDictView)
    view._unsafe_primitive_write("a", 1)
    view._unsafe_primitive_write("b", 2)
    assert len(view) == 0

    view.rebuild_bookkeeping()
    assert len(view) == 2


def test_rebuild_repairs_indexed_index(root_view):
    _write_nested(root_view, EagerIndexedDictView, ("r1", "r2", "r3"))
    idx = root_view.open_child("rows", EagerIndexedDictView)
    keys = idx._keys_view()
    keys.store(["r2", "gone", "r2"])

    root_view.rebuild_bookkeeping()

    assert len(idx) == 3
    assert list(idx.keys()) == ["r2", "r1", "r3"]


def test_rebuild_repairs_log_indexed_log(root_view):
    _write_nested(root_view, EagerLogIndexedDictView, ("r1", "r2", "r3"))
    log = root_view.open_child("rows", EagerLogIndexedDictView)
    entries = list(log.keys_with_log_keys())
    log._keys_container().delete_child(entries[0][0])
    log._append_log_key("r2")
    log._append_log_key("gone")

    root_view.rebuild_bookkeeping()

    assert len(log) == 3
    assert sorted(log.keys()) == ["r1", "r2", "r3"]
    assert list(log.keys())[:2] == ["r2", "r3"]


def test_rebuild_skips_unknown_structures(root_view):
    root_view["a"] = {"x": 1}
    child = root_view.open_child("a", EagerDictView)
    child.container.ctx.put((*root_view.container.site, "odd"), _marker(99))
    root_view._set_length(0)

    root_view.rebuild_bookkeeping()
    assert len(root_view) == 2


def test_rebuild_on_missing_view_is_silent(root_view):
    root_view.open_child("missing", EagerDictView).rebuild_bookkeeping()
    assert not root_view.container.exists()


def _marker(structure: int) -> object:
    from virtuals.container import ContainerProtocol, create_marker

    return create_marker(ContainerStructure(structure), ContainerProtocol.MUTABLE)
