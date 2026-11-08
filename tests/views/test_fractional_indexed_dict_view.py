"""Tests for FractionalIndexedDictView (ordered collection keyed by id).

Behaviour runs on every backend. Key-touch counts run on the seeking
backends (sqlite, lmdb, rocksdb) and must not change with the size of the
collection. Hooks run on mem with the in-memory transport.
"""

from __future__ import annotations

import random
import threading
import time
from collections import Counter
from collections.abc import ItemsView, KeysView, ValuesView
from typing import ClassVar

import pytest

from virtuals._views import (
    EagerDictView,
    EagerFractionalIndexedDictView,
    LazyFractionalIndexedDictView,
)
from virtuals._views.fractional_index import validate_order_key
from virtuals.codecs import BinaryCodec, NoOpCodec
from virtuals.container import ContainerNotFoundError, ContainerStructure
from virtuals.loc.path_nav import navigate_and_ensure
from virtuals.navigator import Navigator
from virtuals.observers.mem import InMemoryObserver
from virtuals.publishers.mem import InMemoryPublisher
from virtuals.storages.lmdb import LMDBStorage
from virtuals.storages.mem import InMemoryStorage
from virtuals.storages.rocksdb import RocksDBStorage
from virtuals.storages.sqlite import SQLiteStorage
from virtuals.tkv.storage import StorageLockTimeoutError, StorageTransactionConflictError
from virtuals.tkv.transport import InMemoryTransport
from virtuals.types import is_empty


BACKENDS = ("mem", "sqlite", "lmdb", "rocksdb")
SEEKING_BACKENDS = ("sqlite", "lmdb", "rocksdb")


def _open_storage(backend, tmp_path, publisher=None):
    if backend == "mem":
        storage = InMemoryStorage(codec=NoOpCodec(), publisher=publisher)
    elif backend == "sqlite":
        storage = SQLiteStorage(path=tmp_path / "t.sqlite", codec=BinaryCodec())
    elif backend == "lmdb":
        storage = LMDBStorage(path=tmp_path / "t.lmdb", codec=BinaryCodec(), map_size=2**27)
    elif backend == "rocksdb":
        storage = RocksDBStorage(path=tmp_path / "t.rocks", codec=BinaryCodec())
    else:
        raise ValueError(backend)
    storage.open()
    return storage


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture(params=BACKENDS)
def backend(request):
    return request.param


@pytest.fixture
def bstorage(backend, tmp_path):
    storage = _open_storage(backend, tmp_path)
    try:
        yield storage
    finally:
        storage.close()


@pytest.fixture
def btx(bstorage):
    with bstorage.transaction() as tx:
        yield tx


@pytest.fixture
def broot(bstorage, btx):
    return Navigator(bstorage).root(btx)


@pytest.fixture
def fdict(broot):
    """Factory for EagerFractionalIndexedDictViews on the parametrized backend."""

    def _create(address="lst", data=None):
        view = broot.open_child(address, EagerFractionalIndexedDictView)
        if data is not None:
            view.store(data)
        else:
            view.ensure_created()
        return view

    return _create


def _assert_consistent(view):
    """Order rows, reverse index and data agree; order keys are valid and unique."""
    rows = list(view.positions())
    okeys = [okey for okey, _ in rows]
    ids = [address for _, address in rows]
    assert okeys == sorted(okeys)
    assert len(set(okeys)) == len(okeys)
    assert len(set(ids)) == len(ids)
    for okey in okeys:
        validate_order_key(okey)
    for okey, address in rows:
        assert view.position(address) == okey
        assert address in view
    assert len(view) == len(rows)
    assert list(view) == ids
    assert list(reversed(view)) == ids[::-1]
    assert list(view.keys()) == ids
    assert [k for k, _ in view.items()] == ids


# ============================================================================
# BASIC OPERATIONS
# ============================================================================


def test_empty(fdict):
    view = fdict()
    assert len(view) == 0
    assert list(view) == []
    assert view.first() is None
    assert view.last() is None
    assert "a" not in view
    assert is_empty(view.get("a"))
    assert view.get("a", "dflt") == "dflt"


def test_set_appends_in_order(fdict):
    view = fdict()
    view["b"] = 2
    view["a"] = 1
    view["c"] = 3
    assert list(view) == ["b", "a", "c"]
    assert view["a"] == 1
    assert len(view) == 3
    _assert_consistent(view)


def test_store_mapping_keeps_order(fdict):
    view = fdict("lst", {"z": 0, "y": 1, "x": 2})
    assert list(view) == ["z", "y", "x"]
    assert view.extract() == {"z": 0, "y": 1, "x": 2}
    assert list(view.extract()) == ["z", "y", "x"]


def test_store_iterable_of_ids(fdict):
    view = fdict()
    view.store(["c", "a", "b"])
    assert list(view) == ["c", "a", "b"]
    assert view["a"] is True


def test_store_replaces(fdict):
    view = fdict("lst", {"a": 1, "b": 2})
    view.store({"c": 3})
    assert list(view) == ["c"]
    view.store({"d": 4}, replace=False)
    assert list(view) == ["c", "d"]


def test_set_existing_keeps_position(fdict):
    view = fdict("lst", {"a": 1, "b": 2, "c": 3})
    okey = view.position("b")
    view["b"] = 20
    assert view["b"] == 20
    assert list(view) == ["a", "b", "c"]
    assert view.position("b") == okey
    assert len(view) == 3


def test_int_ids(fdict):
    view = fdict()
    view[3] = "three"
    view[1] = "one"
    view.insert_before(1, 2, "two")
    assert list(view) == [3, 2, 1]
    assert 2 in view
    assert view[2] == "two"
    _assert_consistent(view)


def test_nested_values(fdict):
    view = fdict()
    view["a"] = {"name": "A", "tags": [1, 2]}
    view["b"] = 5
    assert view["a"] == {"name": "A", "tags": [1, 2]}
    assert list(view.values()) == [{"name": "A", "tags": [1, 2]}, 5]
    assert view.extract() == {"a": {"name": "A", "tags": [1, 2]}, "b": 5}
    _assert_consistent(view)


def test_set_primitive_blob(fdict):
    view = fdict()
    view.set_primitive("a", {"x": 1})
    view.set_primitive("b", [1, 2])
    assert list(view) == ["a", "b"]
    assert view["a"] == {"x": 1}
    assert view["b"] == [1, 2]


def test_get_missing_returns_empty(fdict):
    """Dict access with EMPTY as the absent default: get of a missing id is EMPTY, never None."""
    view = fdict("lst", {"a": 1})
    assert is_empty(view.get("zz"))
    assert view.get("a") == 1


def test_update_and_get(fdict):
    view = fdict()
    view.update({"a": 1}, b=2)
    assert list(view) == ["a", "b"]
    assert view.get("a") == 1
    assert view.get("zz", "dflt") == "dflt"


def test_clear(fdict):
    view = fdict("lst", {"a": 1, "b": 2})
    view.clear()
    assert len(view) == 0
    assert list(view) == []
    assert list(view.positions()) == []
    view["c"] = 3
    assert list(view) == ["c"]


# ============================================================================
# PLACEMENT
# ============================================================================


def test_append_prepend(fdict):
    view = fdict()
    view.append("b", 2)
    view.append("c", 3)
    view.prepend("a", 1)
    view.prepend("z", 0)
    assert list(view) == ["z", "a", "b", "c"]
    assert view["z"] == 0
    assert view.first() == "z"
    assert view.last() == "c"
    _assert_consistent(view)


def test_append_default_value(fdict):
    view = fdict()
    view.append("a")
    assert view["a"] is True


def test_insert_before_after(fdict):
    view = fdict("lst", {"a": 1, "c": 3})
    view.insert_before("c", "b", 2)
    view.insert_after("c", "d", 4)
    view.insert_before("a", "start", 0)
    view.insert_after("d", "end", 5)
    assert list(view) == ["start", "a", "b", "c", "d", "end"]
    assert view["b"] == 2
    assert view["d"] == 4
    _assert_consistent(view)


def test_insert_nested_value(fdict):
    view = fdict("lst", {"a": 1, "c": 3})
    view.insert_before("c", "b", {"k": "v"})
    assert list(view) == ["a", "b", "c"]
    assert view["b"] == {"k": "v"}
    _assert_consistent(view)


def test_insert_existing_raises(fdict):
    view = fdict("lst", {"a": 1, "b": 2})
    with pytest.raises(ValueError, match="already"):
        view.insert_before("b", "a", 9)
    with pytest.raises(ValueError, match="already"):
        view.append("a", 9)
    with pytest.raises(ValueError, match="already"):
        view.prepend("b", 9)
    assert list(view) == ["a", "b"]
    assert view["a"] == 1


def test_insert_missing_ref_raises(fdict):
    view = fdict("lst", {"a": 1})
    with pytest.raises(KeyError):
        view.insert_before("nope", "b", 2)
    with pytest.raises(KeyError):
        view.insert_after("nope", "b", 2)
    assert "b" not in view
    assert list(view) == ["a"]


def test_insert_missing_ref_does_not_create_view(broot):
    view = broot.open_child("lst", EagerFractionalIndexedDictView)
    with pytest.raises(KeyError):
        view.insert_before("nope", "b", 2)
    with pytest.raises(KeyError):
        view.insert_after("nope", "b", 2)
    assert not view.container.exists()


def test_move_before_after(fdict):
    view = fdict("lst", {"a": 1, "b": 2, "c": 3, "d": 4})
    view.move_before("d", "b")
    assert list(view) == ["a", "d", "b", "c"]
    view.move_after("a", "c")
    assert list(view) == ["d", "b", "c", "a"]
    view.move_before("a", "d")
    assert list(view) == ["a", "d", "b", "c"]
    assert view.extract() == {"a": 1, "d": 4, "b": 2, "c": 3}
    _assert_consistent(view)


def test_move_noops(fdict):
    view = fdict("lst", {"a": 1, "b": 2, "c": 3})
    before = dict(view.positions())
    view.move_before("a", "b")  # already directly before
    view.move_after("c", "b")  # already directly after
    view.move_before("b", "b")  # self
    view.move_first("a")
    view.move_last("c")
    assert dict(view.positions()) == before
    assert list(view) == ["a", "b", "c"]


def test_move_first_last(fdict):
    view = fdict("lst", {"a": 1, "b": 2, "c": 3})
    view.move_first("c")
    assert list(view) == ["c", "a", "b"]
    view.move_last("c")
    assert list(view) == ["a", "b", "c"]
    view.move_last("a")
    assert list(view) == ["b", "c", "a"]
    _assert_consistent(view)


def test_move_missing_raises(fdict):
    view = fdict("lst", {"a": 1, "b": 2})
    with pytest.raises(KeyError):
        view.move_before("zz", "a")
    with pytest.raises(KeyError):
        view.move_after("a", "zz")
    with pytest.raises(KeyError):
        view.move_first("zz")
    assert list(view) == ["a", "b"]


def test_move_keeps_value(fdict):
    view = fdict("lst", {"a": {"n": 1}, "b": 2})
    view.move_last("a")
    assert list(view) == ["b", "a"]
    assert view["a"] == {"n": 1}


def test_many_inserts_at_one_gap(fdict):
    view = fdict("lst", {"a": 0, "b": 1})
    for i in range(120):
        view.insert_after("a", f"x{i:03d}", i)
    expected = ["a", *[f"x{i:03d}" for i in reversed(range(120))], "b"]
    assert list(view) == expected
    _assert_consistent(view)
    for i in range(60):
        view.insert_before("b", f"y{i:03d}", i)
    assert list(view)[-61:] == [*[f"y{i:03d}" for i in range(60)], "b"]
    _assert_consistent(view)


def test_random_ops_keep_consistency(fdict):
    import random

    rng = random.Random(3)  # noqa: S311
    view = fdict()
    model: list[str] = []
    for i in range(300):
        op = rng.choice(
            ["append", "prepend", "before", "after", "move_before", "move_after", "del"]
        )
        new = f"n{i}"
        if op == "append" or not model:
            view.append(new, i)
            model.append(new)
        elif op == "prepend":
            view.prepend(new, i)
            model.insert(0, new)
        elif op == "before":
            ref = rng.choice(model)
            view.insert_before(ref, new, i)
            model.insert(model.index(ref), new)
        elif op == "after":
            ref = rng.choice(model)
            view.insert_after(ref, new, i)
            model.insert(model.index(ref) + 1, new)
        elif op == "move_before":
            a, ref = rng.choice(model), rng.choice(model)
            view.move_before(a, ref)
            if a != ref:
                model.remove(a)
                model.insert(model.index(ref), a)
        elif op == "move_after":
            a, ref = rng.choice(model), rng.choice(model)
            view.move_after(a, ref)
            if a != ref:
                model.remove(a)
                model.insert(model.index(ref) + 1, a)
        elif op == "del":
            a = rng.choice(model)
            del view[a]
            model.remove(a)
        assert list(view) == model
    _assert_consistent(view)


# ============================================================================
# DELETE
# ============================================================================


def test_delete(fdict):
    view = fdict("lst", {"a": 1, "b": 2, "c": 3})
    del view["b"]
    assert "b" not in view
    assert list(view) == ["a", "c"]
    assert len(view) == 2
    with pytest.raises(KeyError):
        view.position("b")
    with pytest.raises(KeyError):
        view["b"]
    _assert_consistent(view)


def test_delete_nested(fdict):
    view = fdict("lst", {"a": {"x": {"y": 1}}, "b": 2})
    del view["a"]
    assert list(view) == ["b"]
    assert view.extract() == {"b": 2}


def test_delete_missing_raises(fdict):
    view = fdict("lst", {"a": 1})
    with pytest.raises(KeyError):
        del view["missing"]


def test_delete_on_absent_view_raises_without_creating(broot):
    view = broot.open_child("lst", EagerFractionalIndexedDictView)
    with pytest.raises(KeyError):
        del view["x"]
    assert not view.container.exists()


def test_pop(fdict):
    view = fdict("lst", {"a": 1, "b": 2})
    assert view.pop("a") == 1
    assert list(view) == ["b"]
    assert view.pop("zz", "d") == "d"
    with pytest.raises(KeyError):
        view.pop("zz")


def test_delete_then_reinsert_appends(fdict):
    view = fdict("lst", {"a": 1, "b": 2, "c": 3})
    del view["a"]
    view["a"] = 10
    assert list(view) == ["b", "c", "a"]


# ============================================================================
# ORDERED READS, CURSORS
# ============================================================================


def test_first_last_next_prev(fdict):
    view = fdict("lst", {"a": 1, "b": 2, "c": 3})
    assert view.first() == "a"
    assert view.last() == "c"
    assert view.next_after("a") == "b"
    assert view.next_after("c") is None
    assert view.prev_before("c") == "b"
    assert view.prev_before("a") is None
    with pytest.raises(KeyError):
        view.next_after("zz")


def test_keys_after_with_limit_is_exact(fdict):
    view = fdict("lst", {k: i for i, k in enumerate("abcdefg")})
    assert list(view.keys(after="b", limit=3)) == ["c", "d", "e"]
    assert list(view.keys(after="b")) == ["c", "d", "e", "f", "g"]
    assert list(view.keys(limit=2)) == ["a", "b"]
    assert list(view.keys(after="g", limit=3)) == []
    assert list(view.keys(after="f", limit=3)) == ["g"]
    with pytest.raises(KeyError):
        view.keys(after="zz")  # raised at the call, before any read
    with pytest.raises(KeyError):
        view.keys_reverse(before="zz")


def test_keys_after_a_cursor_of_the_wrong_type_raises_type_error(fdict):
    view = fdict("lst", {"a": 1})
    with pytest.raises(TypeError):
        view.keys(after=1.5)


def test_keys_reverse_before_with_limit_is_exact(fdict):
    view = fdict("lst", {k: i for i, k in enumerate("abcdefg")})
    assert list(view.keys_reverse(before="f", limit=3)) == ["e", "d", "c"]
    assert list(view.keys_reverse(before="f")) == ["e", "d", "c", "b", "a"]
    assert list(view.keys_reverse(limit=2)) == ["g", "f"]
    assert list(view.keys_reverse(before="a", limit=3)) == []
    assert list(view.keys_reverse(before="b", limit=3)) == ["a"]


def _follow(view, cursor=None):
    """Every (order key, id) past ``cursor``, one ``next_key_after`` call each."""
    rows = []
    while (row := view.next_key_after(cursor)) is not None:
        rows.append(row)
        cursor = row[0]
    return rows


def test_next_key_after_steps_by_order_key(fdict):
    view = fdict("lst", {"a": 1, "b": 2, "c": 3})
    rows = _follow(view)
    assert [address for _, address in rows] == ["a", "b", "c"]
    assert rows == list(view.positions())
    assert view.next_key_after(rows[-1][0]) is None


def test_next_key_after_resumes_past_a_deleted_id(fdict):
    view = fdict("lst", {"a": 1, "b": 2, "c": 3})
    cursor = view.position("b")
    del view["b"]
    view["d"] = 4
    assert [address for _, address in _follow(view, cursor)] == ["c", "d"]


@pytest.mark.parametrize("rounds", [40])
def test_next_key_after_sees_an_append_past_a_deleted_last_id(fdict, rounds):
    # The new key is drawn with a random tail; without the end mark it lands
    # below the deleted last key about half the time.
    view = fdict("lst", {"a": 1})
    for i in range(rounds):
        cursor = view.position(view.last())
        del view[view.last()]
        view[f"n{i}"] = i
        assert [address for _, address in _follow(view, cursor)] == [f"n{i}"]


def test_next_key_after_sees_an_append_to_a_view_emptied_under_the_cursor(fdict):
    for i in range(40):
        view = fdict(f"lst{i}", {"a": 1})
        cursor = view.position("a")
        del view["a"]
        view["b"] = 2
        assert [address for _, address in _follow(view, cursor)] == ["b"]


def test_next_key_after_sees_an_append_past_a_last_id_moved_away(fdict):
    for i in range(40):
        view = fdict(f"lst{i}", {"a": 1, "b": 2})
        cursor = view.position("b")
        view.move_first("b")
        view["c"] = 3
        assert [address for _, address in _follow(view, cursor)] == ["c"]


def test_next_key_after_sees_an_append_after_clear(fdict):
    for i in range(20):
        view = fdict(f"lst{i}", {"a": 1, "b": 2})
        cursor = view.position("b")
        view.clear()
        view["c"] = 3
        assert [address for _, address in _follow(view, cursor)] == ["c"]


def test_end_mark_is_not_an_id(fdict):
    view = fdict("lst", {"a": 1, "b": 2})
    del view["b"]
    view["c"] = 3
    assert list(view) == ["a", "c"]
    assert len(view) == 2
    assert view.extract() == {"a": 1, "c": 3}


def test_a_middle_delete_writes_no_end_mark(fdict):
    view = fdict("lst", {"a": 1, "b": 2, "c": 3})
    del view["b"]
    assert view._end_mark() is None
    del view["c"]
    assert view._end_mark() is not None


def test_next_key_after_on_an_empty_view_is_none(fdict):
    view = fdict("lst", {"a": 1})
    del view["a"]
    assert view.next_key_after(None) is None


def test_next_key_after_on_a_never_written_view_raises(broot):
    view = broot.open_child("absent", EagerFractionalIndexedDictView)
    with pytest.raises(ContainerNotFoundError):
        view.next_key_after(None)
    assert not view.container.exists()


def test_keys_returns_keysview_without_args(fdict):
    view = fdict("lst", {"a": 1})
    assert isinstance(view.keys(), KeysView)
    assert "a" in view.keys()


def test_values_items_views(fdict):
    view = fdict("lst", {"a": 1, "b": {"n": 2}, "c": 3})
    values = view.values()
    items = view.items()
    assert isinstance(values, ValuesView)
    assert isinstance(items, ItemsView)
    assert list(values) == [1, {"n": 2}, 3]
    assert list(items) == [("a", 1), ("b", {"n": 2}), ("c", 3)]
    assert list(reversed(values)) == [3, {"n": 2}, 1]
    assert list(reversed(items)) == [("c", 3), ("b", {"n": 2}), ("a", 1)]
    assert 3 in values
    assert ("a", 1) in items


def test_positions_are_opaque_sorted_strings(fdict):
    view = fdict("lst", {"a": 1, "b": 2})
    rows = list(view.positions())
    assert [address for _, address in rows] == ["a", "b"]
    assert all(isinstance(okey, str) for okey, _ in rows)
    assert list(view.positions(reverse=True)) == rows[::-1]


# ============================================================================
# FACETS, REGISTRY, NESTED WRITES
# ============================================================================


def test_lazy_facet_returns_views(fdict):
    view = fdict("lst", {"a": {"n": 1}, "b": 2})
    lazy = view.lazy
    assert isinstance(lazy, LazyFractionalIndexedDictView)
    child = lazy["a"]
    assert isinstance(child, EagerDictView)
    assert child["n"] == 1
    assert lazy["b"] == 2
    assert list(lazy) == ["a", "b"]
    assert [type(v).__name__ for v in lazy.values()] == ["EagerDictView", "int"]
    assert lazy.extract() == {"a": {"n": 1}, "b": 2}
    assert isinstance(lazy.eager, EagerFractionalIndexedDictView)
    assert lazy.eager.extract() == {"a": {"n": 1}, "b": 2}


def test_lazy_facet_mutations(fdict):
    view = fdict("lst", {"a": 1})
    lazy = view.lazy
    lazy.insert_before("a", "z", 0)
    lazy["b"] = {"n": 2}
    lazy.move_first("b")
    assert list(view) == ["b", "z", "a"]
    assert view["b"] == {"n": 2}


def test_structure_id_and_layout(fdict):
    view = fdict("lst", {"a": 1})
    assert view.container.info().structure == ContainerStructure(17)
    assert sorted(view.container.iter_child_keys()) == ["__data__", "__order__", "__pos__"]


def test_parent_walk_through_registry(tmp_path):
    """An eager parent extracts a FractionalIndexed child via the registry alias."""
    from virtuals.views import FractionalIndexedDictView

    storage = _open_storage("mem", tmp_path)
    try:
        nav = Navigator(storage, views=(EagerDictView, FractionalIndexedDictView))
        with storage.transaction() as tx:
            root = nav.root(tx)
            child = root.open_child("lst", EagerFractionalIndexedDictView)
            child.store({"b": {"n": 1}, "a": 2})
        with storage.snapshot() as snap:
            root = nav.root(snap)
            extracted = root["lst"]
            assert extracted == {"b": {"n": 1}, "a": 2}
            assert list(extracted) == ["b", "a"]
            lazy = root.lazy["lst"]
            assert isinstance(lazy, LazyFractionalIndexedDictView)
    finally:
        storage.close()


def test_child_created_through_ensure_path_is_appended(broot):
    path = (
        ("rows", EagerFractionalIndexedDictView),
        ("r2", EagerDictView),
        ("cells", EagerDictView),
    )
    navigate_and_ensure(broot, path)["name"] = "x"
    path = (
        ("rows", EagerFractionalIndexedDictView),
        ("r1", EagerDictView),
        ("cells", EagerDictView),
    )
    navigate_and_ensure(broot, path)["name"] = "y"
    navigate_and_ensure(broot, path)["name"] = "z"  # existing child: no new row
    rows = broot.open_child("rows", EagerFractionalIndexedDictView)
    assert list(rows) == ["r2", "r1"]
    assert len(rows) == 2
    assert rows["r1"] == {"cells": {"name": "z"}}
    _assert_consistent(rows)


# ============================================================================
# BOOKKEEPING: REBUILD, RESPACE
# ============================================================================


def test_rebuild_bookkeeping_repairs_index(fdict):
    view = fdict("lst", {"a": 1, "b": 2, "c": 3})
    oc = view._order_container()
    pc = view._pos_container()
    dc = view._data_container()
    # Lose b's order row, leave a stale row, add data behind the view's back.
    oc.delete_child_primitive_unsafe(view.position("b"))
    oc.put_child_primitive("zzzz", "ghost", validate=False, validate_parent=False)
    pc.put_child_primitive("ghost", "zzzz", validate=False, validate_parent=False)
    dc.put_child_primitive("d", 4, validate=False, validate_parent=False)
    view.rebuild_bookkeeping()
    assert list(view) == ["a", "c", "b", "d"]
    assert "ghost" not in view
    assert len(view) == 4
    _assert_consistent(view)


def test_rebuild_bookkeeping_drops_duplicate_rows(fdict):
    view = fdict("lst", {"a": 1, "b": 2})
    oc = view._order_container()
    oc.put_child_primitive("zzzz", "a", validate=False, validate_parent=False)
    view.rebuild_bookkeeping()
    assert list(view) == ["a", "b"]
    _assert_consistent(view)


def test_respace_rewrites_keys_evenly(fdict):
    view = fdict("lst", {"a": 0, "b": 1})
    for i in range(80):
        view.insert_after("a", f"x{i}", i)
    order = list(view)
    assert max(len(okey) for okey, _ in view.positions()) > 10
    view.respace()
    assert list(view) == order
    assert max(len(okey) for okey, _ in view.positions()) <= 7
    _assert_consistent(view)
    view.insert_after("a", "fresh", 99)
    assert list(view)[:2] == ["a", "fresh"]


def test_respace_empty(fdict):
    view = fdict()
    view.respace()
    assert len(view) == 0


# ============================================================================
# KEY LENGTH: WINDOW RESPACE KEEPS KEYS BOUNDED
# ============================================================================


class _TightView(EagerFractionalIndexedDictView):
    """Respaces early, so a short run hits the window path on every backend."""

    MAX_KEY_LENGTH = 24
    RESPACE_WINDOW = 4
    RESPACE_TARGET = 16


class _CountingRespaceView(_TightView):
    respaces: ClassVar[list] = []

    def _respace_around(self, prev, nxt):
        okey = super()._respace_around(prev, nxt)
        type(self).respaces.append(okey)
        return okey


def _max_key_len(view):
    return max(len(okey) for okey, _ in view.positions())


def test_hot_gap_10k_inserts_on_lmdb_keys_stay_bounded(tmp_path):
    """LMDB caps keys at 511 bytes; 10k inserts after one id used to fail at ~2.3k."""
    storage = _open_storage("lmdb", tmp_path)
    try:
        with storage.transaction() as tx:
            view = Navigator(storage).root(tx).open_child("lst", EagerFractionalIndexedDictView)
            view.store(["head", "tail"])
            for i in range(10_000):
                view.insert_after("head", f"c{i}")
                assert len(view.position(f"c{i}")) <= view.MAX_KEY_LENGTH
            expected = ["head", *[f"c{i}" for i in reversed(range(10_000))], "tail"]
            assert list(view) == expected
            assert _max_key_len(view) <= view.MAX_KEY_LENGTH
            _assert_consistent(view)
    finally:
        storage.close()


@pytest.mark.parametrize("hot", ["after_head", "before_tail", "alternate"])
def test_hot_gap_respaces_bounded_window(broot, hot):
    _CountingRespaceView.respaces = []
    view = broot.open_child("lst", _CountingRespaceView)
    view.store(["head", "tail"])
    model = ["head", "tail"]
    for i in range(600):
        after = hot == "after_head" or (hot == "alternate" and i % 2 == 0)
        if after:
            view.insert_after("head", i)
            model.insert(1, i)
        else:
            view.insert_before("tail", i)
            model.insert(len(model) - 1, i)
    assert list(view) == model
    assert _max_key_len(view) <= view.MAX_KEY_LENGTH
    assert len(_CountingRespaceView.respaces) >= 5
    # The key handed to the gap after a respace is short again.
    assert max(len(k) for k in _CountingRespaceView.respaces) <= view.RESPACE_TARGET
    _assert_consistent(view)


def test_mixed_hot_gap_and_random_ops_match_list_model(broot):
    rng = random.Random(11)  # noqa: S311
    view = broot.open_child("lst", _TightView)
    view.store(["h", "t"])
    model = ["h", "t"]
    nid = 0
    for _ in range(1500):
        r = rng.random()
        if r < 0.35:
            view.insert_after("h", nid)
            model.insert(model.index("h") + 1, nid)
            nid += 1
        elif r < 0.55:
            view.insert_before("t", nid)
            model.insert(model.index("t"), nid)
            nid += 1
        elif r < 0.7:
            ref = rng.choice(model)
            view.insert_before(ref, nid)
            model.insert(model.index(ref), nid)
            nid += 1
        elif r < 0.85:
            x, ref = rng.sample(model, 2)
            view.move_after(x, ref)
            model.remove(x)
            model.insert(model.index(ref) + 1, x)
        elif len(model) > 2:
            x = rng.choice([m for m in model if m not in ("h", "t")])
            del view[x]
            model.remove(x)
        assert _max_key_len(view) <= view.MAX_KEY_LENGTH
    assert list(view) == model
    _assert_consistent(view)


class _NoRoomView(_TightView):
    """Window keys never fit the target, so every respace is the full fallback."""

    RESPACE_TARGET = 3


def test_window_respace_falls_back_to_full_respace(broot):
    ids = [chr(c) for c in range(ord("a"), ord("z") + 1)]
    view = broot.open_child("lst", _NoRoomView)
    view.store(ids)
    far = {"a": view.position("a"), "z": view.position("z")}
    for i in range(120):
        view.insert_after("m", f"x{i}")
    expected = [*ids[:13], *[f"x{i}" for i in reversed(range(120))], *ids[13:]]
    assert list(view) == expected
    assert _max_key_len(view) <= view.MAX_KEY_LENGTH
    # A full respace rewrites rows far outside any window too.
    assert view.position("a") != far["a"]
    assert view.position("z") != far["z"]
    _assert_consistent(view)


def test_window_respace_leaves_far_rows_alone(broot):
    ids = [chr(c) for c in range(ord("a"), ord("z") + 1)]
    view = broot.open_child("lst", _TightView)
    view.store(ids)
    far = {"a": view.position("a"), "z": view.position("z")}
    for i in range(120):
        view.insert_after("m", f"x{i}")
    assert list(view)[:14] == [*ids[:13], "x119"]
    assert _max_key_len(view) <= view.MAX_KEY_LENGTH
    assert view.position("a") == far["a"]
    assert view.position("z") == far["z"]
    _assert_consistent(view)


@pytest.mark.parametrize("backend", SEEKING_BACKENDS)
def test_respacing_insert_touches_are_bounded(backend, tmp_path):
    """A respacing insert costs about 2 * RESPACE_WINDOW rows, then inserts cost as before."""
    storage = _open_storage(backend, tmp_path)
    try:
        with storage.transaction() as tx:
            view = Navigator(storage).root(tx).open_child("lst", EagerFractionalIndexedDictView)
            view.store(["head", "tail"])
            i = 0
            while len(view.position("head" if i == 0 else f"c{i - 1}")) < view.MAX_KEY_LENGTH - 6:
                view.insert_after("head", f"c{i}")
                i += 1
        with storage.transaction() as tx:
            ctx = CountingContext(tx)
            view = Navigator(storage).root(ctx).open_child("lst", EagerFractionalIndexedDictView)
            ctx.reset()
            view.insert_after("head", "plain")
            plain = dict(ctx.counts)
            respacing = None
            for j in range(60):
                ctx.reset()
                view.insert_after("head", f"r{j}")
                if ctx.counts.get("delete", 0):
                    respacing = dict(ctx.counts)
                    break
            assert respacing is not None
            window = view.RESPACE_WINDOW
            assert respacing["delete"] <= 2 * window
            assert respacing["put"] <= 4 * window + 3
            assert respacing["scan"] <= 3
            assert respacing["scan_row"] <= 2 * window + 4
            assert sum(respacing.values()) <= 100
            assert len(view.position(f"r{j}")) <= view.RESPACE_TARGET
            # The next insert is an ordinary one again.
            ctx.reset()
            view.insert_after("head", "after")
            assert dict(ctx.counts) == plain
            assert _max_key_len(view) <= view.MAX_KEY_LENGTH
            _assert_consistent(view)
    finally:
        storage.close()


# ============================================================================
# PARTIAL STATES: ROWS MISSING BY HAND OR BY A FAILED WRITE
# ============================================================================


class _FailingWriteView(EagerFractionalIndexedDictView):
    def _write_value(self, address, value):
        if value == "boom":
            raise RuntimeError("boom")
        super()._write_value(address, value)


def test_failed_value_write_leaves_no_trace(broot):
    view = broot.open_child("lst", _FailingWriteView)
    view.store({"a": 1, "b": 2})
    with pytest.raises(RuntimeError, match="boom"):
        view.insert_after("a", "x", "boom")
    # The position written before the value is rolled back: x is absent.
    assert "x" not in view
    assert len(view) == 2
    assert list(view) == ["a", "b"]
    assert list(view.items()) == [("a", 1), ("b", 2)]
    assert view.extract() == {"a": 1, "b": 2}
    assert is_empty(view.get("x"))
    with pytest.raises(KeyError):
        view["x"]
    with pytest.raises(KeyError):
        view.position("x")
    with pytest.raises(KeyError):
        del view["x"]
    _assert_consistent(view)
    # A retry inserts it as if new and lands where asked.
    view.insert_after("a", "x", 3)
    assert list(view) == ["a", "x", "b"]
    assert view["x"] == 3
    _assert_consistent(view)


def test_failed_value_write_then_set_appends(broot):
    view = broot.open_child("lst", _FailingWriteView)
    view.store({"a": 1})
    with pytest.raises(RuntimeError):
        view.prepend("x", "boom")
    assert "x" not in view
    # No stale position, so a plain set treats x as new and appends it.
    view["x"] = 2
    assert list(view) == ["a", "x"]
    assert view["x"] == 2
    _assert_consistent(view)


def test_stale_position_without_order_row_is_repaired_by_rebuild(fdict):
    """A bypass write can leave pos without order; set does not fix that."""
    view = fdict("lst", {"a": 1, "b": 2})
    view._pos_container().put_child_primitive("x", "zz", validate=False, validate_parent=False)
    view._data_container().put_child_primitive("x", 3)
    assert "x" in view
    assert len(view) == 3
    assert list(view) == ["a", "b"]
    view["x"] = 4
    assert list(view) == ["a", "b"]
    view.rebuild_bookkeeping()
    assert list(view) == ["a", "b", "x"]
    assert view["x"] == 4
    _assert_consistent(view)


def test_data_without_position_is_appended_on_set(fdict):
    view = fdict("lst", {"a": 1, "b": 2, "c": 3})
    view._order_container().delete_child_primitive_unsafe(view.position("b"))
    view._pos_container().delete_child_primitive_unsafe("b")
    assert "b" not in view
    assert view["b"] == 2
    assert list(view) == ["a", "c"]
    view["b"] = 20
    assert list(view) == ["a", "c", "b"]
    assert view["b"] == 20
    _assert_consistent(view)


def test_data_without_position_is_appended_by_update_and_store(fdict):
    view = fdict("lst", {"a": 1, "b": 2, "c": 3})
    for lost in ("a", "c"):
        view._order_container().delete_child_primitive_unsafe(view.position(lost))
        view._pos_container().delete_child_primitive_unsafe(lost)
    assert list(view) == ["b"]
    view.update({"a": 10})
    view.store({"c": 30}, replace=False)
    assert list(view) == ["b", "a", "c"]
    _assert_consistent(view)


def test_order_row_without_data_is_skipped_by_items(fdict):
    view = fdict("lst", {"a": 1, "b": 2, "c": {"n": 3}})
    view._data_container().delete_child("b")
    view._data_container().delete_child("c")
    assert list(view) == ["a", "b", "c"]  # one scan, no per-row data check
    assert list(view.items()) == [("a", 1)]
    assert list(view.values()) == [1]
    assert list(reversed(view.values())) == [1]
    assert view.extract() == {"a": 1}
    assert list(view.lazy.items()) == [("a", 1)]
    view.rebuild_bookkeeping()
    assert list(view) == ["a"]
    _assert_consistent(view)


# ============================================================================
# CONCURRENCY: TWO WRITERS AT THE SAME GAP
# ============================================================================


def test_replayed_interleaving_at_same_gap_gives_unique_keys(fdict):
    """The interleaving ``insert_before`` goes through, replayed by hand in
    one transaction: neighbour lookup for both writers, then placement for
    both. Proves key uniqueness, not isolation; the thread cases below
    run real writers.
    """
    view = fdict("lst", {"a": 1, "b": 2})
    ref_okey = view.position("b")
    prev_x = view._neighbour(ref_okey, before=True)
    prev_y = view._neighbour(ref_okey, before=True)
    assert prev_x == prev_y
    okey_x = view._place("x", prev_x[0], ref_okey)
    okey_y = view._place("y", prev_y[0], ref_okey)
    view._write_value("x", 10)
    view._write_value("y", 20)
    assert okey_x != okey_y
    assert prev_x[0] < okey_x < ref_okey
    assert prev_y[0] < okey_y < ref_okey
    order = list(view)
    assert order[0] == "a"
    assert order[-1] == "b"
    assert set(order[1:3]) == {"x", "y"}
    assert order == sorted(order, key=view.position)
    assert len(view) == 4
    _assert_consistent(view)
    # Later inserts next to the racing keys are ordinary.
    view.insert_before(order[2], "z", 0)
    assert list(view) == [*order[:2], "z", *order[2:]]
    _assert_consistent(view)


def test_replayed_interleaving_appending_gives_unique_keys(fdict):
    """Same replay for ``append``: both read the last row, then both place."""
    view = fdict("lst", {"a": 1})
    last_x = view._edge(last=True)
    last_y = view._edge(last=True)
    okey_x = view._place("x", last_x[0], None)
    okey_y = view._place("y", last_y[0], None)
    view._write_value("x", 10)
    view._write_value("y", 20)
    assert okey_x != okey_y
    assert view.first() == "a"
    assert set(list(view)[1:]) == {"x", "y"}
    _assert_consistent(view)


def _run_writers(*targets, timeout=30.0):
    """Run each target on its own thread; re-raise the first error."""
    errors: list[BaseException] = []

    def wrap(fn):
        try:
            fn()
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=wrap, args=(fn,)) for fn in targets]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout)
    assert not any(t.is_alive() for t in threads), "writer thread stalled"
    if errors:
        raise errors[0]


def _retrying(fn, attempts=200):
    for _ in range(attempts):
        try:
            return fn()
        except (StorageTransactionConflictError, StorageLockTimeoutError):
            time.sleep(0.001)
    raise AssertionError("still conflicting")


@pytest.fixture(params=("sqlite", "lmdb"))
def shared_storage(request, tmp_path):
    storage = _open_storage(request.param, tmp_path)
    try:
        yield storage
    finally:
        storage.close()


def _writer(storage, barrier, op):
    """Meet the peer at the barrier once, then run the transaction with retries.

    The barrier sits outside the retried body, so a retry (a conflict or a
    lock timeout, which the slot's 60s wait makes rare) never waits on a
    barrier generation the peer has left.
    """

    def body():
        with storage.transaction() as tx:
            view = Navigator(storage).root(tx).open_child("lst", EagerFractionalIndexedDictView)
            op(view)

    def run():
        barrier.wait(timeout=5)
        _retrying(body)

    return run


def test_two_writer_threads_inserting_at_the_same_gap(shared_storage):
    """Two transactions on two threads insert before the same neighbour.

    The backend serializes them (one waits at begin or retries on
    conflict); both land, keys are unique, order and index agree.
    """
    storage = shared_storage
    with storage.transaction() as tx:
        Navigator(storage).root(tx).open_child("lst", EagerFractionalIndexedDictView).store(
            {"a": 1, "b": 2}
        )
    for round_ in range(5):
        barrier = threading.Barrier(2)
        x, y = f"x{round_}", f"y{round_}"
        _run_writers(
            _writer(storage, barrier, lambda v, k=x: v.insert_before("b", k, 10)),
            _writer(storage, barrier, lambda v, k=y: v.insert_before("b", k, 20)),
        )
        with storage.snapshot() as snap:
            view = Navigator(storage).root(snap).open_child("lst", EagerFractionalIndexedDictView)
            order = list(view)
            assert order[0] == "a"
            assert order[-1] == "b"
            assert {x, y} <= set(order[1:-1])
            assert view[x] == 10
            assert view[y] == 20
            assert len(view) == 2 + 2 * (round_ + 1)
            _assert_consistent(view)


def test_two_writer_threads_appending(shared_storage):
    storage = shared_storage
    with storage.transaction() as tx:
        Navigator(storage).root(tx).open_child("lst", EagerFractionalIndexedDictView).store(
            {"a": 1}
        )
    barrier = threading.Barrier(2)
    _run_writers(
        _writer(storage, barrier, lambda v: v.append("x", 10)),
        _writer(storage, barrier, lambda v: v.append("y", 20)),
    )
    with storage.snapshot() as snap:
        view = Navigator(storage).root(snap).open_child("lst", EagerFractionalIndexedDictView)
        assert view.first() == "a"
        assert set(list(view)[1:]) == {"x", "y"}
        assert len(view) == 3
        _assert_consistent(view)


def test_two_writer_threads_moving_and_deleting(shared_storage):
    """One writer moves a key while the other deletes its neighbour."""
    storage = shared_storage
    with storage.transaction() as tx:
        Navigator(storage).root(tx).open_child("lst", EagerFractionalIndexedDictView).store(
            {"a": 1, "b": 2, "c": 3, "d": 4}
        )
    barrier = threading.Barrier(2)
    _run_writers(
        _writer(storage, barrier, lambda v: v.move_before("d", "a")),
        _writer(storage, barrier, lambda v: v.__delitem__("b")),
    )
    with storage.snapshot() as snap:
        view = Navigator(storage).root(snap).open_child("lst", EagerFractionalIndexedDictView)
        assert list(view) == ["d", "a", "c"]
        _assert_consistent(view)


# ============================================================================
# HOOKS (mem with the in-memory transport)
# ============================================================================


@pytest.fixture
def observed_storage():
    transport = InMemoryTransport()
    publisher = InMemoryPublisher(transport)
    observer = InMemoryObserver(transport)
    storage = InMemoryStorage(codec=NoOpCodec(), publisher=publisher)
    publisher.connect()
    observer.connect()
    storage.open()
    try:
        yield storage, publisher, observer
    finally:
        storage.close()
        observer.disconnect()
        publisher.disconnect()


class _Hooks:
    def __init__(self, storage, publisher, observer, view_cls=EagerFractionalIndexedDictView):
        self.storage = storage
        self.publisher = publisher
        self.observer = observer
        self.view_cls = view_cls
        self.nav = Navigator(storage)
        self.hits: dict[str, list] = {}
        self.subs = []

    def view(self, ctx):
        return self.nav.root(ctx).open_child("lst", self.view_cls)

    def watch(self, name, options):
        hits: list = []
        self.hits[name] = hits
        sub = self.observer.subscribe(options)
        sub.bind(hits.append)
        self.subs.append(sub)

    def settle(self):
        self.publisher.flush(timeout=2.0)
        self.observer.flush(timeout=2.0)

    def run(self, fn):
        self.settle()
        for hits in self.hits.values():
            hits.clear()
        with self.storage.transaction() as tx:
            fn(self.view(tx))
        self.settle()
        return {name: sorted(hits) for name, hits in self.hits.items()}


def _order_keys(hits):
    return [k for k in hits if "__order__" in k]


def test_hooks_fire_on_add_delete_move(observed_storage):
    h = _Hooks(*observed_storage)
    with h.storage.transaction() as tx:
        h.view(tx).store({"a": 1, "b": 2, "c": 3})
    with h.storage.snapshot() as snap:
        view = h.view(snap)
        h.watch("children", view.on_children_change())
        h.watch("change", view.on_change())
        h.watch("child_b", view.on_child_change("b"))
        h.watch("desc", view.on_descendants_change("*", "n"))

    # Add via insert: exactly one __order__ row (no append-then-move).
    hits = h.run(lambda v: v.insert_before("b", "x", 5))
    assert len(hits["children"]) == 1
    assert len(_order_keys(hits["change"])) == 1
    assert any("__data__" in k and k[-1] == "x" for k in hits["change"])
    assert hits["child_b"] == []

    # Append via __setitem__: one __order__ row too.
    hits = h.run(lambda v: v.__setitem__("y", 6))
    assert len(hits["children"]) == 1

    # Update in place: data only, no order row.
    hits = h.run(lambda v: v.__setitem__("b", 20))
    assert hits["children"] == []
    assert len(hits["change"]) == 1
    assert len(hits["child_b"]) == 1

    # Move: order rows change (old deleted, new written), data untouched.
    hits = h.run(lambda v: v.move_before("c", "a"))
    assert len(hits["children"]) == 2
    assert len(hits["change"]) == 2
    assert all("__order__" in k for k in hits["change"])
    assert hits["child_b"] == []

    # Move of b does not fire on_child_change(b): it is a data watch.
    hits = h.run(lambda v: v.move_last("b"))
    assert len(hits["children"]) == 2
    assert hits["child_b"] == []

    # Delete: one order row gone, data gone.
    hits = h.run(lambda v: v.__delitem__("x"))
    assert len(hits["children"]) == 1
    assert any("__data__" in k and k[-1] == "x" for k in hits["change"])

    # Nested write under a child matches the descendants pattern.
    hits = h.run(lambda v: v.__setitem__("d", {"n": 1}))
    assert len(hits["desc"]) == 1
    assert hits["desc"][0][-2:] == ("d", "n")

    # No-op move: nothing fires.
    hits = h.run(lambda v: v.move_after("a", "c"))
    assert hits["children"] == []
    assert hits["change"] == []


def test_respacing_insert_fires_children_for_rewritten_rows_only(observed_storage):
    """A window respace reads as a move of each rewritten row; data never fires."""
    h = _Hooks(*observed_storage, view_cls=_TightView)
    with h.storage.transaction() as tx:
        h.view(tx).store(["head", "tail"])
    with h.storage.snapshot() as snap:
        view = h.view(snap)
        h.watch("children", view.on_children_change())
        h.watch("change", view.on_change())
    found = False
    for i in range(200):
        with h.storage.snapshot() as snap:
            before = dict(h.view(snap).positions())
        hits = h.run(lambda v, i=i: v.insert_after("head", f"c{i}"))
        with h.storage.snapshot() as snap:
            after = dict(h.view(snap).positions())
            order = list(h.view(snap))
        assert order[:2] == ["head", f"c{i}"]
        if len(hits["children"]) == 1:
            continue
        found = True
        rewritten = {addr for okey, addr in before.items() if after.get(okey) != addr}
        assert 1 <= len(rewritten) <= 2 * view.RESPACE_WINDOW
        # one delete + one put per rewritten row, one put for the new id
        assert len(hits["children"]) == 2 * len(rewritten) + 1
        data_hits = [k for k in hits["change"] if "__data__" in k]
        assert data_hits == [(*view.container.site, "__data__", f"c{i}")]
        break
    assert found


# ============================================================================
# KEY-TOUCH COUNTS: CONSTANT PER OP REGARDLESS OF SIZE
# ============================================================================


class _CountingScan:
    def __init__(self, scan, counts):
        self._scan = scan
        self._counts = counts

    def items(self):
        for row in self._scan.items():
            self._counts["scan_row"] += 1
            yield row

    def keys(self):
        for row in self._scan.keys():
            self._counts["scan_row"] += 1
            yield row

    def values(self):
        for row in self._scan.values():
            self._counts["scan_row"] += 1
            yield row


class CountingContext:
    """Transaction wrapper that counts every storage key touched."""

    def __init__(self, ctx):
        self._ctx = ctx
        self.counts: Counter = Counter()

    def reset(self):
        self.counts = Counter()

    @property
    def storage(self):
        return self._ctx.storage

    @property
    def is_closed(self):
        return self._ctx.is_closed

    @property
    def is_active(self):
        return self._ctx.is_active

    def get(self, key):
        self.counts["get"] += 1
        return self._ctx.get(key)

    def exists(self, key):
        self.counts["get"] += 1
        return self._ctx.exists(key)

    def multiget(self, keys):
        self.counts["get"] += len(keys)
        return self._ctx.multiget(keys)

    def scan(self, options):
        self.counts["scan"] += 1
        return _CountingScan(self._ctx.scan(options), self.counts)

    def put(self, key, value):
        self.counts["put"] += 1
        self._ctx.put(key, value)

    def delete(self, key):
        self.counts["delete"] += 1
        self._ctx.delete(key)

    def commit(self):
        self._ctx.commit()

    def abort(self):
        self._ctx.abort()

    def __enter__(self):
        self._ctx.__enter__()
        return self

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)

    def __getattr__(self, name):
        return getattr(self._ctx, name)


def _mid(n):
    return f"id{n // 2:05d}"


def _ops(n):
    """Named ops to measure; each gets a fresh id so runs do not interfere."""
    return {
        "contains": lambda v: _mid(n) in v,
        "contains_missing": lambda v: "nope" in v,
        "get": lambda v: v[_mid(n)],
        "first": lambda v: v.first(),
        "last": lambda v: v.last(),
        "next_after": lambda v: v.next_after(_mid(n)),
        "prev_before": lambda v: v.prev_before(_mid(n)),
        "keys_after_limit3": lambda v: list(v.keys(after=_mid(n), limit=3)),
        "keys_reverse_limit3": lambda v: list(v.keys_reverse(before=_mid(n), limit=3)),
        "set_existing": lambda v: v.__setitem__(_mid(n), -1),
        "set_new": lambda v: v.__setitem__("set_new", -1),
        "append": lambda v: v.append("app", -1),
        "prepend": lambda v: v.prepend("pre", -1),
        "insert_before": lambda v: v.insert_before(_mid(n), "ib", -1),
        "insert_after": lambda v: v.insert_after(_mid(n), "ia", -1),
        "insert_nested": lambda v: v.insert_after(_mid(n), "in", {"k": 1}),
        "move_before": lambda v: v.move_before("id00001", _mid(n)),
        "move_after": lambda v: v.move_after("id00002", _mid(n)),
        "move_first": lambda v: v.move_first("id00003"),
        "move_last": lambda v: v.move_last("id00004"),
        "delete": lambda v: v.__delitem__("id00005"),
    }


def _measure(storage, n):
    with storage.transaction() as tx:
        root = Navigator(storage).root(tx)
        view = root.open_child("lst", EagerFractionalIndexedDictView)
        view.store({f"id{i:05d}": i for i in range(n)})
    with storage.transaction() as tx:
        ctx = CountingContext(tx)
        view = Navigator(storage).root(ctx).open_child("lst", EagerFractionalIndexedDictView)
        result = {}
        for name, op in _ops(n).items():
            ctx.reset()
            op(view)
            result[name] = dict(ctx.counts)
        assert len(view) == n + 5
    return result


@pytest.mark.parametrize("backend", SEEKING_BACKENDS)
def test_key_touches_are_constant_in_n(backend, tmp_path):
    small = _open_storage(backend, tmp_path / "small")
    try:
        at_10 = _measure(small, 10)
    finally:
        small.close()
    large = _open_storage(backend, tmp_path / "large")
    try:
        at_1000 = _measure(large, 1000)
    finally:
        large.close()
    # An op that takes an order row away seeks once more, to tell whether
    # the row was the last one (the end mark).
    retiring = {"move_before", "move_after", "move_first", "move_last", "delete"}
    for name in at_10:
        assert at_10[name] == at_1000[name], (name, at_10[name], at_1000[name])
        max_scans = 2 if name in retiring else 1
        assert at_10[name].get("scan", 0) <= max_scans, (name, at_10[name])
        assert at_10[name].get("scan_row", 0) <= 5, (name, at_10[name])
        assert sum(at_10[name].values()) <= 30, (name, at_10[name])
    print(f"\n{backend} key touches per op (n=10 == n=1000):")
    for name, counts in at_10.items():
        total = sum(counts.values())
        print(f"  {name:20s} total={total:2d}  {counts}")


# ============================================================================
# HARDENING: bool, repairs through every set door, key collisions, partial inserts
# ============================================================================


def test_bool_on_absent_view_raises_and_writes_nothing(broot):
    view = broot.open_child("absent", EagerFractionalIndexedDictView)
    with pytest.raises(ContainerNotFoundError):
        bool(view)
    assert not view.container.exists()
    view.store({"a": 1})
    assert view
    view.clear()
    assert not view
    assert len(view) == 0


@pytest.mark.parametrize("backend", SEEKING_BACKENDS)
def test_bool_cost_is_one_seek(backend, tmp_path, monkeypatch):
    from tests.support.counting import install_counter, measure, open_backend

    storage = open_backend(backend, tmp_path)
    try:
        counts = install_counter(storage, monkeypatch)
        costs = {}
        for n in (10, 1000):
            with storage.transaction() as tx:
                view = Navigator(storage).root(tx).open_child("lst", EagerFractionalIndexedDictView)
                view.store({f"k{i}": i for i in range(n)})
            costs[n] = measure(
                storage,
                counts,
                lambda root: bool(root.open_child("lst", EagerFractionalIndexedDictView)),
            )
        assert costs[10] == costs[1000], costs
        assert costs[10]["scan"] == 1, costs
        assert costs[10]["scan_rows"] <= 2, costs
    finally:
        storage.close()


def _drop_position(view, address):
    view._order_container().delete_child_primitive_unsafe(view.position(address))
    view._pos_container().delete_child_primitive_unsafe(address)


def test_data_without_position_is_appended_by_set_child_container_as(fdict):
    view = fdict("lst", {"a": {"n": 1}, "b": {"n": 2}, "c": {"n": 3}})
    _drop_position(view, "b")
    assert "b" not in view
    assert list(view) == ["a", "c"]
    view.set_child_container_as("b", {"n": 20}, EagerDictView)
    assert list(view) == ["a", "c", "b"]
    assert "b" in view
    assert view["b"] == {"n": 20}
    assert len(view) == 3
    _assert_consistent(view)


def test_data_without_position_is_appended_by_set_primitive(fdict):
    view = fdict("lst", {"a": 1, "b": 2, "c": 3})
    _drop_position(view, "b")
    view.set_primitive("b", [1, 2])
    assert list(view) == ["a", "c", "b"]
    assert view["b"] == [1, 2]
    _assert_consistent(view)


def test_set_child_container_as_keeps_existing_position(fdict):
    view = fdict("lst", {"a": {"n": 1}, "b": {"n": 2}, "c": {"n": 3}})
    okey = view.position("b")
    view.set_child_container_as("b", {"n": 20}, EagerDictView)
    assert view.position("b") == okey
    assert list(view) == ["a", "b", "c"]
    view.set_child_container_as("d", {"n": 4}, EagerDictView)
    assert list(view) == ["a", "b", "c", "d"]
    _assert_consistent(view)


def test_same_suffix_draw_in_one_gap_gets_a_fresh_key(fdict, monkeypatch):
    """Two keys drawn from the same neighbours with the same random tail do not collide."""
    from virtuals._views import fractional_index as fi

    view = fdict("lst", {"a": 1, "b": 2})
    prev = view.position("a")
    nxt = view.position("b")
    draws = iter(["ZZZZ", "ZZZZ", "ZZZZ", "AAAA"])
    monkeypatch.setattr(fi, "_random_suffix", lambda n: next(draws))
    first = view._place("x", prev, nxt)
    second = view._place("y", prev, nxt)
    view._write_value("x", 10)
    view._write_value("y", 20)
    assert first != second
    assert prev < first < nxt
    assert prev < second < nxt
    assert list(view) == ["a", "y", "x", "b"]
    assert len(view) == 4
    _assert_consistent(view)


def test_allot_gives_up_when_every_draw_is_taken(fdict, monkeypatch):
    from virtuals._views import fractional_index as fi

    view = fdict("lst", {"a": 1, "b": 2})
    prev = view.position("a")
    nxt = view.position("b")
    monkeypatch.setattr(fi, "_random_suffix", lambda n: "ZZZZ")
    # Serialised inserts never collide (the second sees the first as its
    # neighbour); only a placement against stale neighbours can.
    view.insert_after("a", "x")
    view.insert_after("a", "y")
    assert list(view) == ["a", "y", "x", "b"]
    with pytest.raises(RuntimeError, match="No free order key"):
        view._place("z", prev, nxt)
    assert "z" not in view
    assert list(view) == ["a", "y", "x", "b"]
    _assert_consistent(view)


class _HalfWritingView(EagerFractionalIndexedDictView):
    """Writes part of a compound value, then fails: a nested write dying half way."""

    def _write_value(self, address, value):
        if isinstance(value, dict) and "boom" in value:
            super()._write_value(address, {k: v for k, v in value.items() if k != "boom"})
            raise RuntimeError("boom")
        super()._write_value(address, value)


def test_failed_compound_write_leaves_no_data(broot):
    view = broot.open_child("lst", _HalfWritingView)
    view.store({"a": 1, "b": 2})
    with pytest.raises(RuntimeError, match="boom"):
        view.insert_after("a", "x", {"ok": 1, "boom": 2})
    assert "x" not in view
    assert len(view) == 2
    assert list(view) == ["a", "b"]
    assert is_empty(view.get("x"))
    assert not view._data_container().exists_child("x")
    assert list(view.items()) == [("a", 1), ("b", 2)]
    _assert_consistent(view)
    # A plain set after the failure appends, with nothing left over from the failed value.
    view["x"] = {"fresh": True}
    assert list(view) == ["a", "b", "x"]
    assert view["x"] == {"fresh": True}
    _assert_consistent(view)


def test_failed_compound_write_over_stale_data_leaves_no_data(broot):
    """Stale data without a position is dropped too when the overwrite fails."""
    view = broot.open_child("lst", _HalfWritingView)
    view.store({"a": 1, "x": {"old": 1}})
    _drop_position(view, "x")
    with pytest.raises(RuntimeError, match="boom"):
        view.append("x", {"new": 1, "boom": 2})
    assert "x" not in view
    assert is_empty(view.get("x"))
    assert not view._data_container().exists_child("x")
    _assert_consistent(view)
