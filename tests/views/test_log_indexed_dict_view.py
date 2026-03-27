"""Tests for LogIndexedDictView (Eager and Lazy facets)."""

import pytest

from virtuals._views import EagerLogIndexedDictView, LazyLogIndexedDictView


# ============================================================================
# FIXTURE
# ============================================================================


@pytest.fixture
def log_dict_factory(root_view):
    """Factory for creating EagerLogIndexedDictViews with test data."""

    def _create(address, data=None):
        view = root_view.open_child(address, EagerLogIndexedDictView)
        if data is not None:
            view.store(data)
        return view

    return _create


# ============================================================================
# BASIC OPERATIONS
# ============================================================================


def test_factory_creates_empty_view(log_dict_factory):
    dct = log_dict_factory("dct")
    assert dct is not None
    assert len(dct) == 0


def test_factory_with_data(log_dict_factory):
    dct = log_dict_factory("dct", {"a": 1, "b": 2, "c": 3})
    extracted = dct.extract()
    assert extracted == {"a": 1, "b": 2, "c": 3}


def test_set_and_get(log_dict_factory):
    dct = log_dict_factory("dct")
    dct["alice"] = 100
    dct["bob"] = 95
    assert dct["alice"] == 100
    assert dct["bob"] == 95


def test_nested_set_and_get(log_dict_factory):
    dct = log_dict_factory("dct")
    dct["user"] = {"name": "Alice", "age": 30}
    result = dct["user"]
    assert isinstance(result, dict)
    assert result == {"name": "Alice", "age": 30}


def test_delete(log_dict_factory):
    dct = log_dict_factory("dct", {"a": 1, "b": 2, "c": 3})
    del dct["b"]
    assert "b" not in dct
    assert "a" in dct
    assert "c" in dct
    assert len(dct) == 2


def test_delete_missing_raises(log_dict_factory):
    dct = log_dict_factory("dct", {"a": 1})
    with pytest.raises(KeyError):
        del dct["missing"]


def test_contains(log_dict_factory):
    dct = log_dict_factory("dct", {"a": 1, "b": 2})
    assert "a" in dct
    assert "c" not in dct


def test_len(log_dict_factory):
    dct = log_dict_factory("dct")
    assert len(dct) == 0
    dct["a"] = 1
    assert len(dct) == 1
    dct["b"] = 2
    assert len(dct) == 2


def test_iteration_returns_keys_in_insertion_order(log_dict_factory):
    dct = log_dict_factory("dct")
    dct["c"] = 3
    dct["a"] = 1
    dct["b"] = 2
    keys = list(dct)
    assert keys == ["c", "a", "b"]


def test_clear(log_dict_factory):
    dct = log_dict_factory("dct", {"a": 1, "b": 2})
    dct.clear()
    assert len(dct) == 0
    assert list(dct) == []


def test_update(log_dict_factory):
    dct = log_dict_factory("dct", {"a": 1})
    dct.update({"b": 2, "c": 3})
    assert dct["a"] == 1
    assert dct["b"] == 2
    assert dct["c"] == 3


def test_get_with_default(log_dict_factory):
    dct = log_dict_factory("dct", {"a": 1})
    assert dct.get("a") == 1
    assert dct.get("missing", 42) == 42


def test_pop(log_dict_factory):
    dct = log_dict_factory("dct", {"a": 1, "b": 2})
    val = dct.pop("a")
    assert val == 1
    assert "a" not in dct
    assert len(dct) == 1


def test_pop_missing_with_default(log_dict_factory):
    dct = log_dict_factory("dct", {"a": 1})
    val = dct.pop("missing", 99)
    assert val == 99


def test_pop_missing_raises(log_dict_factory):
    dct = log_dict_factory("dct", {"a": 1})
    with pytest.raises(KeyError):
        dct.pop("missing")


# ============================================================================
# CURSOR ITERATION
# ============================================================================


def test_keys_with_log_keys(log_dict_factory):
    dct = log_dict_factory("dct")
    dct["x"] = 10
    dct["y"] = 20
    dct["z"] = 30

    entries = list(dct.keys_with_log_keys())
    assert len(entries) == 3
    # Each entry is (log_key, actual_key)
    actual_keys = [ak for _lk, ak in entries]
    assert actual_keys == ["x", "y", "z"]
    # Log keys should be monotonically increasing strings
    log_keys = [lk for lk, _ak in entries]
    assert log_keys == sorted(log_keys)


def test_keys_with_log_keys_after_cursor(log_dict_factory):
    dct = log_dict_factory("dct")
    dct["a"] = 1
    dct["b"] = 2
    dct["c"] = 3

    entries = list(dct.keys_with_log_keys())
    # Use second entry's log_key as cursor
    cursor = entries[1][0]
    remaining = list(dct.keys_with_log_keys(after=cursor))
    assert len(remaining) == 1
    assert remaining[0][1] == "c"


def test_keys_after_returns_generator(log_dict_factory):
    dct = log_dict_factory("dct")
    dct["a"] = 1
    dct["b"] = 2
    dct["c"] = 3

    entries = list(dct.keys_with_log_keys())
    cursor = entries[0][0]

    result = list(dct.keys(after=cursor))
    assert result == ["b", "c"]


def test_cursor_after_last_returns_empty(log_dict_factory):
    dct = log_dict_factory("dct")
    dct["a"] = 1

    entries = list(dct.keys_with_log_keys())
    cursor = entries[-1][0]
    remaining = list(dct.keys_with_log_keys(after=cursor))
    assert remaining == []


# ============================================================================
# SET_PRIMITIVE (controlled granularity)
# ============================================================================


def test_set_primitive_stores_compound_as_blob(log_dict_factory):
    dct = log_dict_factory("dct")
    compound = {"vsol": 100, "vtok": 200, "rsol": 50, "rtok": 150}
    dct.set_primitive("curve", compound)
    # Should be retrievable
    assert dct["curve"] == compound
    assert "curve" in dct
    assert len(dct) == 1


def test_set_primitive_list(log_dict_factory):
    dct = log_dict_factory("dct")
    dct.set_primitive("accounts", ["addr1", "addr2", "addr3"])
    assert dct["accounts"] == ["addr1", "addr2", "addr3"]


def test_set_primitive_appears_in_keys(log_dict_factory):
    dct = log_dict_factory("dct")
    dct.set_primitive("x", [1, 2, 3])
    dct["y"] = 42
    keys = list(dct)
    assert keys == ["x", "y"]


# ============================================================================
# FACET NAVIGATION
# ============================================================================


def test_eager_to_lazy_facet(log_dict_factory):
    dct = log_dict_factory("dct", {"a": {"nested": 1}})
    lazy = dct.lazy
    assert isinstance(lazy, LazyLogIndexedDictView)
    # Lazy should return a view for container children
    child = lazy["a"]
    assert not isinstance(child, dict)  # should be a View, not extracted


def test_lazy_to_eager_facet(log_dict_factory):
    dct = log_dict_factory("dct", {"a": {"nested": 1}})
    lazy = dct.lazy
    eager = lazy.eager
    assert isinstance(eager, EagerLogIndexedDictView)
    assert eager["a"] == {"nested": 1}


# ============================================================================
# STORE REPLACE
# ============================================================================


def test_store_replaces_existing(log_dict_factory):
    dct = log_dict_factory("dct", {"a": 1, "b": 2})
    dct.store({"x": 10, "y": 20})
    assert set(dct) == {"x", "y"}
    assert dct["x"] == 10


def test_store_no_replace(log_dict_factory):
    dct = log_dict_factory("dct", {"a": 1})
    dct.store({"b": 2, "c": 3}, replace=False)
    assert set(dct) == {"a", "b", "c"}


# ============================================================================
# ORDERING PRESERVED
# ============================================================================


# ============================================================================
# NEXT_KEY_AFTER (micro-snapshot primitive)
# ============================================================================


def test_next_key_after_from_start(log_dict_factory):
    dct = log_dict_factory("dct")
    dct["a"] = 1
    dct["b"] = 2
    dct["c"] = 3

    result = dct.next_key_after(None)
    assert result is not None
    log_key, actual_key = result
    assert actual_key == "a"
    assert isinstance(log_key, str)


def test_next_key_after_advances(log_dict_factory):
    dct = log_dict_factory("dct")
    dct["x"] = 10
    dct["y"] = 20
    dct["z"] = 30

    # Walk through all items one by one
    r1 = dct.next_key_after(None)
    assert r1[1] == "x"

    r2 = dct.next_key_after(r1[0])
    assert r2[1] == "y"

    r3 = dct.next_key_after(r2[0])
    assert r3[1] == "z"

    r4 = dct.next_key_after(r3[0])
    assert r4 is None


def test_next_key_after_empty_view(log_dict_factory):
    dct = log_dict_factory("dct")
    assert dct.next_key_after(None) is None


def test_next_key_after_single_item(log_dict_factory):
    dct = log_dict_factory("dct")
    dct["only"] = 42

    r1 = dct.next_key_after(None)
    assert r1[1] == "only"

    r2 = dct.next_key_after(r1[0])
    assert r2 is None


# ============================================================================
# ORDERING PRESERVED
# ============================================================================


def test_ordering_preserved_across_many_inserts(log_dict_factory):
    """Keys should come back in insertion order (chronological)."""
    dct = log_dict_factory("dct")
    expected = []
    for i in range(20):
        key = f"key_{i:03d}"
        dct[key] = i
        expected.append(key)
    assert list(dct) == expected
