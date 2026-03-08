"""End-to-end tests for IndexedDictView (Eager and Lazy facets)."""

from itertools import islice

from virtuals._views import EagerIndexedDictView, LazyIndexedDictView
from virtuals.view import ViewBase


def test_indexed_dict_factory_creates_empty_view(indexed_dict_factory):
    dct = indexed_dict_factory("dct")
    assert dct is not None
    assert len(dct) == 0


def test_indexed_dict_factory_with_data(indexed_dict_factory):
    dct = indexed_dict_factory("dct", {"a": 1, "b": 2, "c": 3})
    extracted = dct.extract()
    assert extracted == {"a": 1, "b": 2, "c": 3}


# ============================================================================
# BASIC OPERATIONS
# ============================================================================


def test_indexed_dict_set_and_get(indexed_dict_factory):
    dct = indexed_dict_factory("dct")
    dct["alice"] = 100
    dct["bob"] = 95
    assert dct["alice"] == 100
    assert dct["bob"] == 95


def test_indexed_dict_nested_set_and_get(indexed_dict_factory):
    dct = indexed_dict_factory("dct")
    dct["user"] = {"name": "Alice", "age": 30}
    result = dct["user"]
    assert isinstance(result, dict)
    assert result == {"name": "Alice", "age": 30}


def test_indexed_dict_delete(indexed_dict_factory):
    dct = indexed_dict_factory("dct", {"a": 1, "b": 2, "c": 3})
    del dct["b"]
    assert "b" not in dct
    assert "a" in dct
    assert "c" in dct
    assert len(dct) == 2


def test_indexed_dict_delete_missing_raises(indexed_dict_factory):
    import pytest

    dct = indexed_dict_factory("dct", {"a": 1})
    with pytest.raises(KeyError):
        del dct["missing"]


def test_indexed_dict_contains(indexed_dict_factory):
    dct = indexed_dict_factory("dct", {"a": 1, "b": 2})
    assert "a" in dct
    assert "c" not in dct


def test_indexed_dict_len(indexed_dict_factory):
    dct = indexed_dict_factory("dct")
    assert len(dct) == 0

    dct["a"] = 1
    assert len(dct) == 1

    dct["b"] = 2
    assert len(dct) == 2


def test_indexed_dict_iteration(indexed_dict_factory):
    dct = indexed_dict_factory("dct", {"a": 1, "b": 2, "c": 3})
    keys = list(dct)
    assert set(keys) == {"a", "b", "c"}


# ============================================================================
# KEY ORDERING
# ============================================================================


def test_indexed_dict_key_order_preserved(indexed_dict_factory):
    """Keys are stored in insertion order via __keys__ FlatListView."""
    dct = indexed_dict_factory("dct")
    dct["charlie"] = 3
    dct["alice"] = 1
    dct["bob"] = 2
    keys = list(dct.keys())
    assert keys == ["charlie", "alice", "bob"]


def test_indexed_dict_key_at(indexed_dict_factory):
    dct = indexed_dict_factory("dct")
    dct["first"] = 1
    dct["second"] = 2
    dct["third"] = 3
    assert dct.key_at(0) == "first"
    assert dct.key_at(1) == "second"
    assert dct.key_at(2) == "third"


# ============================================================================
# DICT METHODS
# ============================================================================


def test_indexed_dict_keys(indexed_dict_factory):
    dct = indexed_dict_factory("dct", {"a": 1, "b": 2, "c": 3})
    keys = list(dct.keys())
    assert set(keys) == {"a", "b", "c"}


def test_indexed_dict_values(indexed_dict_factory):
    dct = indexed_dict_factory("dct", {"a": 1, "b": 2, "c": 3})
    values = list(dct.values())
    assert set(values) == {1, 2, 3}


def test_indexed_dict_items(indexed_dict_factory):
    dct = indexed_dict_factory("dct", {"a": 1, "b": 2})
    items = dict(dct.items())
    assert items == {"a": 1, "b": 2}


def test_indexed_dict_get_with_default(indexed_dict_factory):
    dct = indexed_dict_factory("dct", {"a": 1})
    assert dct.get("a") == 1
    assert dct.get("missing", 42) == 42


def test_indexed_dict_pop(indexed_dict_factory):
    dct = indexed_dict_factory("dct", {"a": 1, "b": 2})
    value = dct.pop("a")
    assert value == 1
    assert "a" not in dct
    assert len(dct) == 1


def test_indexed_dict_pop_missing_with_default(indexed_dict_factory):
    dct = indexed_dict_factory("dct")
    value = dct.pop("missing", -1)
    assert value == -1


def test_indexed_dict_update(indexed_dict_factory):
    dct = indexed_dict_factory("dct", {"a": 1})
    dct.update({"b": 2, "c": 3})
    assert dct["a"] == 1
    assert dct["b"] == 2
    assert dct["c"] == 3


def test_indexed_dict_clear(indexed_dict_factory):
    dct = indexed_dict_factory("dct", {"a": 1, "b": 2, "c": 3})
    dct.clear()
    assert len(dct) == 0
    assert dct.extract() == {}


# ============================================================================
# NESTED DATA
# ============================================================================


def test_indexed_dict_nested_dicts(indexed_dict_factory):
    dct = indexed_dict_factory(
        "dct",
        {
            "users": {
                "alice": {"name": "Alice"},
                "bob": {"name": "Bob"},
            },
        },
    )
    extracted = dct.extract()
    assert extracted["users"]["alice"]["name"] == "Alice"
    assert extracted["users"]["bob"]["name"] == "Bob"


def test_indexed_dict_nested_lists(indexed_dict_factory):
    dct = indexed_dict_factory(
        "dct",
        {
            "tags": ["python", "rust", "go"],
        },
    )
    extracted = dct.extract()
    assert extracted["tags"] == ["python", "rust", "go"]


def test_indexed_dict_mixed_types(indexed_dict_factory):
    dct = indexed_dict_factory(
        "dct",
        {
            "string": "hello",
            "number": 42,
            "nested": {"key": "value"},
            "list": [1, 2, 3],
        },
    )
    extracted = dct.extract()
    assert extracted["string"] == "hello"
    assert extracted["number"] == 42
    assert extracted["nested"] == {"key": "value"}
    assert extracted["list"] == [1, 2, 3]


# ============================================================================
# STORE AND EXTRACT
# ============================================================================


def test_indexed_dict_store_replaces_content(indexed_dict_factory):
    dct = indexed_dict_factory("dct", {"a": 1, "b": 2})
    dct.store({"c": 3, "d": 4})
    assert "a" not in dct
    assert dct.extract() == {"c": 3, "d": 4}


def test_indexed_dict_extract_returns_dict(indexed_dict_factory):
    dct = indexed_dict_factory("dct", {"x": 10, "y": 20})
    extracted = dct.extract()
    assert isinstance(extracted, dict)
    assert extracted == {"x": 10, "y": 20}


# ============================================================================
# EAGER/LAZY FACETS
# ============================================================================


class TestEagerIndexedDictView:
    def test_eager_property_is_identity(self, indexed_dict_factory):
        dct = indexed_dict_factory("dct", {"a": 1})
        assert dct.eager is dct

    def test_lazy_property_returns_lazy(self, indexed_dict_factory):
        dct = indexed_dict_factory("dct", {"a": 1})
        lazy = dct.lazy
        assert isinstance(lazy, LazyIndexedDictView)

    def test_getitem_returns_value(self, indexed_dict_factory):
        dct = indexed_dict_factory("dct", {"a": 1})
        assert dct["a"] == 1

    def test_getitem_nested_returns_dict(self, indexed_dict_factory):
        dct = indexed_dict_factory("dct", {"user": {"name": "Alice"}})
        result = dct["user"]
        assert isinstance(result, dict)
        assert result == {"name": "Alice"}


class TestLazyIndexedDictView:
    def test_lazy_property_is_identity(self, indexed_dict_factory):
        dct = indexed_dict_factory("dct", {"a": 1})
        lazy = dct.lazy
        assert lazy.lazy is lazy

    def test_eager_property_returns_eager(self, indexed_dict_factory):
        dct = indexed_dict_factory("dct", {"a": 1})
        lazy = dct.lazy
        eager = lazy.eager
        assert isinstance(eager, EagerIndexedDictView)

    def test_getitem_primitive_returns_value(self, indexed_dict_factory):
        dct = indexed_dict_factory("dct", {"a": 42})
        lazy = dct.lazy
        assert lazy["a"] == 42

    def test_getitem_container_returns_view(self, indexed_dict_factory):
        dct = indexed_dict_factory("dct", {"user": {"name": "Alice"}})
        lazy = dct.lazy
        child = lazy["user"]
        assert isinstance(child, ViewBase)
        assert child.extract() == {"name": "Alice"}

    def test_values_yields_views_for_containers(self, indexed_dict_factory):
        dct = indexed_dict_factory("dct", {"a": {"x": 1}, "b": {"x": 2}})
        lazy = dct.lazy
        values = list(lazy.values())
        assert len(values) == 2
        assert all(isinstance(v, ViewBase) for v in values)

    def test_values_yields_primitives_for_leaves(self, indexed_dict_factory):
        dct = indexed_dict_factory("dct", {"x": 42, "y": "hello"})
        lazy = dct.lazy
        values = list(lazy.values())
        assert set(values) == {42, "hello"}

    def test_items_yields_key_view_pairs(self, indexed_dict_factory):
        dct = indexed_dict_factory("dct", {"user": {"name": "Alice"}})
        lazy = dct.lazy
        items = list(lazy.items())
        assert len(items) == 1
        key, view = items[0]
        assert key == "user"
        assert isinstance(view, ViewBase)

    def test_islice_on_lazy_values(self, indexed_dict_factory):
        dct = indexed_dict_factory(
            "dct", {"a": {"x": 1}, "b": {"x": 2}, "c": {"x": 3}, "d": {"x": 4}}
        )
        first_2 = list(islice(dct.lazy.values(), 2))
        assert len(first_2) == 2
        assert all(isinstance(v, ViewBase) for v in first_2)


class TestIndexedDictCrossNavigation:
    def test_eager_to_lazy_to_eager_roundtrip(self, indexed_dict_factory):
        dct = indexed_dict_factory("dct", {"key": "value"})
        lazy = dct.lazy
        eager_again = lazy.eager
        assert eager_again["key"] == "value"

    def test_lazy_mutation_visible_from_eager(self, indexed_dict_factory):
        dct = indexed_dict_factory("dct")
        lazy = dct.lazy
        lazy["new"] = "value"
        assert dct["new"] == "value"

    def test_eager_mutation_visible_from_lazy(self, indexed_dict_factory):
        dct = indexed_dict_factory("dct")
        dct["key"] = 42
        assert dct.lazy["key"] == 42
