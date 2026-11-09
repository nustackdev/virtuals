"""Functional tests for DictView."""

import pytest

from virtuals._views import EagerDictView
from virtuals.container import ContainerNotFoundError


def test_dict_factory_creates_empty_view(dict_factory):
    """Verify dict_factory creates an empty DictView."""
    users = dict_factory("users")
    assert users is not None
    assert len(users) == 0


def test_dict_factory_with_data(dict_factory):
    """Verify dict_factory can populate initial data."""
    users = dict_factory("users", {"alice": {"name": "Alice", "age": 30}})

    data = users.extract()
    assert "alice" in data
    assert data["alice"]["name"] == "Alice"
    assert data["alice"]["age"] == 30


# ============================================================================
# BASIC OPERATIONS
# ============================================================================


def test_dict_set_and_get(dict_factory):
    """Test setting and getting values."""
    users = dict_factory("users")

    users["alice"] = {"name": "Alice", "age": 30}
    users["bob"] = {"name": "Bob", "age": 25}

    assert users["alice"] == {"name": "Alice", "age": 30}
    assert users["bob"] == {"name": "Bob", "age": 25}


def test_dict_delete_item(dict_factory):
    """Test deleting items."""
    users = dict_factory("users", {"alice": 1, "bob": 2, "charlie": 3})

    del users["bob"]

    assert "alice" in users
    assert "bob" not in users
    assert "charlie" in users
    assert len(users) == 2


def test_dict_contains(dict_factory):
    """Test membership checking."""
    users = dict_factory("users", {"alice": 1, "bob": 2})

    assert "alice" in users
    assert "bob" in users
    assert "charlie" not in users


def test_dict_len(dict_factory):
    """Test length tracking."""
    users = dict_factory("users")

    assert len(users) == 0

    users["alice"] = 1
    assert len(users) == 1

    users["bob"] = 2
    assert len(users) == 2

    del users["alice"]
    assert len(users) == 1


# ============================================================================
# DICT METHODS
# ============================================================================


def test_dict_keys(dict_factory):
    """Test keys() iteration."""
    users = dict_factory("users", {"alice": 1, "bob": 2, "charlie": 3})

    keys = set(users.keys())
    assert keys == {"alice", "bob", "charlie"}


def test_dict_values(dict_factory):
    """Test values() iteration."""
    users = dict_factory("users", {"a": 1, "b": 2, "c": 3})

    values = set(users.values())
    assert values == {1, 2, 3}


def test_dict_items(dict_factory):
    """Test items() iteration."""
    users = dict_factory("users", {"alice": 1, "bob": 2})

    items = dict(users.items())
    assert items == {"alice": 1, "bob": 2}


def test_dict_reversed_values(dict_factory):
    """reversed(view.values()) yields values in reverse of forward order."""
    users = dict_factory("users", {"a": 1, "b": 2, "c": 3, "d": 4})

    forward = list(users.values())
    reverse = list(reversed(users.values()))

    assert reverse == list(reversed(forward))


def test_dict_reversed_items(dict_factory):
    """reversed(view.items()) yields items in reverse of forward order."""
    users = dict_factory(
        "users",
        {"a": 1, "b": {"nested": True}, "c": 3, "d": [1, 2]},
    )

    forward = list(users.items())
    reverse = list(reversed(users.items()))

    assert reverse == list(reversed(forward))


def test_dict_reversed_lazy_values(dict_factory):
    """reversed on lazy facet values keeps identity with reversed(forward)."""
    users = dict_factory("users", {"a": 1, "b": 2, "c": 3})

    forward = list(users.lazy.values())
    reverse = list(reversed(users.lazy.values()))
    assert reverse == list(reversed(forward))


def test_dict_reversed_lazy_items(dict_factory):
    """reversed on lazy facet items keeps identity with reversed(forward)."""
    users = dict_factory("users", {"a": 1, "b": 2, "c": 3})

    forward = list(users.lazy.items())
    reverse = list(reversed(users.lazy.items()))
    assert reverse == list(reversed(forward))


def test_dict_get_with_default(dict_factory):
    """Test get() with default value."""
    users = dict_factory("users", {"alice": 1})

    assert users.get("alice") == 1
    assert users.get("bob", 999) == 999


def test_dict_pop(dict_factory):
    """Test pop() operation."""
    users = dict_factory("users", {"alice": 1, "bob": 2})

    value = users.pop("alice")
    assert value == 1
    assert "alice" not in users
    assert len(users) == 1


def test_dict_update(dict_factory):
    """Test update() operation."""
    users = dict_factory("users", {"alice": 1})

    users.update({"bob": 2, "charlie": 3})

    assert users["alice"] == 1
    assert users["bob"] == 2
    assert users["charlie"] == 3
    assert len(users) == 3


def test_dict_clear(dict_factory):
    """Test clear() operation."""
    users = dict_factory("users", {"alice": 1, "bob": 2, "charlie": 3})

    users.clear()

    assert len(users) == 0
    assert list(users.keys()) == []


# ============================================================================
# NESTED DATA
# ============================================================================


def test_dict_nested_dicts(dict_factory):
    """Test nested dictionary structures."""
    data = dict_factory(
        "data",
        {
            "users": {
                "alice": {"name": "Alice", "age": 30},
                "bob": {"name": "Bob", "age": 25},
            }
        },
    )

    extracted = data.extract()
    assert extracted["users"]["alice"]["name"] == "Alice"
    assert extracted["users"]["bob"]["age"] == 25


def test_dict_nested_lists(dict_factory):
    """Test dictionaries containing lists."""
    data = dict_factory(
        "data",
        {
            "alice": {"tags": ["python", "rust"]},
            "bob": {"tags": ["go", "typescript"]},
        },
    )

    extracted = data.extract()
    assert "python" in extracted["alice"]["tags"]
    assert "go" in extracted["bob"]["tags"]


def test_dict_mixed_types(dict_factory):
    """Test dictionary with mixed value types."""
    data = dict_factory(
        "data",
        {
            "string": "hello",
            "number": 42,
            "float": 3.14,
            "bool": True,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
        },
    )

    extracted = data.extract()
    assert extracted["string"] == "hello"
    assert extracted["number"] == 42
    assert extracted["float"] == 3.14
    assert extracted["bool"] is True
    assert extracted["list"] == [1, 2, 3]
    assert extracted["dict"]["nested"] == "value"


# ============================================================================
# STORE AND EXTRACT
# ============================================================================


def test_dict_store_replaces_content(dict_factory):
    """Test store() replaces existing content."""
    users = dict_factory("users", {"alice": 1, "bob": 2})

    users.store({"charlie": 3, "david": 4})

    assert "alice" not in users
    assert "bob" not in users
    assert users["charlie"] == 3
    assert users["david"] == 4


def test_dict_extract_full(dict_factory):
    """Test extract() returns full dictionary."""
    users = dict_factory("users", {"alice": 1, "bob": 2, "charlie": 3})

    extracted = users.extract()

    assert extracted == {"alice": 1, "bob": 2, "charlie": 3}
    assert isinstance(extracted, dict)


# ============================================================================
# CURSOR PAGING
# ============================================================================


def test_dict_keys_after_pages_in_key_order(dict_factory):
    """keys(after=, limit=) walks key order a page at a time, ints before strs."""
    d = dict_factory("d", {"b": 2, "a": 1, "d": {"x": {"y": 1}}, "c": 3, 10: 0, 2: 0})

    assert list(d.keys(limit=3)) == [2, 10, "a"]
    assert list(d.keys(after="a", limit=2)) == ["b", "c"]
    assert list(d.keys(after="c", limit=2)) == ["d"]
    assert list(d.keys(after="d", limit=2)) == []
    assert list(d.keys(after=2)) == [10, "a", "b", "c", "d"]


def test_dict_reversed_keys_pages_newest_key_first(dict_factory):
    d = dict_factory("d", {"a": 1, "b": {"x": 1}, "c": 3, 5: 0})

    assert list(d.reversed_keys()) == ["c", "b", "a", 5]
    assert list(d.reversed_keys(limit=2)) == ["c", "b"]
    assert list(d.reversed_keys(before="b", limit=5)) == ["a", 5]
    assert list(d.reversed_keys(before=5)) == []


def test_dict_cursor_need_not_exist(dict_factory):
    """A cursor is a position in key order: a deleted or never-set key resumes fine."""
    d = dict_factory("d", {"a": 1, "c": 3, "e": 5})

    assert list(d.keys(after="b")) == ["c", "e"]
    assert list(d.reversed_keys(before="d")) == ["c", "a"]
    del d["c"]
    assert list(d.keys(after="c")) == ["e"]


def test_dict_cursor_skips_subtree_of_compound_cursor(dict_factory):
    """The cursor's own subtree sorts after it; paging never yields its fields."""
    d = dict_factory("d", {"a": {"x": 1, "y": {"z": 2}}, "b": 1})

    assert list(d.keys(after="a")) == ["b"]
    assert list(d.reversed_keys(before="b")) == ["a"]


def test_dict_keys_no_args_stays_a_keys_view(dict_factory):
    d = dict_factory("d", {"a": 1, "b": 2})

    keys = d.keys()
    assert len(keys) == 2
    assert "a" in keys
    assert keys | {"z"} == {"a", "b", "z"}


def test_dict_paging_on_empty_and_absent(root_view):
    absent = root_view.open_child("nope", EagerDictView)
    with pytest.raises(ContainerNotFoundError):
        list(absent.keys(after="a"))
    with pytest.raises(ContainerNotFoundError):
        list(absent.reversed_keys(limit=1))

    absent.ensure_created()
    assert list(absent.keys(after="a")) == []
    assert list(absent.keys(limit=5)) == []
    assert list(absent.reversed_keys(before="a")) == []


def test_dict_paging_limit_zero_and_bad_cursor(dict_factory):
    d = dict_factory("d", {"a": 1})

    assert list(d.keys(limit=0)) == []
    with pytest.raises(TypeError):
        d.keys(after=1.5)  # raised at the call, before any read
    with pytest.raises(TypeError):
        d.reversed_keys(before=1.5)
