"""End-to-end tests for FlatDictView."""

import pytest


def test_flat_dict_factory_creates_empty_view(flat_dict_factory):
    scores = flat_dict_factory("scores")
    assert scores is not None
    assert len(scores) == 0


def test_flat_dict_factory_with_data(flat_dict_factory):
    scores = flat_dict_factory("scores", {"alice": 100, "bob": 95})
    data = scores.extract()
    assert data == {"alice": 100, "bob": 95}


# ============================================================================
# BASIC OPERATIONS
# ============================================================================


def test_flat_dict_set_and_get(flat_dict_factory):
    scores = flat_dict_factory("scores")
    scores["alice"] = 100
    scores["bob"] = 95
    assert scores["alice"] == 100
    assert scores["bob"] == 95


def test_flat_dict_get_missing_raises(flat_dict_factory):
    scores = flat_dict_factory("scores")
    with pytest.raises(KeyError):
        scores["missing"]


def test_flat_dict_delete(flat_dict_factory):
    scores = flat_dict_factory("scores", {"alice": 100, "bob": 95, "charlie": 80})
    del scores["bob"]
    assert "bob" not in scores
    assert "alice" in scores
    assert "charlie" in scores


def test_flat_dict_contains(flat_dict_factory):
    scores = flat_dict_factory("scores", {"alice": 100, "bob": 95})
    assert "alice" in scores
    assert "charlie" not in scores


def test_flat_dict_len(flat_dict_factory):
    scores = flat_dict_factory("scores")
    assert len(scores) == 0

    scores["alice"] = 100
    assert len(scores) == 1

    scores["bob"] = 95
    assert len(scores) == 2

    del scores["alice"]
    # After delete, length is updated
    assert len(scores) == 1


def test_flat_dict_iteration(flat_dict_factory):
    scores = flat_dict_factory("scores", {"a": 1, "b": 2, "c": 3})
    keys = set(scores)
    assert keys == {"a", "b", "c"}


# ============================================================================
# DICT METHODS
# ============================================================================


def test_flat_dict_keys(flat_dict_factory):
    scores = flat_dict_factory("scores", {"alice": 100, "bob": 95})
    keys = set(scores.keys())
    assert keys == {"alice", "bob"}


def test_flat_dict_values(flat_dict_factory):
    scores = flat_dict_factory("scores", {"a": 1, "b": 2, "c": 3})
    values = set(scores.values())
    assert values == {1, 2, 3}


def test_flat_dict_items(flat_dict_factory):
    scores = flat_dict_factory("scores", {"a": 1, "b": 2})
    items = dict(scores.items())
    assert items == {"a": 1, "b": 2}


def test_flat_dict_get_with_default(flat_dict_factory):
    scores = flat_dict_factory("scores", {"alice": 100})
    assert scores.get("alice") == 100
    assert scores.get("bob", 0) == 0


def test_flat_dict_pop(flat_dict_factory):
    scores = flat_dict_factory("scores", {"alice": 100, "bob": 95})
    value = scores.pop("alice")
    assert value == 100
    assert "alice" not in scores


def test_flat_dict_pop_missing_with_default(flat_dict_factory):
    scores = flat_dict_factory("scores")
    value = scores.pop("missing", -1)
    assert value == -1


def test_flat_dict_pop_missing_raises(flat_dict_factory):
    scores = flat_dict_factory("scores")
    with pytest.raises(KeyError):
        scores.pop("missing")


def test_flat_dict_update(flat_dict_factory):
    scores = flat_dict_factory("scores", {"alice": 100})
    scores.update({"bob": 95, "charlie": 80})
    assert scores["bob"] == 95
    assert scores["charlie"] == 80
    assert scores["alice"] == 100


def test_flat_dict_clear(flat_dict_factory):
    scores = flat_dict_factory("scores", {"a": 1, "b": 2, "c": 3})
    scores.clear()
    assert len(scores) == 0
    assert scores.extract() == {}


# ============================================================================
# STORE AND EXTRACT
# ============================================================================


def test_flat_dict_store_replaces_content(flat_dict_factory):
    scores = flat_dict_factory("scores", {"alice": 100})
    scores.store({"bob": 95, "charlie": 80})
    assert "alice" not in scores
    assert scores.extract() == {"bob": 95, "charlie": 80}


def test_flat_dict_extract_returns_dict(flat_dict_factory):
    scores = flat_dict_factory("scores", {"a": 1, "b": 2})
    extracted = scores.extract()
    assert isinstance(extracted, dict)
    assert extracted == {"a": 1, "b": 2}


# ============================================================================
# MIXED PRIMITIVE TYPES
# ============================================================================


def test_flat_dict_mixed_primitives(flat_dict_factory):
    data = flat_dict_factory(
        "data",
        {
            "string": "hello",
            "number": 42,
            "float": 3.14,
            "bool": True,
            "none": None,
        },
    )
    extracted = data.extract()
    assert extracted["string"] == "hello"
    assert extracted["number"] == 42
    assert extracted["float"] == 3.14
    assert extracted["bool"] is True
    assert extracted["none"] is None


def test_flat_dict_overwrite_value(flat_dict_factory):
    scores = flat_dict_factory("scores", {"alice": 100})
    scores["alice"] = 200
    assert scores["alice"] == 200
