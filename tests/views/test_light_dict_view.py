"""End-to-end tests for LightDictView."""

import pytest


def test_light_dict_factory_creates_empty_view(light_dict_factory):
    cache = light_dict_factory("cache")
    assert cache is not None
    assert len(cache) == 0


def test_light_dict_factory_with_data(light_dict_factory):
    cache = light_dict_factory("cache", {"key": "value"})
    assert cache.extract() == {"key": "value"}


# ============================================================================
# BASIC OPERATIONS
# ============================================================================


def test_light_dict_set_and_get(light_dict_factory):
    cache = light_dict_factory("cache")
    cache["name"] = "Alice"
    cache["age"] = 30
    assert cache["name"] == "Alice"
    assert cache["age"] == 30


def test_light_dict_get_missing_raises(light_dict_factory):
    cache = light_dict_factory("cache")
    with pytest.raises(KeyError):
        cache["missing"]


def test_light_dict_delete(light_dict_factory):
    cache = light_dict_factory("cache", {"a": 1, "b": 2, "c": 3})
    del cache["b"]
    assert "b" not in cache
    assert "a" in cache
    assert "c" in cache


def test_light_dict_contains(light_dict_factory):
    cache = light_dict_factory("cache", {"a": 1, "b": 2})
    assert "a" in cache
    assert "c" not in cache


def test_light_dict_len(light_dict_factory):
    """LightDictView computes len via iteration (no metadata tracking)."""
    cache = light_dict_factory("cache")
    assert len(cache) == 0

    cache["a"] = 1
    assert len(cache) == 1

    cache["b"] = 2
    assert len(cache) == 2

    del cache["a"]
    assert len(cache) == 1


def test_light_dict_iteration(light_dict_factory):
    cache = light_dict_factory("cache", {"a": 1, "b": 2, "c": 3})
    keys = set(cache)
    assert keys == {"a", "b", "c"}


# ============================================================================
# DICT METHODS
# ============================================================================


def test_light_dict_keys(light_dict_factory):
    cache = light_dict_factory("cache", {"x": 10, "y": 20})
    keys = set(cache.keys())
    assert keys == {"x", "y"}


def test_light_dict_values(light_dict_factory):
    cache = light_dict_factory("cache", {"a": 1, "b": 2})
    values = set(cache.values())
    assert values == {1, 2}


def test_light_dict_items(light_dict_factory):
    cache = light_dict_factory("cache", {"a": 1, "b": 2})
    items = dict(cache.items())
    assert items == {"a": 1, "b": 2}


def test_light_dict_get_with_default(light_dict_factory):
    cache = light_dict_factory("cache", {"a": 1})
    assert cache.get("a") == 1
    assert cache.get("missing", 42) == 42


def test_light_dict_update(light_dict_factory):
    cache = light_dict_factory("cache", {"a": 1})
    cache.update({"b": 2, "c": 3})
    assert cache["a"] == 1
    assert cache["b"] == 2
    assert cache["c"] == 3


def test_light_dict_clear(light_dict_factory):
    cache = light_dict_factory("cache", {"a": 1, "b": 2})
    cache.clear()
    assert len(cache) == 0
    assert cache.extract() == {}


# ============================================================================
# STORE AND EXTRACT
# ============================================================================


def test_light_dict_store_replaces_content(light_dict_factory):
    cache = light_dict_factory("cache", {"a": 1})
    cache.store({"b": 2, "c": 3})
    assert "a" not in cache
    assert cache.extract() == {"b": 2, "c": 3}


def test_light_dict_store_no_replace(light_dict_factory):
    cache = light_dict_factory("cache", {"a": 1})
    cache.store({"b": 2}, replace=False)
    extracted = cache.extract()
    assert "a" in extracted
    assert "b" in extracted


def test_light_dict_extract_returns_dict(light_dict_factory):
    cache = light_dict_factory("cache", {"x": 10, "y": 20})
    extracted = cache.extract()
    assert isinstance(extracted, dict)
    assert extracted == {"x": 10, "y": 20}


# ============================================================================
# MIXED PRIMITIVE TYPES
# ============================================================================


def test_light_dict_mixed_primitives(light_dict_factory):
    cache = light_dict_factory(
        "cache",
        {
            "string": "hello",
            "number": 42,
            "float": 3.14,
            "bool": True,
            "none": None,
        },
    )
    extracted = cache.extract()
    assert extracted["string"] == "hello"
    assert extracted["number"] == 42
    assert extracted["float"] == 3.14
    assert extracted["bool"] is True
    assert extracted["none"] is None


def test_light_dict_overwrite_value(light_dict_factory):
    cache = light_dict_factory("cache", {"key": "old"})
    cache["key"] = "new"
    assert cache["key"] == "new"
