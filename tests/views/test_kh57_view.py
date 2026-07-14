"""End-to-end tests for Kh57View (Eager and Lazy facets)."""

import random

import pytest

from virtuals._views import EagerDictView, EagerKh57View, LazyKh57View


def _rng(seed: int) -> random.Random:
    """Test-only helper — seeded rng for deterministic sampling checks."""
    return random.Random(seed)  # noqa: S311


# ============================================================================
# BASIC OPERATIONS
# ============================================================================


def test_kh57_factory_creates_empty_view(kh57_factory):
    view = kh57_factory("v")
    assert view is not None
    assert len(view) == 0
    assert list(view) == []


def test_kh57_set_and_get(kh57_factory):
    view = kh57_factory("v")
    view[42] = "hello"
    view[100] = 3.14
    assert view[42] == "hello"
    assert view[100] == 3.14


def test_kh57_nested_set_and_get(kh57_factory):
    view = kh57_factory("v")
    view[7] = {"name": "Alice", "age": 30}
    result = view[7]
    assert isinstance(result, dict)
    assert result == {"name": "Alice", "age": 30}


def test_kh57_delete(kh57_factory):
    view = kh57_factory("v", {1: "a", 2: "b", 3: "c"})
    del view[2]
    assert 2 not in view
    assert 1 in view
    assert 3 in view
    assert len(view) == 2


def test_kh57_delete_missing_raises(kh57_factory):
    view = kh57_factory("v", {1: "a"})
    with pytest.raises(KeyError):
        del view[42]


def test_kh57_get_missing_raises(kh57_factory):
    view = kh57_factory("v", {1: "a"})
    with pytest.raises(KeyError):
        _ = view[42]


def test_kh57_get_with_default(kh57_factory):
    view = kh57_factory("v", {1: "a"})
    assert view.get(42, default="miss") == "miss"


def test_kh57_contains(kh57_factory):
    view = kh57_factory("v", {1: "a", 5: "b"})
    assert 1 in view
    assert 5 in view
    assert 99 not in view
    assert "abc" not in view


def test_kh57_len_tracks_puts_and_deletes(kh57_factory):
    view = kh57_factory("v")
    assert len(view) == 0
    view[10] = "x"
    view[20] = "y"
    assert len(view) == 2
    # overwrite same key doesn't grow
    view[10] = "x2"
    assert len(view) == 2
    del view[10]
    assert len(view) == 1


def test_kh57_bad_key_type_raises(kh57_factory):
    view = kh57_factory("v")
    with pytest.raises(TypeError):
        view["bad"] = 1


# ============================================================================
# ITERATION IN ORIGINAL INT ORDER
# ============================================================================


def test_kh57_iter_original_key_order(kh57_factory):
    view = kh57_factory("v")
    for k in [999, 42, 7, 8000, 100]:
        view[k] = f"val_{k}"
    assert list(view) == [7, 42, 100, 999, 8000]


def test_kh57_iter_after_delete(kh57_factory):
    view = kh57_factory("v", {i: str(i) for i in range(20)})
    del view[5]
    del view[10]
    keys = list(view)
    assert 5 not in keys
    assert 10 not in keys
    assert keys == sorted(keys)
    assert len(keys) == 18


def test_kh57_iter_empty(kh57_factory):
    view = kh57_factory("v")
    assert list(view) == []


# ============================================================================
# RANGE
# ============================================================================


def test_kh57_range_basic(kh57_factory):
    view = kh57_factory("v", {i: str(i) for i in range(50)})
    picks = list(view.range(10, 20))
    assert picks == [(i, str(i)) for i in range(10, 20)]


def test_kh57_range_empty_slice(kh57_factory):
    view = kh57_factory("v", {i: i for i in range(10)})
    assert list(view.range(100, 200)) == []


def test_kh57_range_invalid_bounds(kh57_factory):
    view = kh57_factory("v")
    with pytest.raises(ValueError):
        list(view.range(-1, 10))


# ============================================================================
# SAMPLE
# ============================================================================


def test_kh57_sample_returns_distinct_items(kh57_factory):
    view = kh57_factory("v", {i: str(i) for i in range(200)})
    picks = view.sample(10, rng=_rng(0))
    assert len(picks) == 10
    keys = [k for k, _v in picks]
    assert len(set(keys)) == 10
    for k, v in picks:
        assert view[k] == v


def test_kh57_sample_deterministic_with_seeded_rng(kh57_factory):
    view = kh57_factory("v", {i: str(i) for i in range(500)})
    a = view.sample(20, rng=_rng(42))
    b = view.sample(20, rng=_rng(42))
    assert a == b


def test_kh57_sample_empty(kh57_factory):
    view = kh57_factory("v")
    assert view.sample(5) == []


def test_kh57_sample_range(kh57_factory):
    view = kh57_factory("v", {i: i for i in range(1000)})
    picks = view.sample(30, begin=100, end=200, rng=_rng(1))
    for k, _v in picks:
        assert 100 <= k < 200


def test_kh57_sample_stability_under_outer_appends(kh57_factory):
    """Appending outside the queried range must not change the sample."""
    view = kh57_factory("v", {i: str(i) for i in range(1000)})
    before = sorted(view.sample(30, begin=100, end=200, rng=_rng(7)))

    for i in range(5000, 5100):
        view[i] = str(i)

    after = sorted(view.sample(30, begin=100, end=200, rng=_rng(7)))
    assert before == after


# ============================================================================
# EAGER vs LAZY FACETS
# ============================================================================


def test_kh57_eager_returns_extracted_values(kh57_factory):
    view = kh57_factory("v")
    view[1] = {"x": 1}
    got = view[1]
    assert isinstance(got, dict)
    assert got == {"x": 1}


def test_kh57_lazy_returns_child_views(kh57_factory):
    view = kh57_factory("v")
    view[1] = {"x": 1}
    lazy = view.lazy
    got = lazy[1]
    # lazy returns a View for container children
    assert not isinstance(got, dict)
    assert isinstance(got, EagerDictView)
    assert got.extract() == {"x": 1}


def test_kh57_facet_navigation(kh57_factory):
    view = kh57_factory("v", {1: "a"})
    assert isinstance(view.lazy, LazyKh57View)
    assert isinstance(view.lazy.eager, EagerKh57View)


# ============================================================================
# EXTRACT
# ============================================================================


def test_kh57_extract(kh57_factory):
    data = {999: "a", 42: "b", 7: "c"}
    view = kh57_factory("v", data)
    extracted = view.extract()
    assert extracted == data


def test_kh57_clear(kh57_factory):
    view = kh57_factory("v", {i: str(i) for i in range(10)})
    view.clear()
    assert len(view) == 0
    assert list(view) == []


# ============================================================================
# STORE / UPDATE / POP
# ============================================================================


def test_kh57_store_bulk_loads(kh57_factory):
    view = kh57_factory("v")
    view.store({1: "a", 42: "b", 999: "c"})
    assert len(view) == 3
    assert view[42] == "b"
    assert list(view) == [1, 42, 999]


def test_kh57_store_replaces_by_default(kh57_factory):
    view = kh57_factory("v", {1: "old", 2: "old"})
    view.store({10: "new", 20: "new"})
    assert len(view) == 2
    assert 1 not in view
    assert view[10] == "new"


def test_kh57_store_merges_when_replace_false(kh57_factory):
    view = kh57_factory("v", {1: "keep"})
    view.store({2: "add"}, replace=False)
    assert len(view) == 2
    assert view[1] == "keep"
    assert view[2] == "add"


def test_kh57_store_rejects_non_int_key(kh57_factory):
    view = kh57_factory("v")
    with pytest.raises(TypeError):
        view.store({"bad": 1})


def test_kh57_update_from_mapping(kh57_factory):
    view = kh57_factory("v", {1: "a"})
    view.update({2: "b", 3: "c"})
    assert view.extract() == {1: "a", 2: "b", 3: "c"}


def test_kh57_update_overwrites(kh57_factory):
    view = kh57_factory("v", {1: "old"})
    view.update({1: "new"})
    assert view[1] == "new"
    assert len(view) == 1


def test_kh57_pop_returns_and_deletes(kh57_factory):
    view = kh57_factory("v", {1: "a", 2: "b"})
    got = view.pop(1)
    assert got == "a"
    assert 1 not in view
    assert len(view) == 1


def test_kh57_pop_missing_raises(kh57_factory):
    view = kh57_factory("v", {1: "a"})
    with pytest.raises(KeyError):
        view.pop(42)


def test_kh57_pop_missing_with_default(kh57_factory):
    view = kh57_factory("v", {1: "a"})
    assert view.pop(42, default="miss") == "miss"
    assert len(view) == 1
