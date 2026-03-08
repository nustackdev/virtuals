"""Tests for eager/lazy facet pattern.

Validates the core design principle: every collection view has two symmetric
facets — eager (returns Python values) and lazy (returns child Views).
"""

from __future__ import annotations

from itertools import islice

from virtuals._views import EagerDictView, EagerListView, LazyDictView, LazyListView
from virtuals.view import ViewBase


# =============================================================================
# DICT VIEW — EAGER/LAZY FACETS
# =============================================================================


class TestEagerDictView:
    """EagerDictView returns extracted Python values."""

    def test_getitem_returns_value(self, root_view: EagerDictView) -> None:
        root_view["name"] = "Alice"
        assert root_view["name"] == "Alice"

    def test_getitem_nested_returns_dict(self, root_view: EagerDictView) -> None:
        root_view["user"] = {"name": "Alice", "age": 30}
        result = root_view["user"]
        assert isinstance(result, dict)
        assert result == {"name": "Alice", "age": 30}

    def test_values_yields_python_objects(self, root_view: EagerDictView) -> None:
        root_view["a"] = 1
        root_view["b"] = 2
        assert list(root_view.values()) == [1, 2]

    def test_items_yields_python_pairs(self, root_view: EagerDictView) -> None:
        root_view["x"] = 10
        root_view["y"] = 20
        assert list(root_view.items()) == [("x", 10), ("y", 20)]

    def test_eager_property_is_identity(self, root_view: EagerDictView) -> None:
        assert root_view.eager is root_view

    def test_lazy_property_returns_lazy_facet(self, root_view: EagerDictView) -> None:
        lazy = root_view.lazy
        assert isinstance(lazy, LazyDictView)


class TestLazyDictView:
    """LazyDictView returns child Views for containers, values for primitives."""

    def test_getitem_primitive_returns_value(self, root_view: EagerDictView) -> None:
        root_view["name"] = "Alice"
        lazy = root_view.lazy
        assert lazy["name"] == "Alice"

    def test_getitem_container_returns_view(self, root_view: EagerDictView) -> None:
        root_view["user"] = {"name": "Alice", "age": 30}
        lazy = root_view.lazy
        child = lazy["user"]
        assert isinstance(child, ViewBase)

    def test_child_view_is_eager_by_default(self, root_view: EagerDictView) -> None:
        """Lazy is not a mode — child views default to eager."""
        root_view["user"] = {"name": "Alice", "age": 30}
        lazy = root_view.lazy
        child = lazy["user"]
        # Child view supports eager access — extract returns dict
        assert child.extract() == {"name": "Alice", "age": 30}

    def test_values_yields_views_for_containers(self, root_view: EagerDictView) -> None:
        root_view["alice"] = {"name": "Alice"}
        root_view["bob"] = {"name": "Bob"}
        lazy = root_view.lazy
        values = list(lazy.values())
        assert len(values) == 2
        assert all(isinstance(v, ViewBase) for v in values)

    def test_values_yields_primitives_for_leaves(self, root_view: EagerDictView) -> None:
        root_view["x"] = 42
        root_view["y"] = "hello"
        lazy = root_view.lazy
        values = list(lazy.values())
        assert values == [42, "hello"]

    def test_values_mixed_containers_and_primitives(self, root_view: EagerDictView) -> None:
        root_view["name"] = "Alice"
        root_view["profile"] = {"role": "admin"}
        lazy = root_view.lazy
        values = list(lazy.values())
        assert values[0] == "Alice"
        assert isinstance(values[1], ViewBase)

    def test_items_yields_key_view_pairs(self, root_view: EagerDictView) -> None:
        root_view["user"] = {"name": "Alice"}
        lazy = root_view.lazy
        items = list(lazy.items())
        assert len(items) == 1
        key, view = items[0]
        assert key == "user"
        assert isinstance(view, ViewBase)

    def test_keys_same_as_eager(self, root_view: EagerDictView) -> None:
        root_view["a"] = 1
        root_view["b"] = 2
        eager_keys = list(root_view.keys())
        lazy_keys = list(root_view.lazy.keys())
        assert eager_keys == lazy_keys

    def test_len_same_as_eager(self, root_view: EagerDictView) -> None:
        root_view["a"] = 1
        root_view["b"] = 2
        assert len(root_view.lazy) == len(root_view)

    def test_contains_same_as_eager(self, root_view: EagerDictView) -> None:
        root_view["exists"] = 42
        assert "exists" in root_view.lazy
        assert "missing" not in root_view.lazy

    def test_eager_property_returns_eager_facet(self, root_view: EagerDictView) -> None:
        lazy = root_view.lazy
        eager = lazy.eager
        assert isinstance(eager, EagerDictView)

    def test_lazy_property_is_identity(self, root_view: EagerDictView) -> None:
        lazy = root_view.lazy
        assert lazy.lazy is lazy


class TestDictFacetCrossNavigation:
    """Verify cross-navigation between eager and lazy facets."""

    def test_eager_to_lazy_to_eager_roundtrip(self, root_view: EagerDictView) -> None:
        root_view["key"] = "value"
        lazy = root_view.lazy
        eager_again = lazy.eager
        assert eager_again["key"] == "value"

    def test_lazy_mutation_visible_from_eager(self, root_view: EagerDictView) -> None:
        """Mutations through lazy facet are visible from eager."""
        lazy = root_view.lazy
        lazy["new_key"] = "new_value"
        assert root_view["new_key"] == "new_value"

    def test_eager_mutation_visible_from_lazy(self, root_view: EagerDictView) -> None:
        """Mutations through eager facet are visible from lazy."""
        root_view["key"] = 42
        assert root_view.lazy["key"] == 42


class TestDictLazyComposition:
    """Verify lazy views compose with Python ecosystem tools."""

    def test_islice_on_lazy_values(self, root_view: EagerDictView) -> None:
        """islice(dictview.lazy.values(), n) works naturally."""
        root_view.store({"a": {"x": 1}, "b": {"x": 2}, "c": {"x": 3}, "d": {"x": 4}})
        first_2 = list(islice(root_view.lazy.values(), 2))
        assert len(first_2) == 2
        assert all(isinstance(v, ViewBase) for v in first_2)

    def test_islice_then_extract(self, root_view: EagerDictView) -> None:
        """Navigate lazily, materialize selectively."""
        root_view.store({"a": {"val": 1}, "b": {"val": 2}, "c": {"val": 3}})
        first = next(iter(root_view.lazy.values()))
        assert isinstance(first, ViewBase)
        assert first.extract() == {"val": 1}

    def test_lazy_not_a_mode(self, root_view: EagerDictView) -> None:
        """Each navigation step independently chooses eager or lazy."""
        root_view["users"] = {"alice": {"name": "Alice"}}
        # Navigate lazily to get the users child view
        users_view = root_view.lazy["users"]
        assert isinstance(users_view, ViewBase)
        # The child view is eager by default — alice returns extracted dict
        alice_data = users_view["alice"]
        assert alice_data == {"name": "Alice"}
        # But you can also go lazy on the child
        alice_view = users_view.lazy["alice"]
        assert isinstance(alice_view, ViewBase)

    def test_list_over_lazy_values(self, root_view: EagerDictView) -> None:
        """list() works on lazy values."""
        root_view["a"] = 1
        root_view["b"] = 2
        values = list(root_view.lazy.values())
        assert values == [1, 2]  # primitives come through as values


# =============================================================================
# LIST VIEW — EAGER/LAZY FACETS
# =============================================================================


class TestEagerListView:
    """EagerListView returns extracted Python values."""

    def test_getitem_returns_value(self, root_view: EagerDictView) -> None:
        lst = root_view.open_child("list", EagerListView)
        lst.store([10, 20, 30])
        assert lst[0] == 10
        assert lst[2] == 30

    def test_iter_yields_values(self, root_view: EagerDictView) -> None:
        lst = root_view.open_child("list", EagerListView)
        lst.store([1, 2, 3])
        assert list(lst) == [1, 2, 3]

    def test_lazy_property(self, root_view: EagerDictView) -> None:
        lst = root_view.open_child("list", EagerListView)
        lazy = lst.lazy
        assert isinstance(lazy, LazyListView)

    def test_eager_is_identity(self, root_view: EagerDictView) -> None:
        lst = root_view.open_child("list", EagerListView)
        assert lst.eager is lst


class TestLazyListView:
    """LazyListView returns child Views for containers, values for primitives."""

    def test_getitem_primitive_returns_value(self, root_view: EagerDictView) -> None:
        lst = root_view.open_child("list", EagerListView)
        lst.store([10, 20, 30])
        lazy = lst.lazy
        assert lazy[0] == 10

    def test_iter_primitive_yields_values(self, root_view: EagerDictView) -> None:
        lst = root_view.open_child("list", EagerListView)
        lst.store([1, 2, 3])
        assert list(lst.lazy) == [1, 2, 3]

    def test_getitem_container_returns_view(self, root_view: EagerDictView) -> None:
        lst = root_view.open_child("list", EagerListView)
        lst.store([{"name": "Alice"}, {"name": "Bob"}])
        lazy = lst.lazy
        child = lazy[0]
        assert isinstance(child, ViewBase)
        assert child.extract() == {"name": "Alice"}

    def test_iter_container_yields_views(self, root_view: EagerDictView) -> None:
        lst = root_view.open_child("list", EagerListView)
        lst.store([{"a": 1}, {"b": 2}])
        lazy = lst.lazy
        items = list(lazy)
        assert len(items) == 2
        assert all(isinstance(v, ViewBase) for v in items)

    def test_islice_on_lazy_list(self, root_view: EagerDictView) -> None:
        lst = root_view.open_child("list", EagerListView)
        lst.store([{"x": i} for i in range(5)])
        first_3 = list(islice(lst.lazy, 3))
        assert len(first_3) == 3
        assert all(isinstance(v, ViewBase) for v in first_3)

    def test_lazy_list_len(self, root_view: EagerDictView) -> None:
        lst = root_view.open_child("list", EagerListView)
        lst.store([1, 2, 3])
        assert len(lst.lazy) == 3

    def test_eager_roundtrip(self, root_view: EagerDictView) -> None:
        lst = root_view.open_child("list", EagerListView)
        lst.store([1, 2, 3])
        eager_again = lst.lazy.eager
        assert isinstance(eager_again, EagerListView)
        assert list(eager_again) == [1, 2, 3]
