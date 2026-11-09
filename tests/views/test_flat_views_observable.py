"""The flat and light views watch like the views they stand beside.

``FlatDictView`` and ``LightDictView`` back a reactive dict as ``DictView``
does, and ``FlatListView`` a reactive list as ``ListView`` does, so each
serves every watch: the whole view, one child, the children, a pattern.
The flat dicts page their keys as ``DictView`` does.
"""

import pytest
from tests.support.observed import Hooks, observed_storage

from virtuals.collections import ReactiveMappingBase, ReactiveSequenceBase
from virtuals.views import EagerListView, FlatDictView, FlatListView, LightDictView


@pytest.mark.parametrize("view_cls", [FlatDictView, LightDictView])
def test_flat_dict_views_are_reactive_mappings(view_cls):
    assert issubclass(view_cls, ReactiveMappingBase)


@pytest.mark.parametrize("view_cls", [FlatListView, EagerListView])
def test_list_views_are_reactive_sequences(view_cls):
    assert issubclass(view_cls, ReactiveSequenceBase)


@pytest.mark.parametrize("view_cls", [FlatDictView, LightDictView])
def test_flat_dict_watches_fire(view_cls):
    with observed_storage() as observed:
        h = Hooks(*observed, view_cls=view_cls)
        h.write(lambda v: v.store({"a": 1, "b": 2}))
        h.watch_view(
            change=lambda v: v.on_change(),
            child_b=lambda v: v.on_child_change("b"),
            children=lambda v: v.on_children_change(),
            desc=lambda v: v.on_descendants_change("b"),
        )

        hits = h.run(lambda v: v.__setitem__("b", 20))
        assert len(hits["change"]) == 1
        assert len(hits["child_b"]) == 1
        assert len(hits["children"]) == 1
        assert len(hits["desc"]) == 1

        hits = h.run(lambda v: v.__setitem__("c", 3))
        assert len(hits["change"]) == 1
        assert hits["child_b"] == []
        assert len(hits["children"]) == 1
        assert hits["desc"] == []

        hits = h.run(lambda v: v.__delitem__("b"))
        assert len(hits["child_b"]) == 1
        assert len(hits["children"]) == 1


def test_flat_list_watches_fire():
    with observed_storage() as observed:
        h = Hooks(*observed, view_cls=FlatListView)
        h.write(lambda v: v.store(["a", "b"]))
        h.watch_view(
            change=lambda v: v.on_change(),
            child_1=lambda v: v.on_child_change(1),
            children=lambda v: v.on_children_change(),
            desc=lambda v: v.on_descendants_change(0),
        )

        hits = h.run(lambda v: v.__setitem__(1, "B"))
        assert len(hits["change"]) >= 1
        assert len(hits["child_1"]) == 1
        assert len(hits["children"]) == 1
        assert hits["desc"] == []

        hits = h.run(lambda v: v.append("c"))
        assert hits["child_1"] == []
        assert len(hits["children"]) == 1


def test_flat_list_watch_names_an_index_not_written_yet():
    """A watch is a filter, not a read: an index past the end still names a slot."""
    with observed_storage() as observed:
        h = Hooks(*observed, view_cls=FlatListView)
        h.watch_view(child_0=lambda v: v.on_child_change(0))

        hits = h.run(lambda v: v.append("a"))
        assert len(hits["child_0"]) == 1


@pytest.mark.parametrize("factory", ["flat_dict_factory", "light_dict_factory"])
def test_flat_dict_views_page_like_dict_view(factory, request):
    d = request.getfixturevalue(factory)("d", {"b": 2, "a": 1, "c": 3, 5: 0})

    assert list(d.keys(limit=2)) == [5, "a"]
    assert list(d.keys(after="a", limit=5)) == ["b", "c"]
    assert list(d.keys(after="bb")) == ["c"]  # a cursor need not exist
    assert list(d.reversed_keys()) == ["c", "b", "a", 5]
    assert list(d.reversed_keys(before="b", limit=1)) == ["a"]
    assert set(d.keys()) == {5, "a", "b", "c"}
    with pytest.raises(TypeError):
        d.keys(after=1.5)
