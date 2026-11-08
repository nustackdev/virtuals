"""A parent reads back every built-in child view by structure, whichever views write a type.

The registry's views decide which view writes a ``dict`` or a ``list``. A
child stored with another view (an ordered dict, a log dict, a ring) is
still read by its structure when its parent is extracted or iterated.
"""

from __future__ import annotations

import pytest

from virtuals._views import (
    EagerDictView,
    EagerFractionalIndexedDictView,
    EagerLogIndexedDictView,
    RingView,
)
from virtuals.view.exceptions import ViewRegistryError
from virtuals.view.registry import ViewRegistry


def test_a_row_holding_ordered_log_and_ring_children_extracts_whole(root_view):
    row = root_view.open_child("row", EagerDictView)
    row.ensure_created()
    row["name"] = "board"
    row.open_child("cells", EagerFractionalIndexedDictView).store({"c1": True, "c2": True})
    row.open_child("runs", EagerLogIndexedDictView).store({"r1": True})
    row.open_child("out", RingView.of(2)).store([["t", "stdout", "a"], ["t", "stdout", "b"]])
    assert root_view.extract() == {
        "row": {
            "name": "board",
            "cells": {"c1": True, "c2": True},
            "runs": {"r1": True},
            "out": [["t", "stdout", "a"], ["t", "stdout", "b"]],
        }
    }


def test_a_reader_claims_its_structure_but_never_a_type():
    registry = ViewRegistry()
    registry.register(EagerDictView)
    registry.register_reader(EagerFractionalIndexedDictView)
    structure = EagerFractionalIndexedDictView.get_structure()
    assert registry.get_view_for_structure(structure) is EagerFractionalIndexedDictView
    assert registry.get_view_for_type(dict) is EagerDictView
    with pytest.raises(ViewRegistryError, match="already registered"):
        registry.register_reader(EagerFractionalIndexedDictView)


def test_every_built_in_view_reads_back_through_any_navigator():
    from virtuals import _views
    from virtuals.codecs import NoOpCodec
    from virtuals.navigator import Navigator
    from virtuals.storages.mem import InMemoryStorage

    exported = {
        getattr(_views, name).STRUCTURE
        for name in _views.__all__
        if getattr(getattr(_views, name), "STRUCTURE", None) is not None
    }
    assert {view.STRUCTURE for view in _views.BUILTIN_VIEWS} == exported
    registry = Navigator(InMemoryStorage(codec=NoOpCodec()), views=(EagerDictView,)).registry
    for view in _views.BUILTIN_VIEWS:
        assert registry.is_structure_registered(view.STRUCTURE), view
