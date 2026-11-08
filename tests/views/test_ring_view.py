"""Tests for RingView: a capped log, oldest entries drop."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from virtuals._views import EagerDictView, RingView
from virtuals.container import ContainerNotFoundError
from virtuals.navigator import Navigator


@pytest.fixture
def ring_factory(root_view):
    def _create(address="out", capacity=3, data=None):
        view = root_view.open_child(address, RingView.of(capacity))
        if data is not None:
            view.store(data)
        else:
            view.ensure_created()
        return view

    return _create


# ============================================================================
# CAPACITY
# ============================================================================


def test_of_is_one_class_per_capacity():
    assert RingView.of(3) is RingView.of(3)
    assert RingView.of(3) is not RingView.of(4)
    assert RingView.of(3).CAPACITY == 3
    assert RingView.of(3).STRUCTURE == RingView.STRUCTURE
    assert issubclass(RingView.of(3), Sequence)


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "3"])
def test_of_refuses_a_bad_capacity(bad):
    with pytest.raises(ValueError, match="capacity"):
        RingView.of(bad)


def test_ring_without_capacity_reads_but_refuses_writes(ring_factory, root_view):
    ring_factory(data=["a", "b"])
    plain = root_view.open_child("out", RingView)
    assert list(plain) == ["a", "b"]
    with pytest.raises(TypeError, match="read-only"):
        plain.append("c")
    with pytest.raises(TypeError, match="read-only"):
        plain.store(["c"])
    assert list(plain) == ["a", "b"]


# ============================================================================
# APPEND, DROP, READ
# ============================================================================


def test_append_drops_oldest_past_capacity(ring_factory):
    ring = ring_factory(capacity=3)
    seqs = [ring.append(x) for x in "abcde"]

    assert seqs == [0, 1, 2, 3, 4]
    assert list(ring) == ["c", "d", "e"]
    assert list(reversed(ring)) == ["e", "d", "c"]
    assert len(ring) == 3
    assert ring.extract() == ["c", "d", "e"]


def test_compound_entries_are_stored_whole(ring_factory):
    ring = ring_factory(capacity=2)
    ring.append([1.5, "stdout", "hello"])
    ring.append({"ts": 2.0, "text": "bye"})

    assert list(ring) == [[1.5, "stdout", "hello"], {"ts": 2.0, "text": "bye"}]
    rows = list(ring.container.iter_children())
    assert [key for key, _ in rows] == [0, 1]


def test_read_by_position(ring_factory):
    ring = ring_factory(capacity=3, data=["a", "b", "c", "d"])

    assert (ring[0], ring[1], ring[2]) == ("b", "c", "d")
    assert (ring[-1], ring[-3]) == ("d", "b")
    for bad in (3, -4):
        with pytest.raises(IndexError):
            ring[bad]
    with pytest.raises(TypeError):
        ring["0"]


def test_sequence_lookups(ring_factory):
    ring = ring_factory(capacity=4, data=["a", "b", "a", "c", "a"])  # the first "a" dropped

    assert isinstance(ring, Sequence)
    assert "c" in ring
    assert "z" not in ring
    assert ring.index("a") == 1
    assert ring.index("a", 2) == 3
    assert ring.count("a") == 2
    with pytest.raises(ValueError):
        ring.index("a", 0, 1)


def test_empty_ring(ring_factory):
    ring = ring_factory()
    assert len(ring) == 0
    assert not ring
    assert list(ring) == []
    assert list(ring.entries()) == []
    with pytest.raises(IndexError):
        ring[0]
    with pytest.raises(IndexError):
        ring[-1]


def test_never_written_ring_raises_on_read(root_view):
    ring = root_view.open_child("nope", RingView.of(3))
    with pytest.raises(ContainerNotFoundError):
        list(ring)
    ring.clear()  # clear of nothing writes nothing
    assert not ring.container.exists()


def test_extend_oversized_batch_keeps_the_last(ring_factory):
    ring = ring_factory(capacity=3, data=["a"])
    ring.extend(["b", "c", "d", "e", "f"])

    assert list(ring) == ["d", "e", "f"]
    assert [seq for seq, _ in ring.entries()] == [3, 4, 5]
    assert len(ring) == 3
    ring.extend([])
    assert list(ring) == ["d", "e", "f"]


def test_store_replaces_and_numbers_on(ring_factory):
    ring = ring_factory(capacity=3, data=["a", "b", "c", "d"])
    ring.store(["x", "y"])

    assert list(ring.entries()) == [(4, "x"), (5, "y")]
    assert list(ring.entries(after=3)) == [(4, "x"), (5, "y")]


def test_clear_then_append_numbers_on(ring_factory):
    ring = ring_factory(capacity=3, data=["a", "b"])
    ring.clear()
    assert len(ring) == 0
    ring.clear()  # an empty ring keeps its mark
    assert ring.append("c") == 2
    assert list(ring.entries(after=1)) == [(2, "c")]


def test_store_of_nothing_then_append_numbers_on(ring_factory):
    ring = ring_factory(capacity=3, data=["a", "b"])
    ring.store([])
    assert list(ring) == []
    ring.store(["c"])
    assert list(ring.entries()) == [(2, "c")]


def test_a_deleted_ring_written_again_starts_at_zero(root_view):
    parent = root_view.open_child("runs", EagerDictView)
    parent.ensure_created()
    parent.set_child_container_as("out", ["a", "b"], RingView.of(3))
    parent.open_child("out", RingView.of(3)).clear()
    del parent["out"]
    parent.set_child_container_as("out", ["x"], RingView.of(3))
    assert list(parent.open_child("out", RingView).entries()) == [(0, "x")]


def test_smaller_capacity_drops_extra_on_next_append(ring_factory, root_view):
    ring_factory(capacity=5, data=list("abcde"))
    smaller = root_view.open_child("out", RingView.of(2))
    assert list(smaller) == list("abcde")
    smaller.append("f")
    assert list(smaller) == ["e", "f"]
    assert len(smaller) == 2


def test_empty_entry_is_refused_before_any_write(ring_factory):
    from virtuals.types import EMPTY

    ring = ring_factory(data=["a"])
    with pytest.raises(ValueError):
        ring.extend(["b", EMPTY])
    assert list(ring) == ["a"]


# ============================================================================
# CURSOR
# ============================================================================


def test_entries_follow_by_sequence_number(ring_factory):
    ring = ring_factory(capacity=3, data=["a", "b"])
    seen = list(ring.entries())
    assert seen == [(0, "a"), (1, "b")]

    ring.extend(["c", "d"])
    assert list(ring.entries(after=seen[-1][0])) == [(2, "c"), (3, "d")]
    assert list(ring.entries(after=1, limit=1)) == [(2, "c")]


def test_reader_that_fell_behind_resumes_at_oldest_kept(ring_factory):
    ring = ring_factory(capacity=2, data=["a", "b"])
    ring.extend(["c", "d", "e"])  # a..c dropped

    assert list(ring.entries(after=0)) == [(3, "d"), (4, "e")]


# ============================================================================
# NESTING AND WATCHES
# ============================================================================


def test_a_dict_sets_a_ring_child_through_its_store(root_view):
    parent = root_view.open_child("runs", EagerDictView)
    parent.ensure_created()
    parent.set_child_container_as("out", ["a", "b", "c", "d"], RingView.of(3))
    ring = parent.open_child("out", RingView.of(3))
    assert list(ring) == ["b", "c", "d"]
    parent.set_child_container_as("out", ["x"], RingView.of(3))
    assert list(ring) == ["x"]


def test_store_without_replace_appends(ring_factory):
    ring = ring_factory(data=["a"])
    ring.store(["b"], replace=False)
    assert list(ring) == ["a", "b"]


def test_ring_inside_a_dict_survives_rebuild(root_view):
    run = root_view.open_child("run", EagerDictView)
    run.open_child("out", RingView.of(2)).extend(["a", "b", "c"])

    run.rebuild_bookkeeping()  # walks the ring child by its stored structure
    assert list(run.open_child("out", RingView)) == ["b", "c"]


def test_children_watch_fires_per_appended_and_dropped_row():
    from tests.support.observed import Hooks, observed_storage

    with observed_storage() as observed:
        h = Hooks(*observed, view_cls=RingView.of(2))
        h.write(lambda v: v.extend(["a", "b"]))
        h.watch_view(change=lambda v: v.on_change(), children=lambda v: v.on_children_change())

        hits = h.run(lambda v: v.append("c"))
        assert [k[-1] for k in hits["children"]] == [0, 2]
        assert len(hits["change"]) == 2


# ============================================================================
# COST: constant per append, whatever the capacity
# ============================================================================


@pytest.fixture(params=["sqlite", "lmdb", "rocksdb"])
def costed(request, tmp_path, monkeypatch):
    from tests.support.counting import install_counter, open_backend

    storage = open_backend(request.param, tmp_path)
    try:
        yield storage, install_counter(storage, monkeypatch)
    finally:
        storage.close()


def test_append_to_a_full_ring_costs_the_same_at_any_capacity(costed):
    from tests.support.counting import measure

    storage, counts = costed
    costs = {}
    for capacity in (10, 1000):
        ring_cls = RingView.of(capacity)
        with storage.transaction() as tx:
            Navigator(storage).root(tx).open_child(f"r{capacity}", ring_cls).extend(
                range(capacity + 5)
            )
        costs[capacity] = measure(
            storage,
            counts,
            lambda root, c=capacity: root.open_child(f"r{c}", RingView.of(c)).append(1),
        )
        with storage.snapshot() as snap:
            ring = Navigator(storage).root(snap).open_child(f"r{capacity}", ring_cls)
            assert len(ring) == capacity
            assert ring[-1] == 1

    assert costs[10] == costs[1000], costs
    assert costs[10]["put"] == 1, costs
    assert costs[10]["delete"] == 1, costs
    assert costs[10]["scan_rows"] <= 8, costs


def test_len_and_position_read_cost_is_constant(costed):
    from tests.support.counting import POINT_OPS, measure

    storage, counts = costed
    costs = {}
    for capacity in (10, 1000):
        with storage.transaction() as tx:
            Navigator(storage).root(tx).open_child(f"r{capacity}", RingView.of(capacity)).extend(
                range(capacity)
            )
        costs[capacity] = measure(
            storage,
            counts,
            lambda root, c=capacity: (
                len(root.open_child(f"r{c}", RingView)),
                root.open_child(f"r{c}", RingView)[5],
            ),
        )
    assert costs[10] == costs[1000], costs
    assert sum(costs[10][name] for name in POINT_OPS) <= 2, costs
