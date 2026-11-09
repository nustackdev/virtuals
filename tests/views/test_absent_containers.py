"""Absent containers: reads that need the container raise, point reads stay free.

For a never-written container, on every view kind:

- ``len``, iteration, ``in`` and ``extract`` raise ``ContainerNotFoundError``
  (a ``KeyError``) naming the view's own site.
- a point read raises ``KeyError`` / ``IndexError``, as on a missing key.
- adding mutations create the container, removing mutations act as on an
  empty one and write nothing.

A read checks for the container only when its own read came back empty:
data found proves the container exists, so reads on a non-empty container
cost nothing extra.
"""

from __future__ import annotations

import pytest
from tests.support.counting import (
    BACKENDS,
    SEEKING_BACKENDS,
    install_counter,
    measure,
    open_backend,
)

from virtuals._views import (
    ByteArrayView,
    EagerDictView,
    EagerFractionalIndexedDictView,
    EagerIndexedDictView,
    EagerKh57View,
    EagerListView,
    EagerLogIndexedDictView,
    FlatDictView,
    FlatListView,
    FrozenSetView,
    LazyDictView,
    LazyFractionalIndexedDictView,
    LazyIndexedDictView,
    LazyKh57View,
    LazyListView,
    LazyLogIndexedDictView,
    LightDictView,
    RingView,
    SetView,
    TupleView,
)
from virtuals.container import ContainerNotFoundError
from virtuals.navigator import Navigator


MAPPINGS = (
    EagerDictView,
    LazyDictView,
    FlatDictView,
    LightDictView,
    EagerIndexedDictView,
    LazyIndexedDictView,
    EagerLogIndexedDictView,
    LazyLogIndexedDictView,
    EagerFractionalIndexedDictView,
    LazyFractionalIndexedDictView,
    EagerKh57View,
    LazyKh57View,
)
SEQUENCES = (EagerListView, LazyListView, FlatListView, ByteArrayView, TupleView)
SETS = (SetView, FrozenSetView)
ALL_VIEWS = (*MAPPINGS, *SEQUENCES, *SETS)

KH57 = (EagerKh57View, LazyKh57View)


def _key(view_cls):
    return 1 if view_cls in KH57 else "k"


def _member(view_cls):
    """A value ``in`` can ask about."""
    if view_cls in MAPPINGS:
        return _key(view_cls)
    return 7


def _fill(view):
    """Write one item through the view's own adding door."""
    cls = type(view)
    if cls in MAPPINGS:
        view[_key(cls)] = 1
    elif cls in (TupleView, FrozenSetView):
        view.store([7])
    elif cls is SetView:
        view.add(7)
    else:
        view.append(7)


def _extract(view):
    return view.eager.extract()


@pytest.fixture(params=BACKENDS)
def storage(request, tmp_path):
    storage = open_backend(request.param, tmp_path)
    try:
        yield storage
    finally:
        storage.close()


@pytest.fixture
def root(storage):
    with storage.transaction() as tx:
        root = Navigator(storage).root(tx)
        root.ensure_created()
        yield root


# ============================================================================
# PRESENCE AND FACETS, DECLARED ON EVERY VIEW
# ============================================================================


@pytest.mark.parametrize("view_cls", [*ALL_VIEWS, RingView.of(3)])
def test_exists_is_false_until_written(root, view_cls):
    view = root.open_child("absent", view_cls)
    assert view.exists() is False
    view.ensure_created()
    assert view.exists() is True


@pytest.mark.parametrize("view_cls", [*ALL_VIEWS, RingView.of(3)])
def test_every_view_has_both_facets(root, view_cls):
    view = root.open_child("v", view_cls)
    for facet in (view.eager, view.lazy):
        assert facet.container == view.container
    if not view_cls.__name__.startswith(("Eager", "Lazy")):
        assert view.eager is view
        assert view.lazy is view


# ============================================================================
# READS ON A NEVER-WRITTEN CONTAINER
# ============================================================================


@pytest.mark.parametrize("view_cls", ALL_VIEWS)
@pytest.mark.parametrize("nested", (False, True))
def test_reads_that_need_the_container_raise(root, view_cls, nested):
    parent = root.open_child("outer", EagerDictView) if nested else root
    view = parent.open_child("absent", view_cls)
    reads = {
        "len": len,
        "iter": list,
        "contains": lambda v: _member(view_cls) in v,
        "extract": _extract,
    }
    for name, read in reads.items():
        with pytest.raises(ContainerNotFoundError) as info:
            read(view)
        assert isinstance(info.value, KeyError), name
        assert info.value.site == view.container.site, name
        assert str(view.container.site) in str(info.value), name
    assert not view.container.exists()


@pytest.mark.parametrize("view_cls", ALL_VIEWS)
def test_point_read_on_absent_container_is_a_missing_key(root, view_cls):
    view = root.open_child("absent", view_cls)
    key = _key(view_cls) if view_cls in MAPPINGS else 0
    if view_cls in SETS:
        pytest.skip("sets have no point read")
    with pytest.raises((KeyError, IndexError)):
        view[key]


@pytest.mark.parametrize("view_cls", ALL_VIEWS)
def test_reads_on_an_existing_empty_container_are_empty(root, view_cls):
    view = root.open_child("empty", view_cls)
    view.ensure_created()
    assert len(view) == 0
    assert list(view) == []
    assert _member(view_cls) not in view
    assert len(_extract(view)) == 0


@pytest.mark.parametrize("view_cls", ALL_VIEWS)
def test_reads_after_a_write_work(root, view_cls):
    view = root.open_child("full", view_cls)
    _fill(view)
    assert len(view) == 1
    assert len(list(view)) == 1
    assert _member(view_cls) in view
    assert len(_extract(view)) == 1


def test_internal_layout_views_name_their_own_site(root):
    for view_cls in (EagerIndexedDictView, EagerLogIndexedDictView, EagerFractionalIndexedDictView):
        view = root.open_child("absent", view_cls)
        for read in (len, list, lambda v: "x" in v):
            with pytest.raises(ContainerNotFoundError) as info:
                read(view)
            assert info.value.site == ("/", "absent")


def test_ordered_view_edges_raise_on_absent_and_on_empty(root):
    view = root.open_child("absent", EagerFractionalIndexedDictView)
    for read in (lambda v: v.first(), lambda v: v.last(), bool):
        with pytest.raises(ContainerNotFoundError):
            read(view)
    view.ensure_created()
    for read in (lambda v: v.first(), lambda v: v.last()):
        with pytest.raises(IndexError):
            read(view)
    assert not view


# ============================================================================
# MUTATIONS ON A NEVER-WRITTEN CONTAINER
# ============================================================================


@pytest.mark.parametrize(
    "view_cls", (*MAPPINGS, EagerListView, FlatListView, ByteArrayView, SetView)
)
def test_adding_mutation_creates_the_container(root, view_cls):
    view = root.open_child("absent", view_cls)
    _fill(view)
    assert view.container.exists()
    assert len(view) == 1


@pytest.mark.parametrize("view_cls", (EagerListView, LazyListView, ByteArrayView))
def test_list_removals_on_absent_act_as_on_empty(root, view_cls):
    view = root.open_child("absent", view_cls)
    with pytest.raises(IndexError):
        view.eager.pop() if hasattr(view, "eager") else view.pop()
    with pytest.raises(ValueError):
        view.remove(7)
    with pytest.raises(IndexError):
        del view[0]
    view.clear()
    assert not view.container.exists()


def test_set_removals_on_absent_act_as_on_empty(root):
    view = root.open_child("absent", SetView)
    view.discard(7)
    with pytest.raises(KeyError):
        view.remove(7)
    view.clear()
    assert not view.container.exists()


@pytest.mark.parametrize("view_cls", MAPPINGS)
def test_mapping_removals_on_absent_act_as_on_empty(root, view_cls):
    view = root.open_child("absent", view_cls)
    view = view.eager if hasattr(view, "eager") else view
    key = _key(view_cls)
    if hasattr(view, "pop"):
        with pytest.raises(KeyError):
            view.pop(key)
        assert view.pop(key, "dflt") == "dflt"
    with pytest.raises(KeyError):
        del view[key]
    view.clear()
    assert not view.container.exists()


# ============================================================================
# COST: point reads free, scans see the marker row
# ============================================================================


@pytest.fixture(params=SEEKING_BACKENDS)
def costed(request, tmp_path, monkeypatch):
    storage = open_backend(request.param, tmp_path)
    try:
        yield storage, install_counter(storage, monkeypatch)
    finally:
        storage.close()


def test_point_reads_cost_one_get(costed):
    storage, counts = costed
    with storage.transaction() as tx:
        root = Navigator(storage).root(tx)
        root.open_child("d", EagerDictView)["a"] = 1
        root.open_child("f", FlatDictView)["a"] = 1
    for name, cls in (("d", EagerDictView), ("f", FlatDictView), ("missing", EagerDictView)):

        def read(r, name=name, cls=cls):
            try:
                r.open_child(name, cls)["a"]
            except KeyError:
                pass

        cost = measure(storage, counts, read)
        assert cost["get"] == 1 and cost["exists"] == 0 and cost["scan"] == 0, (name, cost)


def _walk(view):
    n = 0
    for _ in view:
        n += 1
    return n


def test_iteration_and_len_on_dicts_need_no_extra_read(costed):
    storage, counts = costed
    with storage.transaction() as tx:
        Navigator(storage).root(tx).open_child("d", EagerDictView).store({"a": 1, "b": 2})
    # list() would ask len() first, so iterate by hand
    for op in (len, _walk, lambda v: v.extract()):
        cost = measure(storage, counts, lambda r, op=op: op(r.open_child("d", EagerDictView)))
        assert cost["scan"] == 1 and cost["get"] == 0 and cost["exists"] == 0, cost

    def absent(r):
        with pytest.raises(ContainerNotFoundError):
            _walk(r.open_child("absent", EagerDictView))

    cost = measure(storage, counts, absent)
    # the absent case is the same one scan: no marker row came back
    assert cost["scan"] == 1 and cost["get"] == 0 and cost["exists"] == 0, cost


def test_contains_hit_costs_one_get_and_miss_one_marker_read(costed):
    storage, counts = costed
    with storage.transaction() as tx:
        Navigator(storage).root(tx).open_child("d", EagerDictView)["a"] = 1
    hit = measure(storage, counts, lambda r: "a" in r.open_child("d", EagerDictView))
    assert hit["get"] + hit["exists"] == 1, hit
    miss = measure(storage, counts, lambda r: "z" in r.open_child("d", EagerDictView))
    assert miss["get"] + miss["exists"] == 2, miss


def test_list_len_reads_the_counter_only(costed):
    storage, counts = costed
    with storage.transaction() as tx:
        Navigator(storage).root(tx).open_child("l", EagerListView).append(1)
    cost = measure(storage, counts, lambda r: len(r.open_child("l", EagerListView)))
    assert cost["get"] == 1 and cost["exists"] == 0 and cost["scan"] == 0, cost
    cost = measure(storage, counts, lambda r: r.open_child("l", EagerListView)[0])
    assert cost["get"] + cost["exists"] == 2 and cost["scan"] == 0, cost  # counter + item
