"""Hardening of every dict view against storage, on every backend.

What the contract asks of a dict view (existence, EMPTY, cost) and the
edges that tend to break at the storage boundary: odd keys, deep nesting,
large values, erase then len. Each case runs on mem, sqlite, lmdb and
rocksdb; op counts run on the seeking backends only.
"""

from __future__ import annotations

import threading

import pytest
from tests.support.counting import (
    BACKENDS,
    POINT_OPS,
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
    LazyDictView,
    LazyFractionalIndexedDictView,
    LazyIndexedDictView,
    LazyKh57View,
    LazyListView,
    LazyLogIndexedDictView,
    LightDictView,
    SetView,
)
from virtuals.container import ContainerNotFoundError, ContainerTypeError
from virtuals.navigator import Navigator
from virtuals.types import EMPTY


NESTED_VIEWS = (
    EagerDictView,
    LazyDictView,
    EagerIndexedDictView,
    EagerLogIndexedDictView,
    EagerFractionalIndexedDictView,
)
"""Dict views whose values may be compound (a child container)."""

FLAT_VIEWS = (FlatDictView, LightDictView, EagerKh57View)
"""Dict views whose values are primitives only."""

DICT_VIEWS = (*NESTED_VIEWS, *FLAT_VIEWS)


def _key(view_cls, i):
    """A key of the type the view takes (Kh57 takes ints only)."""
    return i if view_cls in (EagerKh57View, LazyKh57View) else f"k{i}"


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture(params=BACKENDS)
def backend(request):
    return request.param


@pytest.fixture
def storage(backend, tmp_path):
    storage = open_backend(backend, tmp_path)
    try:
        yield storage
    finally:
        storage.close()


@pytest.fixture
def root(storage):
    with storage.transaction() as tx:
        yield Navigator(storage).root(tx)


@pytest.fixture(params=SEEKING_BACKENDS)
def costed(request, tmp_path, monkeypatch):
    storage = open_backend(request.param, tmp_path)
    try:
        yield storage, install_counter(storage, monkeypatch)
    finally:
        storage.close()


# ============================================================================
# EXISTENCE: a delete never creates the view (contract 6, V8)
# ============================================================================


@pytest.mark.parametrize("view_cls", DICT_VIEWS)
def test_delete_on_absent_view_does_not_create_it(root, view_cls):
    view = root.open_child("absent", view_cls)
    key = _key(view_cls, 1)
    with pytest.raises(KeyError):
        del view[key]
    _assert_absent(view)
    assert _lacks(view, key)


@pytest.mark.parametrize("view_cls", (EagerDictView, FlatDictView, LightDictView))
def test_delete_on_absent_nested_view_leaves_the_chain_absent(root, view_cls):
    """``del a.b["k"]`` on an absent chain stamps neither ``a`` nor ``b``."""
    outer = root.open_child("a", EagerDictView)
    inner = outer.open_child("b", view_cls)
    with pytest.raises(KeyError):
        del inner["k"]
    assert not inner.container.exists()
    assert not outer.container.exists()
    assert _lacks(root, "a")


@pytest.mark.parametrize("view_cls", (EagerDictView, LazyDictView, FlatDictView, LightDictView))
def test_delete_missing_key_on_existing_view_raises_and_keeps_the_rest(root, view_cls):
    view = root.open_child("d", view_cls)
    view["a"] = 1
    with pytest.raises(KeyError):
        del view["zz"]
    assert list(view) == ["a"]
    assert len(view) == 1


def _delete_missing(root, view_cls):
    with pytest.raises(KeyError):
        del root.open_child("d", view_cls)["zz"]


@pytest.mark.parametrize("view_cls", (EagerDictView, FlatDictView))
def test_delete_missing_key_cost_is_constant_and_writes_nothing(costed, view_cls):
    storage, counts = costed
    for present in (False, True):
        if present:
            with storage.transaction() as tx:
                Navigator(storage).root(tx).open_child("d", view_cls).store({"a": 1})
        cost = measure(storage, counts, lambda root: _delete_missing(root, view_cls))
        assert cost["put"] == 0 and cost["delete"] == 0, (present, cost)
        assert cost["scan"] == 0, (present, cost)
        assert cost["get"] + cost["exists"] <= 3, (present, cost)


# ============================================================================
# EMPTY IS NEVER STORED (contract 6, V6)
# ============================================================================


def _rows_under(view):
    """Every storage row under the view's site, as decoded keys."""
    from virtuals.tkv.filter import PrefixFilter
    from virtuals.tkv.storage import StorageScanOptions

    site = view.container.site
    prefix = PrefixFilter(prefix=site)
    opts = StorageScanOptions(start=site, break_filter=prefix, filter=prefix)
    return [k for k, _ in view.container.ctx.scan(opts).items()]


@pytest.mark.parametrize("view_cls", DICT_VIEWS)
def test_set_empty_raises_and_stores_nothing(root, view_cls):
    view = root.open_child("d", view_cls)
    view[_key(view_cls, 1)] = 1
    rows = _rows_under(view)
    with pytest.raises(ValueError, match="EMPTY"):
        view[_key(view_cls, 2)] = EMPTY
    assert _rows_under(view) == rows
    assert len(view) == 1
    assert list(view) == [_key(view_cls, 1)]
    assert _lacks(view, _key(view_cls, 2))
    assert list(view.items()) == [(_key(view_cls, 1), 1)]


@pytest.mark.parametrize("view_cls", DICT_VIEWS)
def test_set_empty_on_absent_view_leaves_it_absent(root, view_cls):
    view = root.open_child("absent", view_cls)
    with pytest.raises(ValueError, match="EMPTY"):
        view[_key(view_cls, 1)] = EMPTY
    assert not view.container.exists()


@pytest.mark.parametrize("view_cls", DICT_VIEWS)
def test_store_and_update_with_empty_raise(root, view_cls):
    view = root.open_child("d", view_cls)
    with pytest.raises(ValueError, match="EMPTY"):
        view.store({_key(view_cls, 1): 1, _key(view_cls, 2): EMPTY})
    with pytest.raises(ValueError, match="EMPTY"):
        view.update({_key(view_cls, 3): EMPTY})
    assert _lacks(view, _key(view_cls, 2))
    assert _lacks(view, _key(view_cls, 3))


@pytest.mark.parametrize("view_cls", NESTED_VIEWS)
def test_nested_empty_raises(root, view_cls):
    """EMPTY inside a compound value is rejected at the leaf."""
    view = root.open_child("d", view_cls)
    with pytest.raises(ValueError, match="EMPTY"):
        view["a"] = {"x": 1, "y": {"z": EMPTY}}
    with pytest.raises(ValueError, match="EMPTY"):
        view["b"] = [1, EMPTY]


@pytest.mark.parametrize("view_cls", (EagerLogIndexedDictView, EagerFractionalIndexedDictView))
def test_set_primitive_empty_raises(root, view_cls):
    view = root.open_child("d", view_cls)
    view.store({"a": 1})
    with pytest.raises(ValueError, match="EMPTY"):
        view.set_primitive("e", EMPTY)
    assert list(view) == ["a"]
    assert len(view) == 1


def test_fractional_insert_empty_leaves_no_trace(root):
    view = root.open_child("lst", EagerFractionalIndexedDictView)
    view.store({"a": 1})
    for op in (
        lambda: view.append("e", EMPTY),
        lambda: view.prepend("e", EMPTY),
        lambda: view.insert_after("a", "e", EMPTY),
    ):
        with pytest.raises(ValueError, match="EMPTY"):
            op()
        assert _lacks(view, "e")
        assert len(view) == 1
        assert list(view) == ["a"]
        assert view.get("e", "dflt") == "dflt"


def test_container_put_child_primitive_rejects_empty(root):
    view = root.open_child("d", EagerDictView)
    view["a"] = 1
    with pytest.raises(ValueError, match="EMPTY"):
        view.container.put_child_primitive("e", EMPTY)
    with pytest.raises(ValueError, match="EMPTY"):
        view.container.put_child_primitive("e", EMPTY, validate=False, validate_parent=False)
    with pytest.raises(ValueError, match="EMPTY"):
        view.container.put_child_primitive_unsafe("e", EMPTY)
    with pytest.raises(ValueError, match="EMPTY"):
        view.container.put_child_primitive_unsafe("e", {"n": [EMPTY]})
    with pytest.raises(ValueError, match="EMPTY"):
        view.container.put_metadata("m", EMPTY)
    assert list(view) == ["a"]
    assert view.container.get_metadata("m", "dflt") == "dflt"


# ============================================================================
# ERASE THEN LEN: every dict view, same transaction and a fresh one
# ============================================================================


@pytest.mark.parametrize("view_cls", DICT_VIEWS)
def test_erase_then_len_agrees_everywhere(storage, view_cls):
    keys = [_key(view_cls, i) for i in range(3)]
    with storage.transaction() as tx:
        view = Navigator(storage).root(tx).open_child("d", view_cls)
        for i, key in enumerate(keys):
            view[key] = i
        del view[keys[1]]
        assert len(view) == 2
        assert list(view) == [keys[0], keys[2]]
        assert _lacks(view, keys[1])
        assert bool(view)
    with storage.transaction() as tx:
        view = Navigator(storage).root(tx).open_child("d", view_cls)
        assert len(view) == 2
        assert sorted(view, key=str) == sorted([keys[0], keys[2]], key=str)
        assert _lacks(view, keys[1])
        del view[keys[0]]
        del view[keys[2]]
        assert len(view) == 0
        assert not view
        assert list(view) == []
    with storage.transaction() as tx:
        view = Navigator(storage).root(tx).open_child("d", view_cls)
        assert len(view) == 0
        assert not view
        assert view.container.exists()


@pytest.mark.parametrize("view_cls", NESTED_VIEWS)
def test_erase_compound_entry_then_len(storage, view_cls):
    with storage.transaction() as tx:
        view = Navigator(storage).root(tx).open_child("d", view_cls)
        view["a"] = 1
        view["b"] = {"x": {"y": {"z": 1}}, "w": [1, 2, 3]}
        assert len(view) == 2
        del view["b"]
        assert len(view) == 1
        assert _lacks(view, "b")
        assert list(view.items()) == [("a", 1)]
    with storage.transaction() as tx:
        view = Navigator(storage).root(tx).open_child("d", view_cls)
        assert len(view) == 1
        assert view.get("b", "dflt") == "dflt"
        # Nothing of the old subtree is left under the key.
        child = view.open_child("b", EagerDictView)
        assert not child.container.exists()
        with pytest.raises(ContainerNotFoundError):
            len(child)


@pytest.mark.parametrize("view_cls", [v for v in DICT_VIEWS if hasattr(v, "pop")])
def test_pop_then_len(storage, view_cls):
    with storage.transaction() as tx:
        view = Navigator(storage).root(tx).open_child("d", view_cls)
        keys = [_key(view_cls, i) for i in range(3)]
        view.update(dict.fromkeys(keys, 7))
        assert view.pop(keys[0]) == 7
        assert view.pop(_key(view_cls, 9), "dflt") == "dflt"
        assert len(view) == 2
        assert list(view) == keys[1:]


# ============================================================================
# KEY EDGES: unicode, separators, int extremes, True/1 (contract: point ops)
# ============================================================================

EDGE_STR_KEYS = [
    "",
    "0",
    "a/b",
    "a\x00b",
    "\x01",
    "é",
    "😀",
    "*",
    "__data__",
    "__len__",
    " ",
    "a.b",
]
EDGE_INT_KEYS = [0, -1, 1, 2**31, -(2**31), 2**63 - 1, -(2**63)]
KH57_INT_KEYS = [0, 1, 2**31, 2**40]  # kh57 keys are non-negative


def _read(view, key):
    """The value at ``key`` as a Python value (lazy facets hand back views)."""
    value = view[key]
    return value.extract() if hasattr(value, "extract") else value


@pytest.mark.parametrize("view_cls", DICT_VIEWS)
def test_edge_keys_round_trip(storage, view_cls):
    keys = KH57_INT_KEYS if view_cls is EagerKh57View else [*EDGE_STR_KEYS, *EDGE_INT_KEYS]
    with storage.transaction() as tx:
        view = Navigator(storage).root(tx).open_child("d", view_cls)
        for i, key in enumerate(keys):
            view[key] = i
        assert len(view) == len(keys)
        for i, key in enumerate(keys):
            assert key in view
            assert view[key] == i
        assert sorted(view, key=repr) == sorted(keys, key=repr)
    with storage.transaction() as tx:
        view = Navigator(storage).root(tx).open_child("d", view_cls)
        assert len(view) == len(keys)
        assert dict(view.items()) == {k: i for i, k in enumerate(keys)}
        for key in keys[::2]:
            del view[key]
        assert len(view) == len(keys) - len(keys[::2])
        for key in keys[::2]:
            assert _lacks(view, key)
        for key in keys[1::2]:
            assert key in view


@pytest.mark.parametrize("view_cls", NESTED_VIEWS)
def test_edge_keys_as_nested_container_names(storage, view_cls):
    with storage.transaction() as tx:
        view = Navigator(storage).root(tx).open_child("d", view_cls)
        for key in EDGE_STR_KEYS:
            view[key] = {"v": key, "nested": {key: 1}}
        assert len(view) == len(EDGE_STR_KEYS)
        for key in EDGE_STR_KEYS:
            assert _read(view, key) == {"v": key, "nested": {key: 1}}
        del view["a/b"]
        del view["a\x00b"]
        assert len(view) == len(EDGE_STR_KEYS) - 2
        assert sorted(view) == sorted(k for k in EDGE_STR_KEYS if k not in ("a/b", "a\x00b"))


def test_separator_in_key_does_not_leak_into_siblings(root):
    """``a/b`` is one key, never the child ``b`` of a key ``a``."""
    view = root.open_child("d", EagerDictView)
    view["a"] = {"b": 1}
    view["a/b"] = 2
    assert sorted(view) == ["a", "a/b"]
    assert view["a"] == {"b": 1}
    assert view["a/b"] == 2
    del view["a"]
    assert list(view) == ["a/b"]
    assert view["a/b"] == 2


@pytest.mark.parametrize("view_cls", (EagerDictView, FlatDictView, EagerFractionalIndexedDictView))
def test_out_of_range_int_key_is_rejected_cleanly(root, view_cls, backend):
    """Ints past int64 do not fit a key: the same clear error on every op and backend."""
    from virtuals.tkv.storage.exceptions import StorageOperationError

    view = root.open_child("d", view_cls)
    view["a"] = 1
    with pytest.raises(StorageOperationError, match="out of supported range"):
        view[2**64 + 5] = 1
    with pytest.raises(StorageOperationError, match="out of supported range"):
        2**64 + 5 in view  # noqa: B015
    assert list(view) == ["a"]


@pytest.mark.parametrize("view_cls", (EagerDictView, FlatDictView, EagerFractionalIndexedDictView))
def test_true_and_one_are_the_same_key(root, view_cls):
    """Python dict semantics: ``True == 1``, one key. The stored key reads back as ``1``."""
    view = root.open_child("d", view_cls)
    view[1] = "int"
    view[True] = "bool"
    assert len(view) == 1
    assert view[1] == "bool"
    assert view[True] == "bool"
    assert True in view
    assert list(view) == [1]
    # The key's Python type is the codec's business: the binary codec reads
    # back an int, mem keeps the last object written. They compare equal.
    assert next(iter(view)) == 1
    del view[True]
    assert _lacks(view, 1)
    assert len(view) == 0


# ============================================================================
# DEEP NESTING AND LARGE VALUES
# ============================================================================


def test_deep_nesting_round_trip(storage, backend):
    """40 levels of dicts: write through views, read back whole, delete the root."""
    depth = 40
    value = {"leaf": 1}
    for i in reversed(range(depth)):
        value = {f"l{i}": value}
    with storage.transaction() as tx:
        view = Navigator(storage).root(tx).open_child("deep", EagerDictView)
        view.store(value)
        cur = view
        for i in range(depth):
            cur = cur.open_child(f"l{i}", EagerDictView)
        assert cur["leaf"] == 1
        cur["leaf2"] = 2
    with storage.transaction() as tx:
        view = Navigator(storage).root(tx).open_child("deep", EagerDictView)
        got = view.extract()
        cur = got
        for i in range(depth):
            cur = cur[f"l{i}"]
        assert cur == {"leaf": 1, "leaf2": 2}
        assert len(view) == 1
        del view["l0"]
        assert len(view) == 0
        assert not view.open_child("l0", EagerDictView).container.exists()


def test_deep_nesting_past_the_lmdb_key_limit(storage, backend):
    """lmdb rejects encoded keys above 511 bytes; the other backends take them.

    The limit is lmdb's own (``MDB_BAD_VALSIZE``) and surfaces as a generic
    StorageOperationError on put. Documented here and in the
    FractionalIndexedDictView module notes; nothing in the lmdb backend
    names it yet.
    """
    from virtuals.tkv.storage.exceptions import StorageOperationError

    with storage.transaction() as tx:
        view = Navigator(storage).root(tx).open_child("deep", EagerDictView)
        cur = view
        for i in range(120):
            cur = cur.open_child(f"level{i:03d}", EagerDictView)
        if backend == "lmdb":
            with pytest.raises(StorageOperationError):
                cur["leaf"] = 1
        else:
            cur["leaf"] = 1
            assert cur["leaf"] == 1


@pytest.mark.parametrize("view_cls", (EagerDictView, FlatDictView))
def test_long_key_past_the_lmdb_key_limit(root, view_cls, backend):
    from virtuals.tkv.storage.exceptions import StorageOperationError

    view = root.open_child("d", view_cls)
    view["a"] = 1
    key = "k" * 600
    if backend == "lmdb":
        with pytest.raises(StorageOperationError):
            view[key] = 1
        assert list(view) == ["a"]
    else:
        view[key] = 1
        assert key in view
        assert len(view) == 2
        del view[key]
        assert list(view) == ["a"]


@pytest.mark.parametrize("view_cls", (EagerDictView, FlatDictView, EagerLogIndexedDictView))
def test_large_values_round_trip(storage, view_cls):
    blob = bytes(range(256)) * (4 * 2**12)  # 4 MiB
    text = "é😀" * (2**19)  # 1 M chars, 3 MiB utf-8
    with storage.transaction() as tx:
        view = Navigator(storage).root(tx).open_child("big", view_cls)
        view["blob"] = blob
        view["text"] = text
        view["small"] = 1
    with storage.transaction() as tx:
        view = Navigator(storage).root(tx).open_child("big", view_cls)
        assert view["blob"] == blob
        assert view["text"] == text
        assert len(view) == 3
        view["blob"] = b""
        assert view["blob"] == b""
        del view["text"]
        assert sorted(view) == ["blob", "small"]


# ============================================================================
# COST: bool is one seek on every dict view, len never reads the data
# ============================================================================


@pytest.mark.parametrize(
    "view_cls",
    (
        EagerDictView,
        FlatDictView,
        LightDictView,
        EagerKh57View,
        EagerLogIndexedDictView,
        EagerFractionalIndexedDictView,
    ),
)
def test_bool_is_one_seek_on_every_live_counted_view(costed, view_cls):
    storage, counts = costed
    costs = {}
    for n in (10, 1000):
        with storage.transaction() as tx:
            view = Navigator(storage).root(tx).open_child("d", view_cls)
            view.store(
                {_key(view_cls, i): i if view_cls in FLAT_VIEWS else {"a": i} for i in range(n)}
            )
        costs[n] = measure(storage, counts, lambda root: bool(root.open_child("d", view_cls)))
    assert costs[10] == costs[1000], costs
    # one storage op: a seek (scan of at most two rows) or, on Kh57, one get
    assert sum(costs[10][k] for k in (*POINT_OPS, "scan")) == 1, costs
    assert costs[10]["scan_rows"] <= 2, costs


@pytest.mark.parametrize("view_cls", (LightDictView, EagerKh57View))
def test_bool_on_absent_flat_view_raises_and_writes_nothing(root, view_cls):
    view = root.open_child("absent", view_cls)
    with pytest.raises(ContainerNotFoundError):
        bool(view)
    assert not view.container.exists()
    assert _rows_under(view) == []
    view[_key(view_cls, 1)] = 1
    assert view


# ============================================================================
# COST: known linear ops, pinned as accepted (contract 7, Q12)
# ============================================================================


def _linear_ops():
    def indexed_delete(root, n):
        view = root.open_child("d", EagerIndexedDictView)
        del view[f"k{n // 2}"]

    def list_insert(root, n):
        root.open_child("l", EagerListView).insert(n // 2, -1)

    def list_delete(root, n):
        del root.open_child("l", EagerListView)[n // 2]

    return (
        pytest.param(EagerIndexedDictView, indexed_delete, id="indexed_dict_delete"),
        pytest.param(EagerListView, list_insert, id="list_insert_mid"),
        pytest.param(EagerListView, list_delete, id="list_delete_mid"),
    )


@pytest.mark.parametrize("view_cls, op", _linear_ops())
def test_known_linear_ops_grow_with_n_and_stay_linear(costed, view_cls, op):
    """IndexedDictView delete rewrites its key list; ListView insert and
    delete shift the tail. Both are O(n) by design and documented as such.
    This pins the cost as accepted: if one of them becomes constant, drop
    its row here and its note in the docs."""
    storage, counts = costed
    costs = {}
    for n in (20, 200):
        with storage.transaction() as tx:
            view = (
                Navigator(storage)
                .root(tx)
                .open_child("d" if view_cls is EagerIndexedDictView else "l", view_cls)
            )
            if view_cls is EagerIndexedDictView:
                view.store({f"k{i}": i for i in range(n)})
            else:
                view.store(list(range(n)))
        costs[n] = measure(storage, counts, lambda root, n=n: op(root, n))
    points = {n: sum(c[k] for k in POINT_OPS) for n, c in costs.items()}
    assert points[200] > points[20], costs
    assert points[200] <= 8 * 200, costs


@pytest.mark.parametrize(
    "view_cls",
    (EagerDictView, EagerLogIndexedDictView, EagerFractionalIndexedDictView),
)
def test_len_scans_children_not_descendants(costed, view_cls):
    storage, counts = costed
    n = 20
    costs = {}
    for width in (5, 200):
        with storage.transaction() as tx:
            view = Navigator(storage).root(tx).open_child("d", view_cls)
            view.store({f"k{i}": {f"f{j}": j for j in range(width)} for i in range(n)})
        costs[width] = measure(storage, counts, lambda root: len(root.open_child("d", view_cls)))
    assert costs[5] == costs[200], costs
    assert costs[5]["scan_rows"] <= 2 * n + 2, costs


# ============================================================================
# LOGINDEXED REVERSE INDEX: stale, wrong or missing rows (mixed-version stores)
# ============================================================================


def _log_key_of(view, key):
    return next(lk for lk, ak in view.keys_with_log_keys() if ak == key)


def _bypass_delete_and_reset(view, key, value):
    """What a writer that does not know ``__log__/`` does: drop data and log row, re-set both."""
    from virtuals._views.log_indexed_dict_view import _generate_log_key

    view._keys_container().delete_child_primitive_unsafe(_log_key_of(view, key))
    view._data_container().delete_child(key)
    view._data_container().put_child_primitive(key, value)
    view._keys_container().put_child_primitive(_generate_log_key(), key)


def test_log_indexed_delete_with_stale_reverse_row_falls_back_to_scan(root):
    """A reverse row pointing at a dead log key must not leave a ghost key."""
    view = root.open_child("d", EagerLogIndexedDictView)
    view.store({"a": 1, "c": 3, "e": 5})
    stale = view._log_container().get_child_primitive_unsafe("a")
    _bypass_delete_and_reset(view, "a", 10)
    assert view._log_container().get_child_primitive_unsafe("a") == stale
    assert view["a"] == 10
    del view["a"]
    assert _lacks(view, "a")
    assert list(view) == ["c", "e"]
    assert len(view) == 2
    assert list(view.items()) == [("c", 3), ("e", 5)]
    assert sorted(view._log_container().iter_child_keys(validate=False)) == ["c", "e"]


def test_log_indexed_delete_with_wrong_reverse_row_keeps_the_other_key(root):
    """A reverse row naming another key's log key must not delete that key's log row."""
    view = root.open_child("d", EagerLogIndexedDictView)
    view.store({"a": 1, "b": 2, "c": 3})
    view._log_container().put_child_primitive(
        "a", _log_key_of(view, "b"), validate=False, validate_parent=False
    )
    del view["a"]
    assert list(view) == ["b", "c"]
    assert len(view) == 2
    assert view["b"] == 2
    assert view._log_container().get_child_primitive_unsafe("b") == _log_key_of(view, "b")
    view["a"] = 11
    assert list(view) == ["b", "c", "a"]
    del view["b"]
    assert list(view) == ["c", "a"]


def test_log_indexed_old_data_without_log_container(root, storage):
    """Data written before ``__log__/`` existed: no marker, no rows, everything still works."""
    view = root.open_child("d", EagerLogIndexedDictView)
    view.store({"a": 1, "b": {"n": 2}, "c": 3})
    view.container.delete_child("__log__")
    assert not view._log_container().exists()
    assert list(view) == ["a", "b", "c"]
    assert view["b"] == {"n": 2}
    assert len(view) == 3
    # delete falls back to the scan and writes no index
    del view["b"]
    assert list(view) == ["a", "c"]
    assert len(view) == 2
    assert not view._log_container().exists()
    # the first write stamps the marker and indexes the new key only
    view["d"] = 4
    assert view._log_container().exists()
    assert list(view._log_container().iter_child_keys(validate=False)) == ["d"]
    del view["d"]
    assert list(view) == ["a", "c"]
    # rebuild fills the rest
    view.rebuild_bookkeeping()
    assert {
        k: v.primitive_value for k, v in view._log_container().iter_children(validate=False)
    } == {
        "a": _log_key_of(view, "a"),
        "c": _log_key_of(view, "c"),
    }
    del view["a"]
    assert list(view) == ["c"]


def test_log_indexed_old_data_survives_a_fresh_transaction(storage):
    with storage.transaction() as tx:
        view = Navigator(storage).root(tx).open_child("d", EagerLogIndexedDictView)
        view.store({"a": 1, "b": 2})
        view.container.delete_child("__log__")
    with storage.transaction() as tx:
        view = Navigator(storage).root(tx).open_child("d", EagerLogIndexedDictView)
        del view["a"]
        assert list(view) == ["b"]
    with storage.transaction() as tx:
        view = Navigator(storage).root(tx).open_child("d", EagerLogIndexedDictView)
        assert list(view) == ["b"]
        assert len(view) == 1
        assert _lacks(view, "a")


# ============================================================================
# EMPTY INSIDE COMPOUND AND BULK WRITES: nothing is written before the check
# ============================================================================


@pytest.mark.parametrize("view_cls", NESTED_VIEWS)
def test_nested_empty_leaves_no_partial_child(root, view_cls):
    view = root.open_child("d", view_cls)
    view["keep"] = 1
    for bad in (
        {"ok": 1, "bad": EMPTY},
        {"ok": 1, "deep": {"x": [1, {"y": EMPTY}]}},
        [1, (2, EMPTY)],
    ):
        with pytest.raises(ValueError, match="EMPTY"):
            view["y"] = bad
        assert _lacks(view, "y")
        assert len(view) == 1
        assert list(view) == ["keep"]
        assert not view.open_child("y", EagerDictView).container.exists()
        assert view.get("y", "dflt") == "dflt"


SCA_VIEWS = (*NESTED_VIEWS, EagerKh57View)
"""Views with a ``set_child_container_as`` door (the Ref write path)."""


@pytest.mark.parametrize("view_cls", SCA_VIEWS)
def test_set_child_container_as_nested_empty_leaves_no_partial_child(root, view_cls):
    """The Ref door: nothing is written before the nested EMPTY is found."""
    view = root.open_child("d", view_cls)
    keep, bad = _key(view_cls, 0), _key(view_cls, 1)
    view[keep] = 1
    rows = _rows_under(view)
    for value in ({"ok": 1, "bad": EMPTY}, {"ok": 1, "deep": {"x": [1, {"y": EMPTY}]}}):
        with pytest.raises(ValueError, match="EMPTY"):
            view.set_child_container_as(bad, value, EagerDictView)
        assert _rows_under(view) == rows
        assert _lacks(view, bad)
        assert len(view) == 1
        assert list(view) == [keep]
        assert not view.open_child(bad, EagerDictView).container.exists()
    # a lazy iterable is materialized and checked before the first write
    with pytest.raises(ValueError, match="EMPTY"):
        view.set_child_container_as(bad, (x for x in [1, {"y": EMPTY}]), EagerListView)
    assert _rows_under(view) == rows
    assert _lacks(view, bad)
    assert list(view) == [keep]
    assert not view.open_child(bad, EagerListView).container.exists()
    # and a clean lazy iterable is stored once, whole
    view.set_child_container_as(bad, (x for x in [1, 2]), EagerListView)
    assert view.open_child(bad, EagerListView).extract() == [1, 2]


@pytest.mark.parametrize("view_cls", SCA_VIEWS)
def test_set_child_container_as_nested_empty_on_absent_view_leaves_it_absent(root, view_cls):
    view = root.open_child("absent", view_cls)
    with pytest.raises(ValueError, match="EMPTY"):
        view.set_child_container_as(_key(view_cls, 1), {"bad": EMPTY}, EagerDictView)
    with pytest.raises(ValueError, match="EMPTY"):
        view.set_child_container_as(_key(view_cls, 1), (x for x in [EMPTY]), EagerListView)
    assert _rows_under(view) == []
    _assert_absent(view)


def test_list_set_child_container_as_nested_empty_writes_nothing(root):
    lst = root.open_child("lst", EagerListView)
    lst.store([1, {"a": 1}])
    rows = _rows_under(lst)
    with pytest.raises(ValueError, match="EMPTY"):
        lst.set_child_container_as(1, {"bad": {"z": EMPTY}}, EagerDictView)
    assert _rows_under(lst) == rows
    assert lst.extract() == [1, {"a": 1}]
    assert len(lst) == 2


@pytest.mark.parametrize("view_cls", DICT_VIEWS)
def test_store_with_empty_leaves_the_view_as_it_was(root, view_cls):
    view = root.open_child("d", view_cls)
    keep = _key(view_cls, 0)
    view[keep] = 1
    with pytest.raises(ValueError, match="EMPTY"):
        view.store({_key(view_cls, 1): 2, _key(view_cls, 2): EMPTY}, replace=True)
    assert list(view) == [keep]
    assert view[keep] == 1
    with pytest.raises(ValueError, match="EMPTY"):
        view.update({_key(view_cls, 1): 2, _key(view_cls, 2): EMPTY})
    assert list(view) == [keep]


def test_store_with_empty_on_absent_view_leaves_it_absent(root):
    view = root.open_child("absent", EagerDictView)
    with pytest.raises(ValueError, match="EMPTY"):
        view.store({"a": 1, "b": {"c": EMPTY}})
    assert not view.container.exists()


def test_fractional_nested_empty_insert_leaves_no_trace(root):
    """The real failure path through ``_insert``: no subclass, a nested EMPTY."""
    view = root.open_child("lst", EagerFractionalIndexedDictView)
    view.store({"a": 1, "b": 2})
    with pytest.raises(ValueError, match="EMPTY"):
        view.insert_after("a", "x", {"ok": 1, "bad": {"z": EMPTY}})
    assert _lacks(view, "x")
    assert len(view) == 2
    assert list(view) == ["a", "b"]
    assert view.get("x") is EMPTY
    assert not view._data_container().exists_child("x")
    assert list(view.items()) == [("a", 1), ("b", 2)]
    with pytest.raises(ValueError, match="EMPTY"):
        view.store({"a": 1, "x": {"bad": EMPTY}})
    assert list(view) == ["a", "b"]
    assert view["b"] == 2


def test_list_views_reject_empty_before_any_shift_or_write(root):
    from virtuals._views import EagerListView, FlatListView, SetView

    lst = root.open_child("lst", EagerListView)
    lst.store([1, 2, 3])
    with pytest.raises(ValueError, match="EMPTY"):
        lst.insert(1, EMPTY)
    with pytest.raises(ValueError, match="EMPTY"):
        lst.insert(1, {"a": EMPTY})
    with pytest.raises(ValueError, match="EMPTY"):
        lst.append(EMPTY)
    with pytest.raises(ValueError, match="EMPTY"):
        lst.extend([4, EMPTY])
    with pytest.raises(ValueError, match="EMPTY"):
        lst.store([4, EMPTY])
    with pytest.raises(ValueError, match="EMPTY"):
        lst[0] = [EMPTY]
    assert lst.extract() == [1, 2, 3]
    assert len(lst) == 3

    flat = root.open_child("flat", FlatListView)
    flat.store([1, 2])
    for op in (
        lambda: flat.append(EMPTY),
        lambda: flat.extend([3, EMPTY]),
        lambda: flat.store([EMPTY]),
    ):
        with pytest.raises(ValueError, match="EMPTY"):
            op()
    assert flat.extract() == [1, 2]
    absent = root.open_child("absent_flat", FlatListView)
    with pytest.raises(ValueError, match="EMPTY"):
        absent.append(EMPTY)
    assert not absent.container.exists()

    s = root.open_child("set", SetView)
    s.store({"a"})
    with pytest.raises(ValueError, match="EMPTY"):
        s.add(EMPTY)
    with pytest.raises(ValueError, match="EMPTY"):
        s.store(["b", EMPTY])
    assert s.extract() == {"a"}


def test_bulk_doors_still_take_lazy_iterables(root):
    from virtuals._views import EagerListView, FlatListView, SetView

    lst = root.open_child("lst", EagerListView)
    lst.store(i for i in range(3))
    lst.extend(i for i in range(3, 5))
    assert lst.extract() == [0, 1, 2, 3, 4]
    flat = root.open_child("flat", FlatListView)
    flat.store(i for i in range(2))
    flat.extend(i for i in range(2, 4))
    assert flat.extract() == [0, 1, 2, 3]
    s = root.open_child("set", SetView)
    s.store(str(i) for i in range(3))
    assert s.extract() == {"0", "1", "2"}
    f = root.open_child("f", EagerFractionalIndexedDictView)
    f.store(f"k{i}" for i in range(3))
    assert list(f) == ["k0", "k1", "k2"]


# ============================================================================
# KIND CHANGE: a key keeps its kind until it is deleted (pinned, Q3)
# ============================================================================


@pytest.mark.parametrize("view_cls", NESTED_VIEWS)
def test_changing_a_keys_kind_raises_and_keeps_the_old_value(root, view_cls):
    """Pre-existing behaviour, pinned: a primitive key cannot take a
    compound value and a compound key cannot take a primitive. The write
    raises ``ContainerTypeError`` and changes nothing. Delete first, then
    set. See docs/view/overview.md, "Write rules"."""
    view = root.open_child("d", view_cls)
    view.store({"p": 1, "c": {"z": 1}})
    rows = _rows_under(view)
    with pytest.raises(ContainerTypeError):
        view["p"] = {"z": 2}
    with pytest.raises(ContainerTypeError):
        view["c"] = 5
    with pytest.raises(ContainerTypeError):
        view.set_child_container_as("p", {"z": 2}, EagerDictView)
    assert _rows_under(view) == rows
    eager = view.eager
    assert eager.extract() == {"p": 1, "c": {"z": 1}}
    assert set(view) == {"p", "c"}
    del view["p"]
    view["p"] = {"z": 2}
    del view["c"]
    view["c"] = 5
    assert eager.extract() == {"p": {"z": 2}, "c": 5}


def test_changing_a_list_items_kind_raises_and_keeps_the_old_value(root):
    lst = root.open_child("lst", EagerListView)
    lst.store([1, {"z": 1}])
    rows = _rows_under(lst)
    with pytest.raises(ContainerTypeError):
        lst[0] = {"z": 2}
    with pytest.raises(ContainerTypeError):
        lst[1] = 5
    with pytest.raises(ContainerTypeError):
        lst.set_child_container_as(0, {"z": 2}, EagerDictView)
    assert _rows_under(lst) == rows
    assert lst.extract() == [1, {"z": 1}]
    del lst[1]
    lst.append(5)
    assert lst.extract() == [1, 5]


# ============================================================================
# CONTRACT 6: lazy navigation, one-read existence, None is a value (Q15)
# ============================================================================


@pytest.mark.parametrize("view_cls", NESTED_VIEWS)
def test_open_child_chain_touches_no_storage(costed, view_cls):
    storage, counts = costed

    def chain(root):
        a = root.open_child("a", view_cls)
        b = a.open_child("b", view_cls)
        c = b.open_child("c", EagerListView)
        return c

    assert sum(measure(storage, counts, chain).values()) == 0
    with storage.transaction() as tx:
        Navigator(storage).root(tx).open_child("a", view_cls)["b"] = {"c": [1]}
    assert sum(measure(storage, counts, chain).values()) == 0


@pytest.mark.parametrize("view_cls", DICT_VIEWS)
def test_exists_is_one_read(costed, view_cls):
    storage, counts = costed
    absent = measure(
        storage, counts, lambda root: root.open_child("d", view_cls).container.exists()
    )
    assert sum(absent[k] for k in POINT_OPS) == 1, absent
    assert absent["scan"] == 0, absent
    with storage.transaction() as tx:
        Navigator(storage).root(tx).open_child("d", view_cls)[_key(view_cls, 1)] = 1
    present = measure(
        storage, counts, lambda root: root.open_child("d", view_cls).container.exists()
    )
    assert sum(present[k] for k in POINT_OPS) == 1, present
    assert present["scan"] == 0, present


@pytest.mark.parametrize("view_cls", DICT_VIEWS)
def test_none_is_a_stored_value(storage, view_cls):
    k = _key(view_cls, 1)
    with storage.transaction() as tx:
        view = Navigator(storage).root(tx).open_child("d", view_cls)
        view[k] = None
        assert k in view
        assert view[k] is None
        assert list(view.items()) == [(k, None)]
        assert len(view) == 1
    with storage.snapshot() as snap:
        view = Navigator(storage).root(snap).open_child("d", view_cls)
        assert k in view
        assert view[k] is None
        assert view.get(k, "dflt") is None
        assert view.eager.extract() == {k: None}


def test_none_is_a_stored_list_item(storage):
    with storage.transaction() as tx:
        lst = Navigator(storage).root(tx).open_child("l", EagerListView)
        lst.store([None, 1])
        lst.append(None)
    with storage.snapshot() as snap:
        lst = Navigator(storage).root(snap).open_child("l", EagerListView)
        assert lst.extract() == [None, 1, None]
        assert lst[0] is None
        assert None in lst
        assert len(lst) == 3


# ============================================================================
# ACCESS: every dict-like view mimics dict, with EMPTY as the absent default
# ============================================================================

ACCESS_VIEWS = (
    *DICT_VIEWS,
    LazyIndexedDictView,
    LazyLogIndexedDictView,
    LazyFractionalIndexedDictView,
    LazyKh57View,
)
"""Every dict-like view and facet. A new dict view belongs here."""


@pytest.mark.parametrize(
    "view_cls", [v for v in ACCESS_VIEWS if v not in (EagerKh57View, LazyKh57View)]
)
def test_reversed_reads_keys_values_and_items_last_first(root, view_cls):
    """Every ordered dict view reverses like a dict: itself, its keys, values and items."""
    view = root.open_child("d", view_cls)
    view.store({"k2": 2, "k1": 1, "k3": 3})
    keys = list(view)
    assert list(reversed(view)) == keys[::-1]
    assert list(reversed(view.keys())) == keys[::-1]
    assert list(reversed(view.values())) == list(view.values())[::-1]
    assert list(reversed(view.items())) == list(view.items())[::-1]


@pytest.mark.parametrize("view_cls", ACCESS_VIEWS)
def test_access_mimics_dict_with_empty_as_the_absent_default(storage, view_cls):
    """The three access rules, on an absent view, in a transaction and in a snapshot.

    ``view[key]`` on an absent key raises KeyError. ``view.get(key)`` returns
    EMPTY, never None, and ``view.get(key, default)`` returns the given
    default. None is a stored value, never a marker for absent. ``pop``
    follows ``get``. A read never creates the view.
    """
    key, absent = _key(view_cls, 1), _key(view_cls, 2)

    def check_absent_key(view):
        with pytest.raises(KeyError):
            view[absent]
        assert view.get(absent) is EMPTY
        assert view.get(absent, "dflt") == "dflt"
        assert view.get(absent, None) is None
        assert _lacks(view, absent)

    def check_stored_none(view):
        assert key in view
        assert view[key] is None
        assert view.get(key) is None
        assert view.get(key, "dflt") is None

    with storage.transaction() as tx:
        view = Navigator(storage).root(tx).open_child("d", view_cls)
        check_absent_key(view)
        assert not view.container.exists()
        view[key] = None
        check_absent_key(view)
        check_stored_none(view)
    with storage.snapshot() as snap:
        view = Navigator(storage).root(snap).open_child("d", view_cls)
        check_absent_key(view)
        check_stored_none(view)
    if not hasattr(view_cls, "pop"):
        return
    with storage.transaction() as tx:
        view = Navigator(storage).root(tx).open_child("d", view_cls)
        with pytest.raises(KeyError):
            view.pop(absent)
        assert view.pop(absent, "dflt") == "dflt"
        assert view.pop(key) is None
        assert _lacks(view, key)


@pytest.mark.parametrize("view_cls", ACCESS_VIEWS)
def test_get_returns_the_default_only_for_an_absent_key(root, view_cls, monkeypatch):
    """``get`` is ``[]`` plus a KeyError catch: any other error propagates."""
    view = root.open_child("d", view_cls)

    def boom(self, address):
        raise RuntimeError("storage failed")

    monkeypatch.setattr(view_cls, "__getitem__", boom)
    with pytest.raises(RuntimeError, match="storage failed"):
        view.get(_key(view_cls, 1))


@pytest.mark.parametrize("view_cls", ACCESS_VIEWS)
def test_get_on_a_closed_storage_raises(storage, view_cls):
    """The real path: a read that fails below the view is not an absent key."""
    from virtuals.tkv.storage.exceptions import StorageClosedError

    key = _key(view_cls, 1)
    with storage.transaction() as tx:
        Navigator(storage).root(tx).open_child("d", view_cls)[key] = 1
    snap = storage.snapshot().__enter__()  # left open on purpose: close() ends it
    view = Navigator(storage).root(snap).open_child("d", view_cls)
    storage.close()
    with pytest.raises(StorageClosedError):
        view.get(key)


def test_access_views_cover_every_dict_view():
    """A view with ``get`` exported from ``_views`` is in ACCESS_VIEWS, or this fails."""
    from virtuals import _views

    found = {
        cls
        for name in _views.__all__
        if not name.endswith("Base")
        and isinstance(cls := getattr(_views, name), type)
        and hasattr(cls, "get")
    }
    assert found == set(ACCESS_VIEWS)


# ============================================================================
# KEY DOMAIN: a key the view cannot hold is absent on reads, rejected on writes
# ============================================================================

FOREIGN_KEYS = (None, 1.5, b"a", ("a",), [1])
"""Keys no dict view holds: not str or int (the last one is not even hashable)."""

KH57_FOREIGN_KEYS = (*FOREIGN_KEYS, "k", -1, 1 << 57)
"""Kh57 holds non-negative ints under 57 bits only."""


@pytest.mark.parametrize("view_cls", ACCESS_VIEWS)
def test_a_key_the_view_cannot_hold_is_absent_on_reads_and_rejected_on_writes(root, view_cls):
    """Dict holds no key of a type it was never given: KeyError, the default, False.

    A write under such a key raises TypeError (ValueError for an int outside
    the Kh57 domain) and leaves the view as it was.
    """
    view = root.open_child("d", view_cls)
    key = _key(view_cls, 1)
    view[key] = 1
    foreign = KH57_FOREIGN_KEYS if view_cls in (EagerKh57View, LazyKh57View) else FOREIGN_KEYS
    for bad in foreign:
        with pytest.raises(KeyError):
            view[bad]
        assert view.get(bad) is EMPTY
        assert view.get(bad, "dflt") == "dflt"
        assert _lacks(view, bad)
        with pytest.raises((TypeError, ValueError)):
            view[bad] = 2
        with pytest.raises(KeyError):
            del view[bad]
        if hasattr(view, "pop"):
            with pytest.raises(KeyError):
                view.pop(bad)
            assert view.pop(bad, "dflt") == "dflt"
    assert list(view) == [key]
    assert _read(view, key) == 1


@pytest.mark.parametrize("view_cls", (EagerKh57View, LazyKh57View))
def test_kh57_reads_follow_python_int_key_semantics(root, view_cls):
    """``1.0 == 1`` and ``True == 1`` name key 1; ``1.5`` names nothing. Writes take ints only."""
    view = root.open_child("k", view_cls)
    view[1] = "one"
    for alias in (1.0, True):
        assert alias in view
        assert view[alias] == "one"
        assert view.get(alias) == "one"
    assert _lacks(view, 1.5)
    assert view.get(1.5) is EMPTY
    with pytest.raises(KeyError):
        view[1.5]
    with pytest.raises(TypeError):
        view[1.0] = "x"
    view[True] = "uno"
    assert view[1] == "uno"
    del view[1.0]
    assert _lacks(view, 1)
    assert len(view) == 0


# ============================================================================
# EXISTENCE: a raising or no-op write never creates the view (contract 6, r3 F3 to F7)
# ============================================================================

ALL_CLEARABLE_VIEWS = (
    *DICT_VIEWS,
    EagerListView,
    LazyListView,
    FlatListView,
    SetView,
    ByteArrayView,
)


def _lacks(view, key):
    """``key not in view``; ``in`` on a never-written view raises, which also means not there."""
    try:
        return key not in view
    except ContainerNotFoundError:
        return not view.container.exists()


def _assert_absent(view):
    """Absent: no marker, and reads that need the container raise (a KeyError)."""
    assert not view.container.exists()
    for read in (len, bool, list):
        with pytest.raises(ContainerNotFoundError):
            read(view)


@pytest.mark.parametrize("view_cls", (EagerListView, LazyListView, ByteArrayView))
def test_delete_out_of_range_on_absent_sequence_does_not_create_it(root, view_cls):
    """``del v[0]`` on an absent list raises like ``pop()`` does and writes nothing."""
    view = root.open_child("absent", view_cls)
    with pytest.raises(IndexError):
        del view[0]
    _assert_absent(view)


@pytest.mark.parametrize("view_cls", (EagerListView, LazyListView, FlatListView, ByteArrayView))
def test_set_out_of_range_on_absent_sequence_does_not_create_it(root, view_cls):
    view = root.open_child("absent", view_cls)
    with pytest.raises(IndexError):
        view[0] = 1
    _assert_absent(view)


@pytest.mark.parametrize("view_cls", (EagerListView, LazyListView, ByteArrayView))
def test_remove_missing_on_absent_sequence_does_not_create_it(root, view_cls):
    view = root.open_child("absent", view_cls)
    with pytest.raises(ValueError):
        view.remove(1)
    _assert_absent(view)


def test_set_discard_and_remove_on_absent_set_do_not_create_it(root):
    view = root.open_child("absent", SetView)
    view.discard("x")
    _assert_absent(view)
    with pytest.raises(KeyError):
        view.remove("x")
    _assert_absent(view)
    assert _lacks(view, "x")


def test_set_discard_of_a_member_still_deletes_it(root):
    view = root.open_child("s", SetView)
    view.add("x")
    view.add("y")
    view.discard("x")
    view.remove("y")
    assert view.container.exists()
    assert len(view) == 0
    assert _lacks(view, "x")


@pytest.mark.parametrize("view_cls", (EagerListView, LazyListView, FlatListView, ByteArrayView))
def test_extend_with_nothing_on_absent_sequence_leaves_it_absent(root, view_cls):
    view = root.open_child("absent", view_cls)
    view.extend([])
    _assert_absent(view)
    view.extend([1, 2])
    assert view.container.exists()
    assert list(view) == [1, 2]


@pytest.mark.parametrize("view_cls", ALL_CLEARABLE_VIEWS)
def test_clear_on_absent_view_leaves_it_absent(root, view_cls):
    """clear() of nothing is not a write: no marker, no internal layout rows."""
    view = root.open_child("absent", view_cls)
    view.clear()
    _assert_absent(view)
    assert _lacks(root, "absent")


@pytest.mark.parametrize("view_cls", ALL_CLEARABLE_VIEWS)
def test_clear_on_existing_view_empties_it_and_keeps_it(root, view_cls):
    view = root.open_child("v", view_cls)
    if view_cls is SetView:
        view.add("x")
    elif view_cls in (EagerListView, LazyListView, FlatListView, ByteArrayView):
        view.append(1)
    else:
        view[_key(view_cls, 1)] = 1
    assert len(view) == 1
    view.clear()
    assert view.container.exists()
    assert len(view) == 0
    assert list(view) == []
    assert "v" in root


# ============================================================================
# MEMBERS AND SLOTS: r3 model audit M1 to M3
# ============================================================================


@pytest.mark.parametrize("member", [(1, 2), frozenset({1, "a"}), ("x", (2, 3)), None, b"raw"])
def test_set_hashable_member_round_trips_as_one_row(root, member):
    """A tuple or frozenset is a member like any other: it never becomes a child
    container, so the set stays iterable and extract() returns it as stored."""
    view = root.open_child("s", SetView)
    view.add("x")
    view.add(member)
    assert len(view) == 2
    assert member in view
    assert set(view) == {"x", member}
    assert view.extract() == {"x", member}
    view.discard(member)
    assert set(view) == {"x"}


@pytest.mark.parametrize("member", [[1, 2], {"a": 1}, {1, 2}, (1, [2])])
def test_set_unhashable_member_is_rejected_before_any_write(root, member):
    """What a Python set refuses, SetView refuses, and an absent set stays absent."""
    view = root.open_child("absent", SetView)
    with pytest.raises(TypeError, match="unhashable"):
        view.add(member)
    _assert_absent(view)
    with pytest.raises(TypeError, match="unhashable"):
        view.store(["ok", member])
    _assert_absent(view)
    view.add("ok")
    with pytest.raises(TypeError, match="unhashable"):
        view.store([member], replace=True)
    assert set(view) == {"ok"}


def test_set_store_keeps_compound_members_readable(root):
    view = root.open_child("s", SetView)
    view.store([(1, 2), "a", frozenset({3})])
    assert set(view) == {(1, 2), "a", frozenset({3})}
    assert len(view) == 3
    view.store([(9,)], replace=False)
    assert len(view) == 4
    assert (9,) in view


def test_frozenset_store_keeps_compound_members_readable(root):
    from virtuals._views import FrozenSetView

    view = root.open_child("fs", FrozenSetView)
    view.store([(1, 2), "a"])
    assert set(view) == {(1, 2), "a"}
    assert view.extract() == frozenset({(1, 2), "a"})
    with pytest.raises(TypeError, match="unhashable"):
        view.store(["b", [1]], replace=True)
    assert set(view) == {(1, 2), "a"}


@pytest.mark.parametrize("view_cls", (EagerListView, LazyListView))
@pytest.mark.parametrize("at", [0, 1, 2])
def test_list_insert_of_a_compound_over_a_primitive_slot_keeps_the_list_whole(root, view_cls, at):
    """insert shifts the tail and vacates the slot: the new value never meets the
    kind of the item that used to sit there. len, iteration and rows agree."""
    view = root.open_child("ls", view_cls)
    view.store([1, 2, 3])
    view.insert(at, {"z": 1})
    expected = [1, 2, 3]
    expected.insert(at, {"z": 1})
    assert len(view) == 4
    assert [v.extract() if hasattr(v, "extract") else v for v in view] == expected
    assert view.container.get_child_type(4).name == "NONE" or not view.container.exists_child(4)
    view.append(9)
    assert len(view) == 5
    assert [v.extract() if hasattr(v, "extract") else v for v in view] == [*expected, 9]


def test_list_insert_of_a_primitive_over_a_compound_slot_keeps_the_list_whole(root):
    view = root.open_child("ls", EagerListView)
    view.store([{"a": 1}, 2])
    view.insert(0, 9)
    view.insert(1, 8)
    assert len(view) == 4
    assert list(view) == [9, 8, {"a": 1}, 2]


@pytest.mark.parametrize("move", ["move_before", "move_after"])
def test_fractional_move_of_an_absent_id_onto_itself_raises(root, move):
    """Naming an absent id raises KeyError on every move, the self move too."""
    view = root.open_child("f", EagerFractionalIndexedDictView)
    view.append("a", 1)
    with pytest.raises(KeyError):
        getattr(view, move)("zz", "zz")
    assert list(view) == ["a"]
    getattr(view, move)("a", "a")  # present self move: a no-op
    assert list(view) == ["a"]


# ============================================================================
# PAGING: fractional view row bound and deleted cursor (contract 7, r3 F14)
# ============================================================================


@pytest.mark.parametrize("limit", [1, 5, 20])
def test_fractional_paging_reads_at_most_limit_plus_two_rows(costed, limit):
    storage, counts = costed
    with storage.transaction() as tx:
        view = Navigator(storage).root(tx).open_child("f", EagerFractionalIndexedDictView)
        for i in range(60):
            view.append(f"k{i:02d}", i)
    for cursor in ("k00", "k30", "k58"):
        cost = measure(
            storage,
            counts,
            lambda root, cursor=cursor: list(
                root.open_child("f", EagerFractionalIndexedDictView).keys(after=cursor, limit=limit)
            ),
        )
        assert cost["scan_rows"] <= limit + 2, (cursor, limit, cost)
    with storage.transaction() as tx:
        view = Navigator(storage).root(tx).open_child("f", EagerFractionalIndexedDictView)
        assert (
            list(view.keys(after="k30", limit=limit))
            == [f"k{i:02d}" for i in range(31, 31 + limit)][: 60 - 31]
        )


def test_fractional_paging_on_a_deleted_cursor_raises(root):
    """The cursor's position went with its row, so the view raises instead of guessing."""
    view = root.open_child("f", EagerFractionalIndexedDictView)
    for i in range(10):
        view.append(f"k{i}", i)
    assert list(view.keys(after="k4", limit=3)) == ["k5", "k6", "k7"]
    del view["k4"]
    with pytest.raises(KeyError):
        list(view.keys(after="k4", limit=3))
    assert list(view.keys(after="k3", limit=3)) == ["k5", "k6", "k7"]


# ============================================================================
# SET MEMBER KEYS: stable for ordered members, canonical for unordered (review S4 to S6)
# ============================================================================

LEGACY_MEMBER_KEYS = [
    # sha256(pickle.dumps(value, protocol=4))[:64], as every release up to d8ee856 wrote it
    (1, "018f5c4626b56e8489da7abb6c8b62331933c42c35d1342037a5242b8ed148f6"),
    (-7, "0643eb8bb54381dc608e300b72fc878501ea0736a7d2245b1f1d434956c439a8"),
    (2**70, "4636a968bc0033c904811e102c53babe8ab88a9f6e294566b61c40edc8f202be"),
    ("x", "e0c27ef51d001d2d87fcc4bd80d352bbee3f02aeb2d4afaefc41e065023bbb69"),
    ("", "8e2a8bd996c29b99c909664ee51eb5ee3b2358306a4269d5b69dd99075e6dd85"),
    ("ünï", "eaf0c6b552fe62468d7692dc69b6506fd0d6f08c756d0a72391001c6d96dbe17"),
    (b"raw", "2f74412bb456938aacd9f0939cd4907129b897503bcb84c397fc156d1d73b075"),
    (b"", "53b733953fb825c982984407c97cd225bf2a1be0613a8675da94eb8625acbcff"),
    ((1, "a", None), "623bee0d9b9105e157d831fbcfb7afb113ab91c4b2c9b85a5acae004f644f5a1"),
    (((1, 2), (3,)), "b5687a69aa6e1cd548c06265b40526d73f3aecad968fe18e2a48a609d97eba22"),
    (None, "9c298d589a2158eb513cb52191144518a2acab2cb0c04f1df14fca0f712fa4a1"),
    (3.5, "31ca616a2c9e6a40025ac55abe5c9c28d3f334b0bf8cd4c4d74e0dfd161dbc9e"),
    (True, "112bda3b495d867b6a98c899fac7c25eb60ca4b6e6fe5ec7ab9299f93e8274bc"),
]


@pytest.mark.parametrize(("member", "key"), LEGACY_MEMBER_KEYS, ids=repr)
def test_set_member_key_is_byte_identical_to_the_old_formula_for_ordered_members(member, key):
    """Keys written by d8ee856 code stay readable: int, str, bytes, tuple, None and friends."""
    from virtuals._views import FrozenSetView
    from virtuals._views._member_key import member_key

    assert member_key(member) == key
    assert SetView._make_key(None, member) == key
    assert FrozenSetView._make_key(None, member) == key


def test_set_member_key_of_an_unordered_member_ignores_build_order():
    from virtuals._views._member_key import member_key

    a = frozenset(["a", "b", "c", "d"])
    b = frozenset(["d", "c", "b", "a"])
    assert member_key(a) == member_key(b)
    assert member_key(("t", a, 1)) == member_key(("t", b, 1))
    assert member_key(frozenset({a, 1})) == member_key(frozenset({1, b}))
    assert member_key(a) != member_key(frozenset("abc"))
    assert member_key(frozenset()) != member_key(())


def test_set_legacy_members_in_storage_stay_members(root):
    """A row written under the old key is the same member today."""
    view = root.open_child("s", SetView)
    view.ensure_created()
    for member, key in LEGACY_MEMBER_KEYS:
        view._put_child_primitive(key, member)
    assert len(view) == len(LEGACY_MEMBER_KEYS)
    for member, _ in LEGACY_MEMBER_KEYS:
        assert member in view
        view.add(member)  # no duplicate
    assert len(view) == len(LEGACY_MEMBER_KEYS)
    view.discard(2**70)
    assert len(view) == len(LEGACY_MEMBER_KEYS) - 1


@pytest.mark.parametrize("backend_name", ("sqlite", "lmdb"))
def test_set_frozenset_members_match_across_spawned_processes(backend_name, tmp_path):
    """One process adds frozenset members, another (different hash seed) sees them and
    adds them again without a duplicate."""
    import multiprocessing as mp
    import os

    from tests.support import set_members

    ctx = mp.get_context("spawn")
    results = []
    for seed, fn in (("1", set_members.add_members), ("2", set_members.check_and_add_again)):
        os.environ["PYTHONHASHSEED"] = seed
        try:
            with ctx.Pool(1) as pool:
                results.append(pool.apply(fn, (backend_name, str(tmp_path))))
        finally:
            os.environ.pop("PYTHONHASHSEED", None)
    report = results[1]
    assert all(report["seen"].values()), report
    assert report["len"] == len(set_members.MEMBERS), report
    assert set(report["members"]) == set(set_members.MEMBERS), report


def test_set_equal_frozensets_built_in_different_orders_are_one_member(root):
    view = root.open_child("s", SetView)
    view.add(frozenset(["a", "b", "c", "d"]))
    view.add(frozenset(["d", "c", "b", "a"]))
    assert len(view) == 1
    assert frozenset("dcba") in view
    view.discard(frozenset("abcd"))
    assert len(view) == 0


@pytest.mark.parametrize("member", [lambda: 1, ("t", lambda: 1), threading.Lock()])
def test_set_unpicklable_member_is_rejected_before_any_write(root, member):
    """Hashable but not picklable: rejected like an unhashable one, nothing written."""
    from virtuals._views import FrozenSetView

    view = root.open_child("absent", SetView)
    with pytest.raises(Exception, match=r"pickle|Lock"):
        view.add(member)
    _assert_absent(view)
    with pytest.raises(Exception, match=r"pickle|Lock"):
        view.store(["a", member])
    _assert_absent(view)
    view.add("a")
    with pytest.raises(Exception, match=r"pickle|Lock"):
        view.store(["b", member], replace=False)
    assert set(view) == {"a"}
    fs = root.open_child("fs", FrozenSetView)
    with pytest.raises(Exception, match=r"pickle|Lock"):
        fs.store(["a", member])
    _assert_absent(fs)


def test_set_add_replaces_a_member_an_old_write_left_as_a_child_container(root):
    """Before members were opaque rows a tuple member became a child container and the
    set could not be iterated. add() of that member repairs it, the count stays right."""
    view = root.open_child("s", SetView)
    view.add("x")
    view._populate_child_container(view._make_key((1, 2)), {"a": 1})  # the old layout
    assert len(view) == 2
    assert (1, 2) in view
    with pytest.raises(ContainerTypeError):
        list(view)
    view.add((1, 2))
    assert len(view) == 2
    assert set(view) == {"x", (1, 2)}
    view.store([(1, 2), "y"], replace=False)
    assert set(view) == {"x", "y", (1, 2)}
    assert len(view) == 3


def test_kh57_backend_delete_on_absent_view_does_not_create_it(root):
    view = root.open_child("absent", EagerKh57View)
    view._delete_encoded(7)
    _assert_absent(view)
