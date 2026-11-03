"""Hardening of every dict view against storage, on every backend.

What the contract asks of a dict view (existence, EMPTY, cost) and the
edges that tend to break at the storage boundary: odd keys, deep nesting,
large values, erase then len. Each case runs on mem, sqlite, lmdb and
rocksdb; op counts run on the seeking backends only.
"""

from __future__ import annotations

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
    EagerDictView,
    EagerFractionalIndexedDictView,
    EagerIndexedDictView,
    EagerKh57View,
    EagerListView,
    EagerLogIndexedDictView,
    FlatDictView,
    LazyDictView,
    LightDictView,
)
from virtuals.container import ContainerTypeError
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

SILENT_DELETE_VIEWS = (EagerDictView, LazyDictView, FlatDictView, LightDictView)
"""Views whose delete of a missing key is a no-op; the others raise KeyError."""


def _key(view_cls, i):
    """A key of the type the view takes (Kh57 takes ints only)."""
    return i if view_cls is EagerKh57View else f"k{i}"


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
    if view_cls in SILENT_DELETE_VIEWS:
        del view[key]
    else:
        with pytest.raises(KeyError):
            del view[key]
    assert not view.container.exists()
    assert len(view) == 0
    assert key not in view
    assert list(view) == []


@pytest.mark.parametrize("view_cls", (EagerDictView, FlatDictView, LightDictView))
def test_delete_on_absent_nested_view_leaves_the_chain_absent(root, view_cls):
    """``del a.b["k"]`` on an absent chain stamps neither ``a`` nor ``b``."""
    outer = root.open_child("a", EagerDictView)
    inner = outer.open_child("b", view_cls)
    del inner["k"]
    assert not inner.container.exists()
    assert not outer.container.exists()
    assert "a" not in root


@pytest.mark.parametrize("view_cls", SILENT_DELETE_VIEWS)
def test_delete_missing_key_on_existing_view_is_silent(root, view_cls):
    view = root.open_child("d", view_cls)
    view["a"] = 1
    del view["zz"]
    assert list(view) == ["a"]
    assert len(view) == 1


@pytest.mark.parametrize("view_cls", (EagerDictView, FlatDictView))
def test_delete_missing_key_cost_is_constant_and_writes_nothing(costed, view_cls):
    storage, counts = costed
    for present in (False, True):
        if present:
            with storage.transaction() as tx:
                Navigator(storage).root(tx).open_child("d", view_cls).store({"a": 1})
        cost = measure(
            storage, counts, lambda root: root.open_child("d", view_cls).__delitem__("zz")
        )
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
    assert _key(view_cls, 2) not in view
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
    assert _key(view_cls, 2) not in view
    assert _key(view_cls, 3) not in view


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
        assert "e" not in view
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
        assert keys[1] not in view
        assert bool(view)
    with storage.transaction() as tx:
        view = Navigator(storage).root(tx).open_child("d", view_cls)
        assert len(view) == 2
        assert sorted(view, key=str) == sorted([keys[0], keys[2]], key=str)
        assert keys[1] not in view
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
        assert "b" not in view
        assert list(view.items()) == [("a", 1)]
    with storage.transaction() as tx:
        view = Navigator(storage).root(tx).open_child("d", view_cls)
        assert len(view) == 1
        assert view.get("b", "dflt") == "dflt"
        # Nothing of the old subtree is left under the key.
        child = view.open_child("b", EagerDictView)
        assert not child.container.exists()
        assert len(child) == 0


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
            assert key not in view
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
    assert 1 not in view
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
def test_bool_on_absent_flat_view_is_false_and_writes_nothing(root, view_cls):
    view = root.open_child("absent", view_cls)
    assert not view
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
    assert "a" not in view
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
        assert "a" not in view


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
        assert "y" not in view
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
        assert bad not in view
        assert len(view) == 1
        assert list(view) == [keep]
        assert not view.open_child(bad, EagerDictView).container.exists()
    # a lazy iterable is materialized and checked before the first write
    with pytest.raises(ValueError, match="EMPTY"):
        view.set_child_container_as(bad, (x for x in [1, {"y": EMPTY}]), EagerListView)
    assert _rows_under(view) == rows
    assert bad not in view
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
    assert not view.container.exists()
    assert _rows_under(view) == []
    assert len(view) == 0
    assert list(view) == []


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
    assert "x" not in view
    assert len(view) == 2
    assert list(view) == ["a", "b"]
    assert view.get("x") is None
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
    eager = getattr(view, "eager", view)
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
        assert getattr(view, "eager", view).extract() == {k: None}


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
