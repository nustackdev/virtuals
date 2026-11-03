"""Storage cost of DictView and FlatDictView operations, on real backends.

Point ops (set, delete, contains) must touch a constant number of storage
keys whatever the dict size, and scans (``len``, ``keys()``) must cost
O(children), not O(descendants). Counted at the storage boundary: point
ops on the transaction class, rows visited by a scan through the codec
(every row a backend looks at is decoded exactly once). mem is left out
on purpose: its scan sorts the whole store by design.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import pytest

from virtuals._views import EagerDictView, FlatDictView
from virtuals.codecs import BinaryCodec
from virtuals.navigator import Navigator


if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from pathlib import Path

    from virtuals.tkv.storage import StorageProtocol


POINT_OPS = ("get", "exists", "put", "delete")
COUNTED_OPS = (*POINT_OPS, "scan")
SIZES = (10, 1000)


# =============================================================================
# COUNTING
# =============================================================================


class _CountingCodec:
    """Codec proxy that counts rows a scan visits (one ``decode_key`` each)."""

    def __init__(self, inner: object, counts: Counter) -> None:
        self._inner = inner
        self._counts = counts

    def decode_key(self, encoded: object) -> object:
        self._counts["scan_rows"] += 1
        return self._inner.decode_key(encoded)  # type: ignore[attr-defined]

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)


@pytest.fixture(params=["sqlite", "rocksdb"])
def storage(request: pytest.FixtureRequest, tmp_path: Path) -> Generator[StorageProtocol]:
    if request.param == "sqlite":
        from virtuals.storages.sqlite import SQLiteStorage

        s = SQLiteStorage(path=tmp_path / "kv.sqlite", codec=BinaryCodec())
    else:
        from virtuals.storages.rocksdb import RocksDBStorage

        s = RocksDBStorage(path=tmp_path / "db", codec=BinaryCodec())
    s.open()
    try:
        yield s
    finally:
        s.close()


@pytest.fixture
def counts(storage: StorageProtocol, monkeypatch: pytest.MonkeyPatch) -> Counter:
    """Live counter of storage ops: ``get``/``exists``/``put``/``delete``/``scan`` and ``scan_rows``."""
    counter: Counter = Counter()

    with storage.transaction() as tx:
        tx_cls = type(tx)
    for name in COUNTED_OPS:
        original = getattr(tx_cls, name)

        def counted(self, *args, _name=name, _original=original, **kwargs):
            counter[_name] += 1
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(tx_cls, name, counted)

    monkeypatch.setattr(storage, "_codec", _CountingCodec(storage._codec, counter))  # type: ignore[attr-defined]
    return counter


def _measure(
    storage: StorageProtocol, counts: Counter, op: Callable[[EagerDictView], object]
) -> Counter:
    """Run ``op`` on the root view in its own transaction, return the ops it cost."""
    counts.clear()
    with storage.transaction() as tx:
        op(Navigator(storage).root(tx))
    return Counter(counts)


def _fill(storage: StorageProtocol, view_cls: type, n: int, value: Callable[[int], object]) -> None:
    with storage.transaction() as tx:
        d = Navigator(storage).root(tx).open_child("d", view_cls)
        d.store({f"k{i}": value(i) for i in range(n)})


def _prim(i: int) -> int:
    return i


def _compound(width: int) -> Callable[[int], object]:
    """A child whose subtree holds ``width`` descendant rows."""
    return lambda i: {f"f{j}": j for j in range(width)}


# =============================================================================
# POINT OPS: CONSTANT IN DICT SIZE
# =============================================================================


def _set_new(view_cls: type) -> Callable[[EagerDictView], object]:
    return lambda root: root.open_child("d", view_cls).__setitem__("new", 1)


def _set_existing(view_cls: type) -> Callable[[EagerDictView], object]:
    return lambda root: root.open_child("d", view_cls).__setitem__("k5", 2)


def _delete(view_cls: type) -> Callable[[EagerDictView], object]:
    return lambda root: root.open_child("d", view_cls).__delitem__("k5")


def _contains(view_cls: type) -> Callable[[EagerDictView], object]:
    return lambda root: (
        "k5" in root.open_child("d", view_cls),
        "zz" in root.open_child("d", view_cls),
    )


@pytest.mark.parametrize("view_cls", [EagerDictView, FlatDictView])
@pytest.mark.parametrize(
    "op_name",
    ["set_new", "set_existing", "delete", "contains"],
)
def test_point_op_cost_is_constant(storage, counts, view_cls, op_name):
    """Set, delete and contains touch the same keys on a 10-key and a 1000-key dict."""
    op = {
        "set_new": _set_new,
        "set_existing": _set_existing,
        "delete": _delete,
        "contains": _contains,
    }[op_name](view_cls)

    costs = {}
    for n in SIZES:
        _fill(storage, view_cls, n, _prim)
        costs[n] = _measure(storage, counts, op)

    assert costs[SIZES[0]] == costs[SIZES[-1]], costs
    assert costs[SIZES[0]]["scan_rows"] == 0, costs
    assert sum(costs[SIZES[0]][name] for name in POINT_OPS) <= 8, costs


def test_dict_delete_of_compound_child_touches_only_its_subtree(storage, counts):
    """Deleting a compound entry costs its own subtree, not the dict's size."""
    costs = {}
    for n in SIZES:
        _fill(storage, EagerDictView, n, _compound(3))
        costs[n] = _measure(storage, counts, _delete(EagerDictView))

    assert costs[SIZES[0]] == costs[SIZES[-1]], costs
    # marker + 3 fields deleted, nothing else scanned
    assert costs[SIZES[0]]["delete"] == 4, costs
    assert costs[SIZES[0]]["scan_rows"] <= 6, costs


def test_dict_set_compound_cost_is_constant(storage, counts):
    costs = {}
    for n in SIZES:
        _fill(storage, EagerDictView, n, _prim)
        costs[n] = _measure(
            storage,
            counts,
            lambda root: root.open_child("d", EagerDictView).__setitem__("new", {"a": 1}),
        )

    assert costs[SIZES[0]] == costs[SIZES[-1]], costs
    # a child created by the write has nothing to clear: no scan at all
    assert costs[SIZES[0]]["scan"] == 0, costs
    assert costs[SIZES[0]]["scan_rows"] == 0, costs


def test_dict_set_compound_over_existing_replaces_it(storage, counts):
    """Overwriting an existing compound entry clears its old subtree first."""
    _fill(storage, EagerDictView, 10, _compound(3))
    cost = _measure(
        storage,
        counts,
        lambda root: root.open_child("d", EagerDictView).__setitem__("k5", {"z": 1}),
    )
    with storage.transaction() as tx:
        assert Navigator(storage).root(tx).open_child("d", EagerDictView)["k5"] == {"z": 1}
    # one child scan over the old entry: marker + 3 fields + seek peeks, nothing wider
    assert cost["scan"] == 1, cost
    assert cost["scan_rows"] <= 6, cost


# =============================================================================
# SCANS: O(CHILDREN), NOT O(DESCENDANTS)
# =============================================================================


def _len(view_cls: type) -> Callable[[EagerDictView], object]:
    return lambda root: len(root.open_child("d", view_cls))


def _keys(view_cls: type) -> Callable[[EagerDictView], object]:
    # a comprehension, not list(): list() first asks KeysView for len(), a second scan
    return lambda root: [k for k in root.open_child("d", view_cls).keys()]  # noqa: C416


def _keys_reversed(view_cls: type) -> Callable[[EagerDictView], object]:
    return lambda root: list(root.open_child("d", view_cls).container.iter_child_keys(reverse=True))


@pytest.mark.parametrize("op_name", ["len", "keys", "keys_reversed"])
def test_dict_scan_cost_scales_with_children_not_descendants(storage, counts, op_name):
    """10 children with 50 descendants each cost the same scan as with 500 each."""
    op = {"len": _len, "keys": _keys, "keys_reversed": _keys_reversed}[op_name](EagerDictView)
    children = 10

    costs = {}
    for width in (50, 500):
        _fill(storage, EagerDictView, children, _compound(width))
        costs[width] = _measure(storage, counts, op)

    assert costs[50] == costs[500], costs
    # one row per child, one peek into each subtree before the seek, the marker
    assert costs[500]["scan_rows"] <= 2 * children + 2, costs
    for name in POINT_OPS:
        assert costs[500][name] <= 2, costs


def test_flat_dict_scan_cost_is_linear_in_keys(storage, counts):
    costs = {}
    for n in SIZES:
        _fill(storage, FlatDictView, n, _prim)
        costs[n] = _measure(storage, counts, _len(FlatDictView))

    for n in SIZES:
        assert n <= costs[n]["scan_rows"] <= n + 2, costs
