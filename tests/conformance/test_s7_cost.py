"""Contract section 7: a point delete touches a constant number of keys."""

from __future__ import annotations

import pytest

from tests.conformance.helpers import CountingTxn
from virtuals.navigator import Navigator
from virtuals.tkv import StorageScanOptions
from virtuals.tkv.filter import PrefixFilter
from virtuals.tkv.types import EMPTY
from virtuals.views import EagerDictView


SIZES = (10, 1000)


def _storage_delete_cost(storage, n: int) -> int:
    with storage.transaction() as t:
        for i in range(n):
            t.put(("p", n, i), i)
    with storage.transaction() as tx:
        ct = CountingTxn(tx)
        ct.delete(("p", n, n // 2))
        touched = ct.touched()
    with storage.snapshot() as s:
        assert s.get(("p", n, n // 2)) is EMPTY
        left = list(
            s.scan(
                StorageScanOptions(start=("p", n), break_filter=PrefixFilter(prefix=("p", n)))
            ).keys()
        )
        assert len(left) == n - 1
    return touched


@pytest.mark.timeout(20)
def test_s7_storage_delete_is_point_delete(storage):
    """txn.delete of one key among many touches one key, at any sibling count."""
    costs = {n: _storage_delete_cost(storage, n) for n in SIZES}
    assert costs == dict.fromkeys(SIZES, 1), costs


def _view_delete_cost(storage, n: int) -> int:
    name = f"d{n}"
    with storage.transaction() as tx:
        view = Navigator(storage).root(tx).open_child(name, EagerDictView)
        view.store({f"k{i}": {"a": i, "b": [1, 2]} for i in range(n)})
    with storage.transaction() as tx:
        ct = CountingTxn(tx)
        view = Navigator(storage).root(ct).open_child(name, EagerDictView)
        del view["k5"]
        return ct.touched()


@pytest.mark.timeout(30)
def test_s7_view_dict_delete_is_point_delete(storage):
    """Deleting one key from a dict view touches the same number of keys at 10 and 1000 siblings."""
    costs = {n: _view_delete_cost(storage, n) for n in SIZES}
    assert len(set(costs.values())) == 1, f"keys touched by one delete, by dict size: {costs}"
