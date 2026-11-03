"""Contract section 7: a point delete touches a constant number of keys.

Keys are counted inside the backend by a counting codec, not by wrapping
the transaction: a backend that walks the store to delete one key decodes
every key it walks. A timing check covers walks over raw bytes.
"""

from __future__ import annotations

import gc
import statistics
import time

import pytest

from tests.conformance.helpers import CountingCodec
from virtuals.navigator import Navigator
from virtuals.tkv import StorageScanOptions
from virtuals.tkv.filter import PrefixFilter
from virtuals.tkv.types import EMPTY
from virtuals.views import EagerDictView


SIZES = (10, 1000)
POINT_DELETE_KEYS = 1  # one key encoded, none decoded


@pytest.fixture
def counted(backend, tmp_path):
    codec = CountingCodec()
    st = backend.make(tmp_path, codec=codec)
    st.open()
    yield st, codec
    st.close()


def _fill(storage, prefix: tuple, n: int) -> None:
    with storage.transaction() as t:
        for i in range(n):
            t.put((*prefix, i), i)


def _storage_delete_cost(storage, codec: CountingCodec, n: int) -> int:
    _fill(storage, ("p", n), n)
    with storage.transaction() as tx:
        codec.reset()
        tx.delete(("p", n, n // 2))
        touched = codec.touched()
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
def test_s7_storage_delete_is_point_delete(counted):
    """txn.delete of one key among many handles one key inside the backend, at any sibling count."""
    storage, codec = counted
    costs = {n: _storage_delete_cost(storage, codec, n) for n in SIZES}
    assert costs == dict.fromkeys(SIZES, POINT_DELETE_KEYS), f"keys handled by one delete: {costs}"


TIMED = (100, 10_000)
TIMED_DELETES = 31
TIME_RATIO = 10.0  # a walk over siblings is about 100x slower at 10k than at 100


def _median_delete_time(storage, n: int) -> float:
    prefix = ("t", n)
    _fill(storage, prefix, n)
    times = []
    gc.disable()
    try:
        with storage.transaction() as tx:
            for i in range(TIMED_DELETES):
                key = (*prefix, i * (n // TIMED_DELETES))
                t0 = time.perf_counter()
                tx.delete(key)
                times.append(time.perf_counter() - t0)
    finally:
        gc.enable()
    return statistics.median(times)


@pytest.mark.timeout(30)
def test_s7_storage_delete_time_flat(storage):
    """Median time of one txn.delete grows less than TIME_RATIO from 100 to 10k siblings."""
    small, large = (_median_delete_time(storage, n) for n in TIMED)
    ratio = large / small
    assert ratio < TIME_RATIO, (
        f"delete median {small * 1e6:.0f}us at {TIMED[0]}, {large * 1e6:.0f}us at {TIMED[1]}"
    )


def _view_delete_cost(storage, codec: CountingCodec, n: int) -> int:
    name = f"d{n}"
    with storage.transaction() as tx:
        view = Navigator(storage).root(tx).open_child(name, EagerDictView)
        view.store({f"k{i}": {"a": i, "b": [1, 2]} for i in range(n)})
    with storage.transaction() as tx:
        view = Navigator(storage).root(tx).open_child(name, EagerDictView)
        codec.reset()
        del view["k5"]
        return codec.touched()


@pytest.mark.timeout(30)
def test_s7_view_dict_delete_is_point_delete(counted):
    """Deleting one key from a dict view handles as many keys at 10 as at 1000 siblings."""
    storage, codec = counted
    costs = {n: _view_delete_cost(storage, codec, n) for n in SIZES}
    assert len(set(costs.values())) == 1, f"keys handled by one dict delete, by size: {costs}"
