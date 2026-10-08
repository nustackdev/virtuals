"""RocksDB-specific behaviour beyond the shared StorageProtocol compliance suite.

Covers what compliance cannot: scans that outlive their transaction or the
storage (a use-after-free in rdbpy before 0.2.7), reverse scans with start
bounds, secondary instances, TransactionDB options and aborting on
BaseException.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from typing import TYPE_CHECKING, Any

import pytest

from virtuals._backends.storages.rocksdb import RocksDBStorage
from virtuals.codecs import BinaryCodec
from virtuals.tkv import StorageScanOptions
from virtuals.tkv.filter import PrefixFilter
from virtuals.tkv.storage import StorageClosedError


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def storage(tmp_path: Path):
    s = RocksDBStorage(path=tmp_path / "db", codec=BinaryCodec())
    s.open()
    yield s
    s.close()


def _fill(storage: RocksDBStorage, keys: list[tuple]) -> None:
    with storage.transaction() as txn:
        for k in keys:
            txn.put(k, repr(k))


KEYS = [(p, i) for p in ("a", "b", "c") for i in range(50)]


# =============================================================================
# Scans outliving their context
# =============================================================================


@pytest.mark.parametrize("end", ["commit", "abort"])
def test_scan_resumed_after_context_ends_raises(storage: RocksDBStorage, end: str) -> None:
    _fill(storage, KEYS)
    txn = storage.begin_transaction()
    keys = txn.scan(StorageScanOptions()).keys()
    assert next(keys) == ("a", 0)
    getattr(txn, end)()
    with pytest.raises(StorageClosedError):
        next(keys)


def test_scan_resumed_after_snapshot_close_raises(storage: RocksDBStorage) -> None:
    _fill(storage, KEYS)
    snap = storage.begin_snapshot()
    items = snap.scan(StorageScanOptions(reverse=True)).items()
    assert next(items)[0] == ("c", 49)
    snap.close()
    with pytest.raises(StorageClosedError):
        next(items)


def test_scan_resumed_after_storage_close_raises(tmp_path: Path) -> None:
    s = RocksDBStorage(path=tmp_path / "db", codec=BinaryCodec())
    s.open()
    _fill(s, KEYS)
    txn = s.begin_transaction()
    values = txn.scan(StorageScanOptions()).values()
    next(values)
    s.close()
    with pytest.raises(StorageClosedError):
        next(values)


def test_suspended_scan_in_traceback_survives_close_and_gc(tmp_path: Path) -> None:
    # The original crash: a scan raises mid-iteration, the traceback keeps the
    # generator (and its rdbpy iterator), the storage closes, gc frees it all.
    # Runs in a subprocess with freed memory poisoned so a regression shows up
    # as a crash of the child, not of the test run.
    code = textwrap.dedent(
        f"""
        import gc

        from virtuals._backends.storages.rocksdb import RocksDBStorage
        from virtuals.codecs import BinaryCodec
        from virtuals.tkv import StorageScanOptions

        kept = []
        for i in range(10):
            s = RocksDBStorage(path={str(tmp_path)!r} + "/db%d" % i, codec=BinaryCodec())
            s.open()
            with s.transaction() as txn:
                for k in range(500):
                    txn.put(("k", k), k)
            try:
                with s.transaction() as txn:
                    scan = txn.scan(StorageScanOptions()).items()
                    next(scan)
                    raise AttributeError("boom")
            except AttributeError as e:
                e.self_ref = e
                kept.append(e)
            s.close()
            del s, txn, scan
            kept.clear()
            gc.collect()
        print("DONE")
        """
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    env["PYTHONMALLOC"] = "debug"
    env["MallocScribble"] = "1"
    env["MALLOC_PERTURB_"] = "85"
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "DONE" in proc.stdout


# =============================================================================
# Reverse scans
# =============================================================================


@pytest.mark.parametrize(
    "start", [None, ("b",), ("b", 10), ("b", 10, "x"), ("a",), ("zzz",), ("0",)]
)
def test_reverse_scan_with_start_matches_forward(storage: RocksDBStorage, start: Any) -> None:
    _fill(storage, KEYS)
    codec = storage.codec
    with storage.snapshot() as snap:
        forward = list(snap.scan(StorageScanOptions()).keys())
        if start is None:
            expected = forward[::-1]
        else:
            bound = codec.encode_key(start)
            expected = [k for k in forward[::-1] if codec.encode_key(k) <= bound]
        keys = list(snap.scan(StorageScanOptions(start=start, reverse=True)).keys())
        items = list(snap.scan(StorageScanOptions(start=start, reverse=True)).items())
    assert keys == expected
    assert [k for k, _ in items] == expected
    assert all(v == repr(k) for k, v in items)


def test_reverse_prefix_scan_from_upper_bound(storage: RocksDBStorage) -> None:
    _fill(storage, KEYS)
    prefix = ("b",)
    options = StorageScanOptions(
        start_encoded=storage.codec.upper_bound_of_prefix(prefix),
        reverse=True,
        break_filter=PrefixFilter(prefix=prefix),
    )
    with storage.snapshot() as snap:
        keys = list(snap.scan(options).keys())
    assert keys == [("b", i) for i in reversed(range(50))]


class _CountingIterator:
    def __init__(self, it: Any) -> None:
        self._it = it
        self.steps = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._it, name)

    def skip(self) -> None:
        self.steps += 1
        self._it.skip()

    def skip_back(self) -> None:
        self.steps += 1
        self._it.skip_back()


class _CountingTxn:
    def __init__(self, txn: Any) -> None:
        self._txn = txn
        self.iterators: list[_CountingIterator] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._txn, name)

    def iterkeys(self) -> _CountingIterator:
        it = _CountingIterator(self._txn.iterkeys())
        self.iterators.append(it)
        return it


def test_reverse_prefix_scan_seeks_instead_of_walking(storage: RocksDBStorage) -> None:
    _fill(storage, KEYS)
    snap = storage.begin_snapshot()
    counting = _CountingTxn(snap._rdbpy_txn)
    snap._rdbpy_txn = counting
    options = StorageScanOptions(
        start=("a", 10), reverse=True, break_filter=PrefixFilter(prefix=("a",))
    )
    keys = list(snap.scan(options).keys())
    snap._rdbpy_txn = counting._txn
    snap.close()
    assert keys == [("a", i) for i in range(10, -1, -1)]
    # 11 yielded keys, then one step off the start of the keyspace; no walk
    # over the 100 keys after the start bound.
    assert counting.iterators[0].steps == 11


# =============================================================================
# Secondary instances
# =============================================================================


def test_secondary_close_and_snapshot_close(tmp_path: Path) -> None:
    primary = RocksDBStorage(path=tmp_path / "db", codec=BinaryCodec())
    primary.open()
    _fill(primary, KEYS)
    secondary = RocksDBStorage(
        path=tmp_path / "db",
        codec=BinaryCodec(),
        read_only=True,
        secondary_path=tmp_path / "secondary",
    )
    secondary.open()
    try:
        snap = secondary.begin_snapshot()
        assert snap.get(("a", 1)) == repr(("a", 1))
        snap.close()
        assert snap.is_closed
        with pytest.raises(StorageClosedError):
            snap.get(("a", 1))
        closing = secondary.begin_snapshot()
        early = closing.scan(StorageScanOptions()).keys()
        next(early)
        closing.close()
        with pytest.raises(StorageClosedError):
            next(early)
        live = secondary.begin_snapshot()
        scan = live.scan(StorageScanOptions()).keys()
        next(scan)
    finally:
        secondary.close()
        primary.close()
    with pytest.raises(StorageClosedError):
        next(scan)


# =============================================================================
# Options and abort paths
# =============================================================================


def test_txn_db_options_open(tmp_path: Path) -> None:
    s = RocksDBStorage(
        path=tmp_path / "db",
        codec=BinaryCodec(),
        txn_db_options={"transaction_lock_timeout": 50},
    )
    s.open()
    try:
        assert s._db is not None
        _fill(s, KEYS[:3])
    finally:
        s.close()


def test_transaction_aborts_on_base_exception(storage: RocksDBStorage) -> None:
    with pytest.raises(KeyboardInterrupt), storage.transaction() as txn:
        txn.put(("k",), 1)
        raise KeyboardInterrupt
    assert txn._aborted
    assert txn not in storage._active_transactions
    # The row lock is released: another transaction writes the key at once.
    with storage.transaction() as other:
        other.put(("k",), 2)
    with storage.snapshot() as snap:
        assert snap.get(("k",)) == 2


def test_write_batch_after_storage_close_raises(tmp_path: Path) -> None:
    s = RocksDBStorage(path=tmp_path / "db", codec=BinaryCodec())
    s.open()
    batch = s.begin_write_batch()
    batch.put(("k",), 1)
    s._active_write_batches.discard(batch)  # simulate a batch close() did not track
    s.close()
    with pytest.raises(StorageClosedError):
        batch.write()
