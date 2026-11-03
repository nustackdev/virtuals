"""RocksDB-specific behaviour beyond the shared StorageProtocol compliance suite.

Covers what compliance cannot: scans that outlive their transaction or the
storage (a use-after-free in rdbpy before 0.2.7), reverse scans with start
bounds, secondary instances, TransactionDB options and aborting on
BaseException. The kv contract cases live in ``test_rocksdb_contract.py``.
"""

from __future__ import annotations

import gc
import os
import subprocess
import sys
import textwrap
import weakref
from typing import TYPE_CHECKING, Any

import pytest

from tests._backends.storages import _write_slot_cases as cases
from virtuals._backends.storages.rocksdb import RocksDBStorage
from virtuals._backends.storages.rocksdb.snapshot import RocksDBSnapshot
from virtuals._backends.storages.rocksdb.transaction import RocksDBTransaction
from virtuals.codecs import BinaryCodec
from virtuals.tkv import StorageScanOptions
from virtuals.tkv.filter import PrefixFilter
from virtuals.tkv.storage import StorageClosedError, StorageError
from virtuals.tkv.types import EMPTY


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def storage(tmp_path: Path):
    s = RocksDBStorage(path=tmp_path / "db", codec=BinaryCodec(), busy_timeout=5.0)
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
        self.rows = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._it, name)

    def next_batch(self, n: int) -> list:
        rows = self._it.next_batch(n)
        self.rows += len(rows)
        return rows


class _CountingDB:
    def __init__(self, db: Any) -> None:
        self._db = db
        self.iterators: list[_CountingIterator] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._db, name)

    def iterator(self, **kwargs: Any) -> _CountingIterator:
        it = _CountingIterator(self._db.iterator(**kwargs))
        self.iterators.append(it)
        return it


def test_reverse_prefix_scan_seeks_instead_of_walking(storage: RocksDBStorage) -> None:
    _fill(storage, KEYS)
    counting = _CountingDB(storage._db)
    storage._db = counting
    try:
        options = StorageScanOptions(
            start=("a", 10), reverse=True, break_filter=PrefixFilter(prefix=("a",))
        )
        with storage.snapshot() as snap:
            keys = list(snap.scan(options).keys())
    finally:
        storage._db = counting._db
    assert keys == [("a", i) for i in range(10, -1, -1)]
    # 11 yielded keys from the seek down to the start of the keyspace; no
    # walk over the 100 keys after the start bound.
    assert counting.iterators[0].rows == 11


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


# =============================================================================
# Shared writer slot cases (every backend on the shared slot runs these)
# =============================================================================


@pytest.mark.timeout(10)
@pytest.mark.parametrize("ctx", ["transaction", "batch_write"])
def test_cancel_inside_block_releases_slot(storage: RocksDBStorage, ctx: str) -> None:
    cases.cancel_inside_block(storage, ctx)


@pytest.mark.timeout(10)
@pytest.mark.parametrize("ctx", ["transaction", "batch_write"])
def test_keyboard_interrupt_inside_block_releases_slot(storage: RocksDBStorage, ctx: str) -> None:
    cases.keyboard_interrupt_inside_block(storage, ctx)


@pytest.mark.timeout(10)
def test_cancel_while_reserving_holds_nothing(storage: RocksDBStorage) -> None:
    cases.cancel_while_reserving(storage)


@pytest.mark.timeout(10)
def test_cancel_between_reserve_and_begin_releases_slot(storage: RocksDBStorage) -> None:
    cases.cancel_between_reserve_and_begin(storage)


@pytest.mark.timeout(10)
def test_to_thread_nested_raises_fast(storage: RocksDBStorage) -> None:
    cases.to_thread_nested_raises_fast(storage)


@pytest.mark.timeout(10)
def test_asyncio_run_inside_sync_txn_is_nested(storage: RocksDBStorage) -> None:
    cases.asyncio_run_inside_sync_txn_is_nested(storage)


@pytest.mark.timeout(10)
def test_racing_ends_release_slot_once(storage: RocksDBStorage) -> None:
    tx = storage.begin_transaction()
    cases.racing_ends_release_once(storage, tx, lambda t: t._mark_closed(), "_holds_write_slot")


@pytest.mark.timeout(10)
def test_handoff_worker_is_not_nested(storage: RocksDBStorage) -> None:
    cases.handoff_worker_is_not_nested(storage)


@pytest.mark.timeout(10)
def test_handoff_then_writers_on_same_worker(storage: RocksDBStorage) -> None:
    cases.handoff_then_writers_on_same_worker(storage)


@pytest.mark.timeout(10)
def test_spawned_task_waits(storage: RocksDBStorage) -> None:
    cases.spawned_task_waits(storage)


@pytest.mark.timeout(10)
def test_reserved_slot_does_not_survive_close(storage: RocksDBStorage) -> None:
    cases.reserved_slot_does_not_survive_close(storage)


@pytest.mark.timeout(10)
def test_waiter_on_stale_slot_wakes_on_close(storage: RocksDBStorage) -> None:
    cases.waiter_on_stale_slot_wakes_on_close(storage)


@pytest.mark.timeout(10)
def test_writer_waiting_across_close_gets_closed_error(storage: RocksDBStorage) -> None:
    cases.writer_waiting_across_close_gets_closed_error(storage)


@pytest.mark.timeout(10)
def test_dropped_transaction_frees_slot(storage: RocksDBStorage) -> None:
    cases.dropped_transaction_frees_slot(storage)


@pytest.mark.timeout(10)
@pytest.mark.parametrize("ctx", ["transaction", "batch_write"])
def test_sync_begin_on_loop_thread_is_bounded(storage: RocksDBStorage, ctx: str) -> None:
    cases.sync_begin_on_loop_thread_is_bounded(storage, ctx)


@pytest.mark.timeout(10)
def test_context_end_after_slot_reset_touches_nothing(storage: RocksDBStorage) -> None:
    cases.context_end_after_slot_reset_touches_nothing(storage, storage.begin_transaction())


# =============================================================================
# Lifecycle and open edges (audit 03, R1 and R4 to R9)
# =============================================================================


def test_dropped_transaction_is_collected_and_frees_slot(storage: RocksDBStorage) -> None:
    """The active set is weak: a dropped transaction is collected, its finalizer frees the slot."""
    txn = storage.begin_transaction()
    txn.put(("k",), 1)
    ref = weakref.ref(txn)
    del txn
    gc.collect()
    assert ref() is None
    assert not storage._writer.lock.locked()
    assert not storage._active_transactions
    with storage.snapshot() as snap:
        assert snap.get(("k",)) is EMPTY  # rolled back natively


def test_begin_during_close_window_raises_closed_error(tmp_path: Path) -> None:
    """Between the native close and the state update, begin reports a closed storage."""
    s = RocksDBStorage(path=tmp_path / "db", codec=BinaryCodec())
    s.open()
    s._db.close()
    with pytest.raises(StorageClosedError):
        s.begin_transaction()
    assert not s._writer.lock.locked()  # the slot taken for the begin was given back
    with pytest.raises(StorageClosedError):
        s.begin_snapshot()
    s._db = None
    s._opened = False


@pytest.mark.parametrize("mode", ["read_only", "secondary"])
def test_read_only_open_of_missing_path_leaves_no_directory(tmp_path: Path, mode: str) -> None:
    """A read-only or secondary open of a missing path fails and leaves nothing behind.

    (A read-write open with ``create_if_missing=False`` fails too, but
    rocksdb itself drops its LOCK and LOG files there first; a later open
    finds no data in them, so nothing is adopted.)
    """
    path = tmp_path / "nope"
    kwargs: dict = {"read_only": True}
    if mode == "secondary":
        kwargs["secondary_path"] = tmp_path / "secondary"
    s = RocksDBStorage(path=path, codec=BinaryCodec(), **kwargs)
    with pytest.raises(StorageError):
        s.open()
    assert not path.exists()
    # A read-write open afterwards creates a fresh database as usual.
    with RocksDBStorage(path=path, codec=BinaryCodec()) as rw, rw.transaction() as txn:
        txn.put(("k",), 1)


def test_snapshot_close_closes_its_scan_iterators(storage: RocksDBStorage) -> None:
    """A suspended scan's native iterator is closed with the snapshot, not with the generator."""
    _fill(storage, KEYS)
    snap = storage.begin_snapshot()
    keys = snap.scan(StorageScanOptions()).keys()
    assert next(keys) == ("a", 0)
    (handle,) = list(snap._scans)
    assert not handle.iterator.closed
    snap.close()
    assert handle.iterator.closed
    with pytest.raises(StorageClosedError):
        next(keys)  # the generator is still there and still refuses
    assert snap._snapshot is None


@pytest.mark.parametrize("end", ["commit", "abort", "close"])
def test_transaction_end_closes_its_scan_iterators(tmp_path: Path, end: str) -> None:
    s = RocksDBStorage(path=tmp_path / "db", codec=BinaryCodec())
    s.open()
    _fill(s, KEYS)
    txn = s.begin_transaction()
    items = txn.scan(StorageScanOptions()).items()
    next(items)
    (handle,) = list(txn._scans)
    if end == "close":
        s.close()
    else:
        getattr(txn, end)()
    assert handle.iterator.closed
    with pytest.raises(StorageClosedError):
        next(items)
    s.close()


def test_finished_scan_is_forgotten_by_its_context(storage: RocksDBStorage) -> None:
    _fill(storage, KEYS)
    with storage.snapshot() as snap:
        list(snap.scan(StorageScanOptions(limit=3)).keys())
        gc.collect()
        assert not snap._scans  # the handle died with the generator


class _Publisher:
    def __init__(self) -> None:
        self.batches: list[set] = []

    def notify(self, keys) -> None:
        self.batches.append(set(keys))


def test_delete_of_absent_key_notifies_from_txn_and_batch_alike(tmp_path: Path) -> None:
    """Contract section 5 allows a spurious wake; the two write paths agree on it."""
    pub = _Publisher()
    with RocksDBStorage(path=tmp_path / "db", codec=BinaryCodec(), publisher=pub) as s:
        with s.transaction() as txn:
            txn.delete(("absent", 1))
        with s.batch_write() as batch:
            batch.delete(("absent", 2))
    assert pub.batches == [{("absent", 1)}, {("absent", 2)}]


def test_exists_reads_the_key_only(storage: RocksDBStorage, monkeypatch) -> None:
    """exists never fetches the value, and sees a transaction's own pending writes."""
    _fill(storage, KEYS)

    def boom(self, encoded_key):
        raise AssertionError("exists fetched the value")

    monkeypatch.setattr(RocksDBSnapshot, "_get_raw", boom)
    monkeypatch.setattr(RocksDBTransaction, "_get_raw", boom)
    with storage.snapshot() as snap:
        assert snap.exists(("a", 0))
        assert not snap.exists(("a", 0, "below"))  # a longer key with this prefix is not it
        assert not snap.exists(("zzz",))
    txn = storage.begin_transaction()
    try:
        txn.put(("new",), 1)
        txn.delete(("a", 1))
        assert txn.exists(("new",))
        assert not txn.exists(("a", 1))
        assert txn.exists(("a", 2))
    finally:
        txn.abort()


def test_second_open_explains_single_process(storage: RocksDBStorage) -> None:
    other = RocksDBStorage(path=storage.path, codec=BinaryCodec())
    with pytest.raises(StorageError, match="single-process"):
        other.open()
    code = textwrap.dedent(
        f"""
        from virtuals._backends.storages.rocksdb import RocksDBStorage
        from virtuals.codecs import BinaryCodec
        from virtuals.tkv.storage import StorageError
        try:
            RocksDBStorage(path={str(storage.path)!r}, codec=BinaryCodec()).open()
        except StorageError as e:
            print("single-process" in str(e))
        """
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0 and proc.stdout.strip() == "True", proc.stdout + proc.stderr
