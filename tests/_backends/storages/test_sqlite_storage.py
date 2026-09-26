"""SQLite-specific behaviour beyond the shared StorageProtocol compliance suite.

Covers what compliance cannot: byte ordering against LMDB, paged scans past
a page boundary, snapshot isolation, the writer lock across threads and
processes (including a writer killed mid-transaction), read-only opens and
reopening after close.
"""

from __future__ import annotations

import random
import subprocess
import sys
import textwrap
import threading
import time
from typing import TYPE_CHECKING

import pytest

from virtuals._backends.storages.sqlite import SQLiteStorage
from virtuals.codecs import BinaryCodec
from virtuals.tkv import StorageScanOptions
from virtuals.tkv.filter import LengthFilter, PrefixFilter
from virtuals.tkv.storage import StorageError, StorageLockTimeoutError
from virtuals.tkv.types import EMPTY


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "kv.sqlite"


@pytest.fixture
def storage(db_path: Path):
    s = SQLiteStorage(path=db_path, codec=BinaryCodec(), busy_timeout=5.0)
    s.open()
    yield s
    s.close()


def _keys(n_top: int = 20, n_mid: int = 12, n_leaf: int = 12) -> list[tuple]:
    """A tree of mixed int/str keys, with containers at every level."""
    rng = random.Random(57)  # noqa: S311
    keys: set[tuple] = set()
    for i in range(n_top):
        top = rng.choice([i, -i - 1, f"t{i}", f"t{i}" * 3, "zz" * (i + 1), 2**40 + i])
        keys.add((top,))
        for j in range(n_mid):
            mid = rng.choice([j, f"m{j}", -j * 1000, "a" * (j + 1)])
            keys.add((top, mid))
            for k in range(n_leaf):
                keys.add((top, mid, rng.choice([k, f"l{k}", -(2**33) - k])))
    return sorted(keys, key=str)


def _fill(storage, keys: list[tuple]) -> None:
    with storage.transaction() as tx:
        for i, k in enumerate(keys):
            tx.put(k, f"v{i}".encode())


def _scan(storage, options: StorageScanOptions) -> list:
    with storage.snapshot() as snap:
        return list(snap.scan(options).items())


# =========================================================================
# Ordering and scans
# =========================================================================


def test_schema_and_pragmas(storage: SQLiteStorage, db_path: Path) -> None:
    conn = storage._acquire_connection()
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1  # NORMAL
        sql = conn.execute("SELECT sql FROM sqlite_master WHERE name = 'kv'").fetchone()[0]
        assert "WITHOUT ROWID" in sql
    finally:
        storage._release_connection(conn)

    full = SQLiteStorage(path=db_path, codec=BinaryCodec(), synchronous="FULL")
    with full:
        conn = full._acquire_connection()
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 2
        full._release_connection(conn)


def test_scan_order_is_encoded_byte_order(storage: SQLiteStorage) -> None:
    keys = _keys()
    _fill(storage, keys)
    codec = storage.codec

    got = [k for k, _ in _scan(storage, StorageScanOptions())]
    assert got == sorted(keys, key=codec.encode_key)

    got_rev = [k for k, _ in _scan(storage, StorageScanOptions(reverse=True))]
    assert got_rev == got[::-1]


def _option_grid(keys: list[tuple], codec: BinaryCodec) -> list[StorageScanOptions]:
    rng = random.Random(3)  # noqa: S311
    grid: list[StorageScanOptions] = [StorageScanOptions(), StorageScanOptions(reverse=True)]
    containers = [k for k in keys if len(k) < 3]
    for site in rng.sample(containers, 25):
        prefix = PrefixFilter(prefix=site)
        child = prefix & LengthFilter(length=len(site) + 1)
        grid += [
            StorageScanOptions(start=site, break_filter=prefix),
            StorageScanOptions(start=site, break_filter=prefix, filter=child),
            StorageScanOptions(start=site, break_filter=prefix, filter=child, limit=3),
            StorageScanOptions(start=site, break_filter=prefix, limit=1),
            StorageScanOptions(
                start_encoded=codec.upper_bound_of_prefix(site),
                reverse=True,
                break_filter=prefix,
                filter=child,
            ),
            StorageScanOptions(
                start_encoded=codec.upper_bound_of_prefix(site), reverse=True, break_filter=prefix
            ),
            StorageScanOptions(start=site, reverse=True, limit=40),
            StorageScanOptions(start=site, limit=100),
            StorageScanOptions(break_filter=prefix),
            StorageScanOptions(reverse=True, break_filter=prefix),
        ]
    return grid


def test_scans_match_lmdb(storage: SQLiteStorage, tmp_path: Path) -> None:
    """Every scan shape the views use returns exactly what LMDB returns."""
    pytest.importorskip("lmdb")
    from virtuals._backends.storages.lmdb import LMDBStorage

    keys = _keys()
    assert len(keys) > 2 * 1024  # crosses several scan pages
    _fill(storage, keys)
    with LMDBStorage(path=tmp_path / "ref.lmdb", codec=BinaryCodec(), map_size=2**26) as ref:
        _fill(ref, keys)
        for options in _option_grid(keys, storage.codec):
            assert _scan(storage, options) == _scan(ref, options), options


def test_keys_and_values_iterators(storage: SQLiteStorage) -> None:
    _fill(storage, [("a", i) for i in range(100)])
    with storage.snapshot() as snap:
        opts = StorageScanOptions(start=("a",), break_filter=PrefixFilter(prefix=("a",)))
        assert list(snap.scan(opts).keys()) == [("a", i) for i in range(100)]
        assert list(snap.scan(opts).values()) == [f"v{i}".encode() for i in range(100)]


def test_delete_while_iterating(storage: SQLiteStorage) -> None:
    """Paged scans hold no statement open, so writing mid-scan is safe."""
    _fill(storage, [("x", i) for i in range(3000)] + [("y", 0)])
    opts = StorageScanOptions(start=("x",), break_filter=PrefixFilter(prefix=("x",)))
    with storage.transaction() as tx:
        n = 0
        for key in tx.scan(opts).keys():
            tx.delete(key)
            n += 1
        assert n == 3000
        assert list(tx.scan(opts).keys()) == []
    with storage.snapshot() as snap:
        assert snap.get(("y", 0)) == b"v3000"
        assert snap.get(("x", 5)) is EMPTY


# =========================================================================
# Snapshots
# =========================================================================


def test_snapshot_isolation(storage: SQLiteStorage) -> None:
    with storage.transaction() as tx:
        tx.put(("k",), b"old")

    snap = storage.begin_snapshot()  # pinned now, before any read
    with storage.transaction() as tx:
        tx.put(("k",), b"new")
        tx.put(("k2",), b"added")

    assert snap.get(("k",)) == b"old"
    assert snap.get(("k2",)) is EMPTY
    assert [k for k, _ in snap.scan(StorageScanOptions()).items()] == [("k",)]
    snap.close()

    with storage.snapshot() as fresh:
        assert fresh.get(("k",)) == b"new"


def test_snapshot_does_not_wait_for_writer(storage: SQLiteStorage) -> None:
    """A held write lock, and a writer queued behind it, never delay snapshots."""
    tx = storage.begin_transaction()
    tx.put(("k",), b"pending")

    queued = threading.Thread(target=lambda: storage.begin_transaction().commit())
    queued.start()
    time.sleep(0.05)  # queued writer is now waiting for the lock

    t0 = time.monotonic()
    with storage.snapshot() as snap:
        assert snap.get(("k",)) is EMPTY
    assert time.monotonic() - t0 < 0.5

    tx.commit()
    queued.join(5)
    assert not queued.is_alive()


# =========================================================================
# Writers: threads
# =========================================================================


def test_same_thread_second_writer_fails_fast(storage: SQLiteStorage) -> None:
    tx = storage.begin_transaction()
    t0 = time.monotonic()
    with pytest.raises(StorageLockTimeoutError):
        storage.begin_transaction()
    assert time.monotonic() - t0 < 0.5
    tx.abort()
    storage.begin_transaction().commit()  # slot released by the abort


def test_threads_serialize_writers(storage: SQLiteStorage) -> None:
    n_threads, rounds = 8, 50

    def work() -> None:
        for _ in range(rounds):
            with storage.transaction() as tx:
                cur = tx.get(("n",))
                tx.put(("n",), str(int(cur) + 1 if cur is not EMPTY else 1).encode())

    threads = [threading.Thread(target=work) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    with storage.snapshot() as snap:
        assert int(snap.get(("n",))) == n_threads * rounds


def test_transaction_can_finish_on_another_thread(storage: SQLiteStorage) -> None:
    tx = storage.begin_transaction()
    tx.put(("k",), b"v")
    t = threading.Thread(target=tx.commit)
    t.start()
    t.join(5)
    with storage.snapshot() as snap:
        assert snap.get(("k",)) == b"v"
    storage.begin_transaction().abort()  # slot released across threads


def test_lock_timeout_between_threads(db_path: Path) -> None:
    with SQLiteStorage(path=db_path, codec=BinaryCodec(), busy_timeout=0.1) as s:
        tx = s.begin_transaction()
        err: list[BaseException] = []

        def other() -> None:
            try:
                s.begin_transaction()
            except BaseException as e:
                err.append(e)

        t = threading.Thread(target=other)
        t.start()
        t.join(5)
        assert err and isinstance(err[0], StorageLockTimeoutError)
        tx.abort()


# =========================================================================
# Write batches
# =========================================================================


def test_write_batch_holds_no_lock_until_write(storage: SQLiteStorage) -> None:
    with storage.transaction() as tx:
        tx.put(("gone",), b"x")
    batch = storage.begin_write_batch()
    batch.put(("a",), b"1")
    batch.put(("a",), b"2")  # last write wins
    batch.delete(("gone",))

    with storage.transaction() as tx:  # not blocked by the open batch
        tx.put(("b",), b"3")

    batch.write()
    with storage.snapshot() as snap:
        assert snap.get(("a",)) == b"2"
        assert snap.get(("b",)) == b"3"
        assert snap.get(("gone",)) is EMPTY


# =========================================================================
# Lifecycle and read-only
# =========================================================================


def test_close_aborts_open_contexts_and_reopens(db_path: Path) -> None:
    s = SQLiteStorage(path=db_path, codec=BinaryCodec())
    s.open()
    with s.transaction() as tx:
        tx.put(("kept",), b"1")
    tx = s.begin_transaction()
    tx.put(("lost",), b"1")
    snap = s.begin_snapshot()
    s.close()
    assert tx.is_closed and snap.is_closed

    s.open()
    with s.snapshot() as snap2:
        assert snap2.get(("kept",)) == b"1"
        assert snap2.get(("lost",)) is EMPTY
    s.begin_transaction().commit()  # writer slot was released by close
    s.close()


def test_read_only(db_path: Path) -> None:
    with pytest.raises(StorageError):
        SQLiteStorage(path=db_path, codec=BinaryCodec(), read_only=True).open()

    rw = SQLiteStorage(path=db_path, codec=BinaryCodec())
    rw.open()
    with rw.transaction() as tx:
        tx.put(("k",), b"1")

    with SQLiteStorage(path=db_path, codec=BinaryCodec(), read_only=True) as ro:
        with ro.snapshot() as snap:
            assert snap.get(("k",)) == b"1"
        with pytest.raises(StorageError):
            ro.begin_transaction()
        with pytest.raises(StorageError):
            ro.begin_write_batch()

        with rw.transaction() as tx:  # a live writer next to the reader
            tx.put(("k",), b"2")
        with ro.snapshot() as snap:
            assert snap.get(("k",)) == b"2"
    rw.close()

    # Reopening read-only once no writer is left (WAL checkpointed away).
    with SQLiteStorage(path=db_path, codec=BinaryCodec(), read_only=True) as ro:
        with ro.snapshot() as snap:
            assert snap.get(("k",)) == b"2"


# =========================================================================
# Writers: processes
# =========================================================================


_HOLDER = textwrap.dedent(
    """
    import sys, time
    from virtuals._backends.storages.sqlite import SQLiteStorage
    from virtuals.codecs import BinaryCodec

    s = SQLiteStorage(path=sys.argv[1], codec=BinaryCodec())
    s.open()
    tx = s.begin_transaction()
    tx.put(("uncommitted",), b"x")
    print("ready", flush=True)
    time.sleep(60)
    """
)


def test_killed_writer_releases_lock(db_path: Path) -> None:
    """A writer killed mid-transaction leaves no stuck lock and no partial write."""
    with SQLiteStorage(path=db_path, codec=BinaryCodec(), busy_timeout=0.2) as s:
        proc = subprocess.Popen(  # noqa: S603
            [sys.executable, "-c", _HOLDER, str(db_path)],
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            assert proc.stdout is not None
            assert proc.stdout.readline().strip() == "ready"

            with pytest.raises(StorageLockTimeoutError):
                s.begin_transaction()
            with s.snapshot() as snap:  # readers unaffected by the other writer
                assert snap.get(("uncommitted",)) is EMPTY
        finally:
            proc.kill()
            proc.wait(10)

        t0 = time.monotonic()
        with s.transaction() as tx:
            assert tx.get(("uncommitted",)) is EMPTY
            tx.put(("after",), b"1")
        assert time.monotonic() - t0 < 1.0
