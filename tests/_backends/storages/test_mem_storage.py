"""In-memory storage: isolation, snapshots, the writer slot, scans and notifications.

Isolation cases are ported from the txncheck audit script (lost update,
write skew, insert race, torn read, repeatable read) with small iteration
counts. Scans are checked against a sorted() reference and against the
SQLite adapter over the same option grid the views use.
"""

from __future__ import annotations

import asyncio
import random
import threading
import time
from bisect import bisect_left, bisect_right

import pytest

from virtuals._backends.storages.mem import InMemoryStorage
from virtuals._backends.storages.mem.storage import InMemoryStorage as _Base
from virtuals._backends.storages.sqlite import SQLiteStorage
from virtuals.codecs import BinaryCodec, NoOpCodec, TextCodec
from virtuals.tkv import StorageScanOptions
from virtuals.tkv.filter import LengthFilter, PrefixFilter
from virtuals.tkv.storage import (
    StorageClosedError,
    StorageError,
    StorageLockTimeoutError,
    StorageNestedTransactionError,
    StorageTransactionConflictError,
    StorageTransactionError,
)
from virtuals.tkv.types import EMPTY


RETRYABLE = (StorageTransactionConflictError, StorageLockTimeoutError)

K = ("c",)
X, Y = ("x",), ("y",)


@pytest.fixture
def storage():
    s = InMemoryStorage(codec=NoOpCodec(), busy_timeout=5.0)
    s.open()
    yield s
    s.close()


def _put(storage, key, value) -> None:
    with storage.transaction() as tx:
        tx.put(key, value)


def _get(storage, key):
    with storage.snapshot() as snap:
        return snap.get(key)


def _retry(fn, attempts: int = 50):
    for i in range(attempts):
        try:
            return fn()
        except RETRYABLE:
            time.sleep(0.0005 * (i + 1))
    raise AssertionError("gave up retrying")


def _in_thread(fn, timeout: float = 5.0):
    """Run ``fn`` on a thread, join it, return its result (or raise its error)."""
    box: dict = {}

    def run() -> None:
        try:
            box["result"] = fn()
        except BaseException as e:
            box["error"] = e

    t = threading.Thread(target=run)
    t.start()
    t.join(timeout)
    assert not t.is_alive(), "thread did not finish"
    if "error" in box:
        raise box["error"]
    return box.get("result")


# =========================================================================
# Isolation (ported from audits/scripts/txncheck/storage_tests.py)
# =========================================================================


def test_lost_update_threads(storage: InMemoryStorage) -> None:
    n, iters = 4, 40
    _put(storage, K, 0)

    def inc() -> None:
        with storage.transaction() as t:
            v = t.get(K)
            time.sleep(0.0002)
            t.put(K, v + 1)

    def worker() -> None:
        for _ in range(iters):
            _retry(inc)

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert _get(storage, K) == n * iters


def test_write_skew_on_disjoint_keys(storage: InMemoryStorage) -> None:
    """Invariant: at most one of x, y is on. Both read before either writes."""
    barrier = threading.Barrier(2)
    for _ in range(5):
        with storage.transaction() as t:
            t.put(X, False)
            t.put(Y, False)

        def claim(mine, other) -> None:
            def body() -> None:
                with storage.transaction() as t:
                    o = t.get(other)
                    try:
                        barrier.wait(timeout=0.05)
                    except threading.BrokenBarrierError:
                        pass
                    if not o:
                        t.put(mine, True)

            _retry(body)

        a = threading.Thread(target=claim, args=(X, Y))
        b = threading.Thread(target=claim, args=(Y, X))
        a.start()
        b.start()
        a.join()
        b.join()
        barrier.reset()
        with storage.snapshot() as s:
            assert not (s.get(X) and s.get(Y))


def test_insert_race_same_key(storage: InMemoryStorage) -> None:
    """Create only if absent: exactly one of two racing creators wins."""
    barrier = threading.Barrier(2)
    for r in range(5):
        key = ("slot", r)
        won: list[str] = []

        def create(me, key=key, won=won) -> None:
            def body() -> bool:
                with storage.transaction() as t:
                    present = t.get(key)
                    try:
                        barrier.wait(timeout=0.05)
                    except threading.BrokenBarrierError:
                        pass
                    if present is EMPTY:
                        t.put(key, me)
                        return True
                    return False

            if _retry(body):
                won.append(me)

        a = threading.Thread(target=create, args=("a",))
        b = threading.Thread(target=create, args=("b",))
        a.start()
        b.start()
        a.join()
        b.join()
        barrier.reset()
        assert len(won) == 1


def test_scan_based_write_skew(storage: InMemoryStorage) -> None:
    """Two transactions each count a range, then insert if it is empty: one insert."""
    prefix = ("q",)
    opts = StorageScanOptions(start=prefix, break_filter=PrefixFilter(prefix=prefix))
    barrier = threading.Barrier(2)

    def claim(me) -> None:
        def body() -> None:
            with storage.transaction() as t:
                n = sum(1 for _ in t.scan(opts).keys())
                try:
                    barrier.wait(timeout=0.05)
                except threading.BrokenBarrierError:
                    pass
                if n == 0:
                    t.put((*prefix, me), True)

        _retry(body)

    a = threading.Thread(target=claim, args=("a",))
    b = threading.Thread(target=claim, args=("b",))
    a.start()
    b.start()
    a.join()
    b.join()
    with storage.snapshot() as s:
        assert len(list(s.scan(opts).keys())) == 1


def test_read_your_own_writes(storage: InMemoryStorage) -> None:
    _put(storage, ("a", 1), "committed")
    _put(storage, ("a", 3), "gone")
    opts = StorageScanOptions(start=("a",), break_filter=PrefixFilter(prefix=("a",)))
    with storage.transaction() as t:
        t.put(("a", 2), "pending")
        t.put(("a", 1), "overwritten")
        t.delete(("a", 3))
        assert t.get(("a", 2)) == "pending"
        assert t.get(("a", 1)) == "overwritten"
        assert t.get(("a", 3)) is EMPTY
        assert t.exists(("a", 2)) and not t.exists(("a", 3))
        assert t.multiget([("a", 1), ("a", 2), ("a", 3)]) == {
            ("a", 1): "overwritten",
            ("a", 2): "pending",
        }
        assert list(t.scan(opts).items()) == [(("a", 1), "overwritten"), (("a", 2), "pending")]
        assert [k for k, _ in t.scan(StorageScanOptions(reverse=True)).items()] == [
            ("a", 2),
            ("a", 1),
        ]
        # Writes after the first scan are seen by the next one.
        t.put(("a", 0), "late")
        assert list(t.scan(opts).keys()) == [("a", 0), ("a", 1), ("a", 2)]
        # Outside the transaction nothing has moved yet.
        with storage.snapshot() as s:
            assert s.get(("a", 2)) is EMPTY
            assert s.get(("a", 3)) == "gone"
    with storage.snapshot() as s:
        assert list(s.scan(opts).keys()) == [("a", 0), ("a", 1), ("a", 2)]


def test_atomic_commit_never_torn(storage: InMemoryStorage) -> None:
    """x + y == 0 holds in every snapshot and every transaction, under a racing writer."""
    with storage.transaction() as t:
        t.put(X, 0)
        t.put(Y, 0)
    stop = threading.Event()
    commits = [0]

    def writer() -> None:
        i = 0
        while not stop.is_set():
            i += 1
            with storage.transaction() as t:
                t.put(X, i)
                t.put(Y, -i)
            commits[0] += 1

    w = threading.Thread(target=writer)
    w.start()
    try:
        end = time.monotonic() + 0.3
        reads = 0
        while time.monotonic() < end:
            with storage.snapshot() as s:
                a = s.get(X)
                time.sleep(0.0002)
                b = s.get(Y)
            assert a + b == 0
            with storage.transaction() as t:
                a = t.get(X)
                time.sleep(0.0002)
                b = t.get(Y)
            assert a + b == 0
            reads += 1
    finally:
        stop.set()
        w.join()
    assert reads > 10 and commits[0] > 10


def test_repeatable_snapshot(storage: InMemoryStorage) -> None:
    _put(storage, X, 1)
    with storage.snapshot() as s:
        assert s.get(X) == 1
        _in_thread(lambda: _put(storage, X, 2))
        assert s.get(X) == 1
        assert _get(storage, X) == 2
    assert _get(storage, X) == 2


def test_snapshot_scan_is_point_in_time(storage: InMemoryStorage) -> None:
    keys = [("k", i) for i in range(10)]
    with storage.transaction() as t:
        for i, k in enumerate(keys):
            t.put(k, i)
    with storage.snapshot() as s:
        _in_thread(lambda: _mutate(storage))
        assert list(s.scan(StorageScanOptions()).items()) == [(k, i) for i, k in enumerate(keys)]
        assert [k for k, _ in s.scan(StorageScanOptions(reverse=True)).items()] == keys[::-1]
        assert s.get(("k", 3)) == 3  # deleted since, still visible here
        assert s.get(("k", 10)) is EMPTY  # inserted since, invisible here
        assert s.get(("k", 5)) == 5  # overwritten since
        with storage.snapshot() as fresh:
            assert fresh.get(("k", 3)) is EMPTY
            assert fresh.get(("k", 10)) == "new"
            assert fresh.get(("k", 5)) == "changed"
    # Both snapshots closed: nothing is retained.
    assert storage._store._history == {}
    assert not storage._store._undo_log


def _mutate(storage) -> None:
    with storage.transaction() as t:
        t.delete(("k", 3))
        t.put(("k", 10), "new")
        t.put(("k", 5), "changed")


def test_nested_snapshots_keep_their_own_versions(storage: InMemoryStorage) -> None:
    _put(storage, X, 0)
    s0 = storage.begin_snapshot()
    _put(storage, X, 1)
    s1 = storage.begin_snapshot()
    _put(storage, X, 2)
    s2 = storage.begin_snapshot()
    _put(storage, X, 3)
    assert (s0.get(X), s1.get(X), s2.get(X)) == (0, 1, 2)
    s1.close()
    assert (s0.get(X), s2.get(X)) == (0, 2)
    s0.close()
    assert s2.get(X) == 2
    assert len(storage._store._history[storage.codec.encode_key(X)]) == 1
    s2.close()
    assert storage._store._history == {}


def test_nested_transaction_raises_distinct_error(storage: InMemoryStorage) -> None:
    with storage.transaction() as outer:
        outer.put(X, 1)
        with pytest.raises(StorageNestedTransactionError):
            storage.begin_transaction()
        with pytest.raises(StorageNestedTransactionError):
            with storage.batch_write() as b:
                b.put(Y, 1)
    assert _get(storage, X) == 1  # the outer transaction was not disturbed

    async def main() -> None:
        slot = await storage.areserve_write_slot()
        try:
            with pytest.raises(StorageNestedTransactionError):
                await storage.areserve_write_slot()
            tx = storage.begin_transaction(write_slot=slot)
            with pytest.raises(StorageNestedTransactionError):
                storage.begin_transaction()
            tx.abort()
        finally:
            slot.release()

    asyncio.run(main())


def test_retryable_classes() -> None:
    assert not issubclass(StorageNestedTransactionError, RETRYABLE)
    assert issubclass(StorageNestedTransactionError, StorageTransactionError)
    assert issubclass(StorageLockTimeoutError, StorageTransactionError)


def test_lock_timeout_between_threads_is_retryable() -> None:
    with InMemoryStorage(codec=NoOpCodec(), busy_timeout=0.05) as s:
        tx = s.begin_transaction()
        t0 = time.monotonic()
        with pytest.raises(RETRYABLE):
            _in_thread(s.begin_transaction)
        assert 0.04 < time.monotonic() - t0 < 2.0
        tx.abort()
        _in_thread(lambda: s.begin_transaction().abort())


def test_process_model_declared(storage: InMemoryStorage) -> None:
    assert storage.process_model == "single"


# =========================================================================
# Writer slot: threads and coroutines
# =========================================================================


def test_transaction_can_finish_on_another_thread(storage: InMemoryStorage) -> None:
    tx = storage.begin_transaction()
    tx.put(X, 1)
    _in_thread(tx.commit)
    assert _get(storage, X) == 1
    storage.begin_transaction().abort()  # slot is free again


async def _increment(storage: InMemoryStorage, key: tuple, pause: float) -> None:
    slot = await storage.areserve_write_slot()
    try:
        tx = storage.begin_transaction(write_slot=slot)
        try:
            cur = tx.get(key)
            await asyncio.sleep(pause)
            tx.put(key, (cur if cur is not EMPTY else 0) + 1)
        except BaseException:
            tx.abort()
            raise
        tx.commit()
    finally:
        slot.release()


def test_coroutines_increment_without_stalling_the_loop(storage: InMemoryStorage) -> None:
    n = 16

    async def main() -> int:
        ticks = 0
        stop = asyncio.Event()

        async def ticker() -> None:
            nonlocal ticks
            while not stop.is_set():
                ticks += 1
                await asyncio.sleep(0.001)

        tick_task = asyncio.create_task(ticker())
        t0 = time.monotonic()
        await asyncio.gather(*(_increment(storage, ("n",), 0.005) for _ in range(n)))
        elapsed = time.monotonic() - t0
        stop.set()
        await tick_task
        assert elapsed < 2.0
        return ticks

    ticks = asyncio.run(main())
    assert _get(storage, ("n",)) == n
    assert ticks >= 20  # the loop kept turning while writers waited


def test_sync_begin_behind_loop_peer_fails_fast(storage: InMemoryStorage) -> None:
    async def main() -> None:
        slot = await storage.areserve_write_slot()
        tx = storage.begin_transaction(write_slot=slot)

        async def peer() -> None:
            t0 = time.monotonic()
            with pytest.raises(StorageLockTimeoutError, match="areserve_write_slot"):
                storage.begin_transaction()
            assert time.monotonic() - t0 < 0.5

        await asyncio.create_task(peer())
        tx.abort()
        slot.release()

    asyncio.run(main())


def test_async_waiter_times_out_without_blocking_loop() -> None:
    with InMemoryStorage(codec=NoOpCodec(), busy_timeout=0.1) as s:

        async def main() -> int:
            held = await s.areserve_write_slot()
            ticks = 0
            stop = asyncio.Event()

            async def ticker() -> None:
                nonlocal ticks
                while not stop.is_set():
                    ticks += 1
                    await asyncio.sleep(0.005)

            tick_task = asyncio.create_task(ticker())
            t0 = time.monotonic()

            async def waiter() -> None:
                await s.areserve_write_slot()

            with pytest.raises(StorageLockTimeoutError, match="Timed out"):
                await asyncio.create_task(waiter())
            elapsed = time.monotonic() - t0
            stop.set()
            await tick_task
            held.release()
            assert 0.08 < elapsed < 1.0
            return ticks

        assert asyncio.run(main()) >= 5


def test_unused_slot_is_released(storage: InMemoryStorage) -> None:
    async def main() -> None:
        slot = await storage.areserve_write_slot()
        assert slot.held
        slot.release()
        assert not slot.held
        slot.release()  # idempotent
        (await storage.areserve_write_slot()).release()

    asyncio.run(main())
    storage.begin_transaction().abort()


def test_coroutines_and_threads_share_the_slot(storage: InMemoryStorage) -> None:
    rounds = 10

    def thread_work() -> None:
        for _ in range(rounds):
            with storage.transaction() as tx:
                cur = tx.get(("n",))
                tx.put(("n",), (cur if cur is not EMPTY else 0) + 1)

    async def main() -> None:
        threads = [threading.Thread(target=thread_work) for _ in range(3)]
        for t in threads:
            t.start()
        await asyncio.gather(*(_increment(storage, ("n",), 0.0) for _ in range(3 * rounds)))
        for t in threads:
            await asyncio.get_running_loop().run_in_executor(None, t.join)

    asyncio.run(main())
    assert _get(storage, ("n",)) == 6 * rounds


# =========================================================================
# Write batch and lifecycle
# =========================================================================


def test_write_batch_holds_no_lock_until_write(storage: InMemoryStorage) -> None:
    batch = storage.begin_write_batch()
    batch.put(X, 1)
    batch.delete(("absent",))
    tx = storage.begin_transaction()  # not blocked by the open batch
    tx.put(Y, 2)
    tx.commit()
    with storage.snapshot() as s:
        assert s.get(X) is EMPTY
    batch.write()
    with storage.snapshot() as s:
        assert s.get(X) == 1 and s.get(Y) == 2
    with pytest.raises(StorageClosedError):
        batch.write()


def test_batch_write_waits_for_writer(storage: InMemoryStorage) -> None:
    tx = storage.begin_transaction()
    tx.put(X, "txn")
    batch = storage.begin_write_batch()
    batch.put(X, "batch")
    t = threading.Thread(target=batch.write)
    t.start()
    time.sleep(0.02)
    assert t.is_alive()  # waiting for the slot
    tx.commit()
    t.join(2)
    assert _get(storage, X) == "batch"


def test_double_commit_and_abort_semantics(storage: InMemoryStorage) -> None:
    tx = storage.begin_transaction()
    tx.commit()
    with pytest.raises(StorageClosedError):
        tx.commit()
    tx.abort()  # no-op once closed
    tx = storage.begin_transaction()
    tx.put(X, 1)
    tx.abort()
    with pytest.raises(StorageClosedError):
        tx.put(X, 2)
    assert _get(storage, X) is EMPTY


def test_close_aborts_open_contexts_and_reopens_empty() -> None:
    s = InMemoryStorage(codec=NoOpCodec())
    s.open()
    _put(s, X, 1)
    tx = s.begin_transaction()
    tx.put(Y, 1)
    snap = s.begin_snapshot()
    s.close()
    assert tx.is_closed and snap.is_closed
    with pytest.raises(StorageClosedError):
        s.begin_snapshot()
    s.open()
    assert _get(s, X) is EMPTY
    s.begin_transaction().abort()  # slot was released by close
    s.close()


def test_read_only_refuses_writes() -> None:
    with InMemoryStorage(codec=NoOpCodec(), read_only=True) as s:
        with pytest.raises(StorageError):
            s.begin_transaction()
        with pytest.raises(StorageError):
            s.begin_write_batch()
        s.begin_snapshot().close()


def test_values_stored_by_reference(storage: InMemoryStorage) -> None:
    value = [1, 2]
    _put(storage, X, value)
    value.append(3)
    assert _get(storage, X) == [1, 2, 3]


def test_nustd_style_subclass_with_late_init() -> None:
    class Fabric(_Base):
        def __init__(self) -> None:
            self.tag = "mine"

        def setup(self) -> None:
            _Base.__init__(self, codec=NoOpCodec())
            self.open()

    f = Fabric()
    f.setup()
    assert f.tag == "mine"
    _put(f, X, 1)
    assert _get(f, X) == 1
    f.close()


class _Publisher:
    def __init__(self) -> None:
        self.batches: list[set] = []

    def notify(self, keys) -> None:
        self.batches.append(set(keys))


def test_notifies_after_commit_only_for_real_changes() -> None:
    pub = _Publisher()
    with InMemoryStorage(codec=NoOpCodec(), publisher=pub) as s:
        with s.transaction() as t:
            t.put(X, 1)
            t.delete(("never",))
            assert pub.batches == []  # nothing before commit
        assert pub.batches == [{X}]
        with s.batch_write() as b:
            b.delete(X)
            b.put(Y, 1)
        assert pub.batches[-1] == {X, Y}
        tx = s.begin_transaction()
        tx.put(("z",), 1)
        tx.abort()
        with s.snapshot() as snap:
            list(snap.scan(StorageScanOptions()).keys())
        assert len(pub.batches) == 2


# =========================================================================
# Scans
# =========================================================================


def _keys(n_top: int = 12, n_mid: int = 10, n_leaf: int = 10, *, big: bool = True) -> list[tuple]:
    """A tree of mixed int/str keys. ``big=False`` keeps ints inside the string codec's range."""
    rng = random.Random(57)  # noqa: S311
    wide = 2**40 if big else 40000
    deep = 2**33 if big else 30000
    keys: set[tuple] = set()
    for i in range(n_top):
        top = rng.choice([i, -i - 1, f"t{i}", f"t{i}" * 3, "zz" * (i + 1), wide + i])
        keys.add((top,))
        for j in range(n_mid):
            mid = rng.choice([j, f"m{j}", -j * 1000, "a" * (j + 1)])
            keys.add((top, mid))
            for k in range(n_leaf):
                keys.add((top, mid, rng.choice([k, f"l{k}", -deep - k])))
    return sorted(keys, key=str)


def _option_grid(keys: list[tuple], codec) -> list[StorageScanOptions]:
    rng = random.Random(3)  # noqa: S311
    grid = [StorageScanOptions(), StorageScanOptions(reverse=True), StorageScanOptions(limit=0)]
    containers = [k for k in keys if len(k) < 3]
    for site in rng.sample(containers, 20):
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
            StorageScanOptions(
                start_encoded=codec.upper_bound_of_prefix(site), reverse=True, limit=1
            ),
            StorageScanOptions(start=site, reverse=True, limit=40),
            StorageScanOptions(start=site, limit=100),
            StorageScanOptions(break_filter=prefix),
            StorageScanOptions(reverse=True, break_filter=prefix),
        ]
    return grid


def _reference_scan(state: dict, codec, options: StorageScanOptions) -> list:
    """What a scan must return over ``state`` (key -> value), by the protocol's rules."""
    encoded = sorted((codec.encode_key(k), k) for k in state)
    encs = [e for e, _ in encoded]
    if options.start_encoded is not None:
        bound = options.start_encoded
    elif options.start is not None:
        bound = codec.encode_key(options.start)
    else:
        bound = None
    if options.reverse:
        rows = encoded[: bisect_right(encs, bound)][::-1] if bound is not None else encoded[::-1]
    else:
        rows = encoded[bisect_left(encs, bound) :] if bound is not None else encoded
    out = []
    for _, key in rows:
        if options.break_filter is not None and not options.break_filter.matches(key):
            break
        if options.filter is not None and not options.filter.matches(key):
            continue
        if options.limit is not None and len(out) >= options.limit:
            break
        out.append((key, state[key]))
    return out


def _fill(storage, items: dict) -> None:
    with storage.transaction() as tx:
        for k, v in items.items():
            tx.put(k, v)


@pytest.mark.parametrize("codec_cls", [BinaryCodec, TextCodec])
def test_scans_match_reference(codec_cls) -> None:
    keys = _keys(big=codec_cls is BinaryCodec)
    assert len(keys) > 1024  # crosses several scan pages
    state = {k: f"v{i}" for i, k in enumerate(keys)}
    with InMemoryStorage(codec=codec_cls()) as s:
        _fill(s, state)
        codec = s.codec
        for options in _option_grid(keys, codec):
            with s.snapshot() as snap:
                assert list(snap.scan(options).items()) == _reference_scan(state, codec, options)


def test_scans_match_sqlite(tmp_path) -> None:
    keys = _keys()
    state = {k: f"v{i}".encode() for i, k in enumerate(keys)}
    with (
        InMemoryStorage(codec=BinaryCodec()) as mem,
        SQLiteStorage(path=tmp_path / "ref.sqlite", codec=BinaryCodec()) as ref,
    ):
        _fill(mem, state)
        _fill(ref, state)
        for options in _option_grid(keys, mem.codec):
            with mem.snapshot() as a, ref.snapshot() as b:
                assert list(a.scan(options).items()) == list(b.scan(options).items()), options


def test_transaction_scans_merge_pending_writes() -> None:
    """Scans inside a transaction equal a reference over committed + pending state."""
    rng = random.Random(11)  # noqa: S311
    keys = _keys(8, 8, 8)
    committed = dict.fromkeys(keys, "c")
    with InMemoryStorage(codec=BinaryCodec()) as s:
        _fill(s, committed)
        codec = s.codec
        with s.transaction() as tx:
            state = dict(committed)
            for k in rng.sample(keys, len(keys) // 3):
                tx.delete(k)
                del state[k]
            for k in rng.sample(keys, len(keys) // 3):
                tx.put(k, "p")
                state[k] = "p"
            for i in range(200):
                k = ("new", rng.choice([i, f"n{i}", -i]))
                tx.put(k, "n")
                state[k] = "n"
            for options in _option_grid(keys, codec):
                assert list(tx.scan(options).items()) == _reference_scan(state, codec, options)
            tx.abort()
        with s.snapshot() as snap:
            assert list(snap.scan(StorageScanOptions()).items()) == _reference_scan(
                committed, codec, StorageScanOptions()
            )


def test_snapshot_scans_resolve_versions() -> None:
    """A snapshot's scans equal a reference over the state when it opened, after many commits."""
    rng = random.Random(12)  # noqa: S311
    keys = _keys(8, 8, 8)
    state = dict.fromkeys(keys, "c")
    with InMemoryStorage(codec=BinaryCodec()) as s:
        _fill(s, state)
        codec = s.codec
        with s.snapshot() as snap:
            for _ in range(20):
                with s.transaction() as tx:
                    for k in rng.sample(keys, 10):
                        tx.delete(k)
                    for k in rng.sample(keys, 10):
                        tx.put(k, "x")
                    for _ in range(10):
                        tx.put(("new", rng.randrange(1000)), "n")
            for options in _option_grid(keys, codec):
                assert list(snap.scan(options).items()) == _reference_scan(state, codec, options)
        assert s._store._history == {}


def test_delete_while_iterating(storage: InMemoryStorage) -> None:
    _fill(storage, {("a", i): i for i in range(300)})
    opts = StorageScanOptions(start=("a",), break_filter=PrefixFilter(prefix=("a",)))
    with storage.transaction() as tx:
        seen = 0
        for key in tx.scan(opts).keys():
            tx.delete(key)
            seen += 1
        assert seen == 300
        assert list(tx.scan(opts).keys()) == []
    with storage.snapshot() as snap:
        assert list(snap.scan(opts).keys()) == []


def test_keys_values_and_first_last(storage: InMemoryStorage) -> None:
    _fill(storage, {("a", i): i for i in range(100)})
    codec = storage.codec
    with storage.snapshot() as snap:
        opts = StorageScanOptions(start=("a",), break_filter=PrefixFilter(prefix=("a",)))
        assert list(snap.scan(opts).keys()) == [("a", i) for i in range(100)]
        assert list(snap.scan(opts).values()) == list(range(100))
        first = StorageScanOptions(start=("a",), limit=1)
        last = StorageScanOptions(
            start_encoded=codec.upper_bound_of_prefix(("a",)), reverse=True, limit=1
        )
        assert list(snap.scan(first).keys()) == [("a", 0)]
        assert list(snap.scan(last).keys()) == [("a", 99)]


def test_scan_on_closed_context_raises(storage: InMemoryStorage) -> None:
    snap = storage.begin_snapshot()
    scan = snap.scan(StorageScanOptions())
    snap.close()
    with pytest.raises(StorageClosedError):
        list(scan.items())
    with pytest.raises(StorageClosedError):
        snap.scan(StorageScanOptions())
