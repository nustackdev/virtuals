"""Section 8: random storage ops against a reference model, on every backend.

A Hypothesis state machine drives one storage with begin (sync on the main
thread, ``areserve_write_slot`` on a loop thread, ``begin_transaction`` on a
worker thread), put, delete, get, exists, multiget, scan, commit, abort,
drop without end, ops on an ended transaction, nested begin through every
route, contended begin from another thread, task or the loop, write batches,
up to three open snapshots, close and reopen with open contexts, bulk rows
across scan pages, big values and odd or oversized keys.

The model is a committed dict, a pending overlay with tombstones and one
frozen dict per open snapshot. Scan expectations come from the model sorted
by encoded key bytes, honouring start, reverse, limit, filter, break filter
and ``max_depth``.

Two tiers, see ``tests/README.md``: ``fast`` runs in the default suite with a
fixed seed and a few short examples; ``slow`` (``-m slow``) runs many more
and takes about a minute per backend. ``CONFORMANCE_MODEL_EXAMPLES`` and
``CONFORMANCE_MODEL_SEED`` drive the slow tier.
"""

from __future__ import annotations

import asyncio
import gc
import os
import shutil
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import hypothesis.strategies as st
import pytest
from hypothesis import HealthCheck, Phase, seed, settings
from hypothesis.stateful import (
    RuleBasedStateMachine,
    invariant,
    precondition,
    rule,
    run_state_machine_as_test,
)

from virtuals.tkv.filter import LengthFilter, PrefixFilter
from virtuals.tkv.storage import (
    StorageClosedError,
    StorageError,
    StorageInterfaceError,
    StorageLockTimeoutError,
    StorageNestedTransactionError,
    StorageScanOptions,
)
from virtuals.tkv.types import EMPTY, is_empty


if TYPE_CHECKING:
    from tests.conformance.backends import BackendSpec


# ---------------------------------------------------------------------------- tiers


@dataclass(frozen=True)
class Tier:
    name: str
    examples: int
    steps: int
    seed: int
    busy: float  # busy_timeout of the storage under test: contention resolves fast


FAST = Tier("fast", examples=8, steps=40, seed=20261104, busy=0.1)
SLOW = Tier(
    "slow",
    examples=int(os.environ.get("CONFORMANCE_MODEL_EXAMPLES", 60)),
    steps=80,
    seed=int(os.environ.get("CONFORMANCE_MODEL_SEED", 20261104)),
    busy=0.3,
)

TIERS = [
    pytest.param(FAST, id="fast", marks=pytest.mark.timeout(120)),
    pytest.param(SLOW, id="slow", marks=[pytest.mark.slow, pytest.mark.timeout(900)]),
]


def hypothesis_settings(tier: Tier) -> settings:
    return settings(
        max_examples=tier.examples,
        stateful_step_count=tier.steps,
        deadline=None,
        database=None,
        suppress_health_check=list(HealthCheck),
        phases=[Phase.generate, Phase.shrink],
        print_blob=True,
    )


# ---------------------------------------------------------------------------- strategies

TOMB = object()

ATOMS = ["a", "b", "ab", 0, 1, 10]
keys_st = st.lists(st.sampled_from(ATOMS), min_size=1, max_size=3).map(tuple)
values_st = st.one_of(
    st.integers(-5, 5),
    st.text(max_size=3),
    st.none(),
    st.binary(max_size=3),
    st.lists(st.integers(0, 3), max_size=2),
    st.dictionaries(st.text(max_size=2), st.integers(0, 3), max_size=2),
    st.booleans(),
)
prefix_st = st.lists(st.sampled_from(ATOMS), min_size=0, max_size=2).map(tuple)


@st.composite
def scan_opts(draw):
    start = draw(st.one_of(st.none(), keys_st))
    reverse = draw(st.booleans())
    limit = draw(st.one_of(st.none(), st.integers(1, 4)))
    brk = draw(st.one_of(st.none(), prefix_st.map(lambda p: PrefixFilter(prefix=p))))
    flt = draw(
        st.one_of(
            st.none(),
            prefix_st.map(lambda p: PrefixFilter(prefix=p)),
            st.integers(1, 3).map(lambda n: LengthFilter(length=n)),
            st.tuples(prefix_st, st.integers(1, 3)).map(
                lambda t: PrefixFilter(prefix=t[0]) & LengthFilter(length=t[1])
            ),
        )
    )
    max_depth = draw(st.one_of(st.none(), st.integers(1, 3)))
    return StorageScanOptions(
        start=start, reverse=reverse, limit=limit, filter=flt, break_filter=brk, max_depth=max_depth
    )


# Values stay well under the 64MB LMDB map of the conformance spec, even when
# one transaction writes a few of them (freed pages return after commit).
BIG_SIZES = [1 << 16, 1 << 20, 2 << 20]
KEY_LENGTHS = [200, 480, 600, 4000]
ODD_KEYS = [("\x00",), ("a\x00b", 1), ("é", "ß"), ("￿",), ("",), (2**62,), (-(2**62),)]


# ---------------------------------------------------------------------------- machine


def expect(exc, fn) -> None:
    """``fn`` must raise ``exc``. An AssertionError (not a pytest outcome) keeps hypothesis shrinking."""
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"did not raise {exc}")


class LoopThread:
    """An event loop on its own thread; ``call`` runs a plain fn on it, ``run`` a coroutine."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True, name="s8-loop")
        self.thread.start()

    def call(self, fn, *args):
        fut = asyncio.run_coroutine_threadsafe(self._wrap(fn, *args), self.loop)
        return fut.result(30)

    async def _wrap(self, fn, *args):
        return fn(*args)

    def run(self, coro_fn, *args):
        fut = asyncio.run_coroutine_threadsafe(coro_fn(*args), self.loop)
        return fut.result(30)

    def stop(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(5)
        self.loop.close()


class StorageMachine(RuleBasedStateMachine):
    """One storage against the model. Subclassed per backend and tier by the test."""

    spec: BackendSpec
    tier: Tier
    scratch: Path
    ops: Counter

    def __init__(self) -> None:
        super().__init__()
        self.root = Path(tempfile.mkdtemp(prefix=f"s8-{self.spec.name}-", dir=self.scratch))
        self.busy = self.tier.busy
        self.storage = self.spec.make(self.root, lock_timeout=self.busy)
        self.storage.open()
        self.codec = self.storage.codec
        self.committed: dict = {}
        self.pending: dict | None = None
        self.txn = None
        self.txn_mode = None  # "sync" | "async" | "thread"
        self.batch = None
        self.batch_pending: dict = {}
        self.snaps: list[tuple[object, dict]] = []
        self.loop = LoopThread()
        self.worker = ThreadPoolExecutor(1, thread_name_prefix="s8-worker")
        self.opened = True

    # ------------------------------------------------------------------ helpers
    def count(self, name: str) -> None:
        self.ops[name] += 1

    def on(self, fn, *args):
        """Run ``fn`` on the thread that owns the open transaction."""
        if self.txn_mode == "async":
            return self.loop.call(fn, *args)
        if self.txn_mode == "thread":
            return self.worker.submit(fn, *args).result(30)
        return fn(*args)

    def view(self) -> dict:
        d = dict(self.committed)
        if self.pending is not None:
            for k, v in self.pending.items():
                if v is TOMB:
                    d.pop(k, None)
                else:
                    d[k] = v
        return d

    def expect_scan(self, model: dict, opts: StorageScanOptions) -> list:
        enc = {k: self.codec.encode_key(k) for k in model}
        order = sorted(model, key=lambda k: enc[k], reverse=opts.reverse)
        if opts.start is not None:
            s = self.codec.encode_key(opts.start)
            order = [k for k in order if (enc[k] <= s if opts.reverse else enc[k] >= s)]
        out = []
        for k in order:
            if opts.break_filter is not None and not opts.break_filter.matches(k):
                break
            if opts.max_depth is not None and len(k) > opts.max_depth:
                continue
            if opts.filter is not None and not opts.filter.matches(k):
                continue
            out.append((k, model[k]))
            if opts.limit is not None and len(out) == opts.limit:
                break
        return out

    def check_reads(self, ctx, model: dict, key, keys, opts) -> None:
        got = self.on(ctx.get, key)
        if key in model:
            assert not is_empty(got) and got == model[key], (key, got, model.get(key, EMPTY))
        else:
            assert is_empty(got), (key, got)
        assert self.on(ctx.exists, key) == (key in model)
        mg = self.on(ctx.multiget, list(keys))
        assert mg == {k: model[k] for k in keys if k in model}, (mg, keys)
        rows = self.on(lambda: list(ctx.scan(opts).items()))
        exp = self.expect_scan(model, opts)
        assert rows == exp, f"scan {opts}\n got={rows}\n exp={exp}"
        ks = self.on(lambda: list(ctx.scan(opts).keys()))
        assert ks == [k for k, _ in exp]

    def begin_sync(self):
        return self.storage.begin_transaction()

    def begin_async(self):
        async def go():
            slot = await self.storage.areserve_write_slot()
            return self.storage.begin_transaction(write_slot=slot)

        return self.loop.run(go)

    def begin_thread(self):
        return self.worker.submit(self.storage.begin_transaction).result(30)

    def _end_txn(self) -> None:
        self.txn = None
        self.pending = None
        self.txn_mode = None

    # ------------------------------------------------------------------ rules: txn lifecycle
    @precondition(lambda s: s.opened and s.txn is None and s.batch is None)
    @rule(mode=st.sampled_from(["sync", "async", "thread"]))
    def begin(self, mode):
        self.count("begin_" + mode)
        begin = {"sync": self.begin_sync, "async": self.begin_async, "thread": self.begin_thread}
        self.txn = begin[mode]()
        self.txn_mode = mode
        self.pending = {}
        assert self.on(lambda: self.txn.is_active)

    @precondition(lambda s: s.opened and s.txn is not None)
    @rule(key=keys_st, value=values_st)
    def put(self, key, value):
        self.count("put")
        self.on(self.txn.put, key, value)
        self.pending[key] = value

    @precondition(lambda s: s.opened and s.txn is not None)
    @rule(key=keys_st)
    def delete(self, key):
        self.count("delete")
        self.on(self.txn.delete, key)
        self.pending[key] = TOMB

    @precondition(lambda s: s.opened and s.txn is not None)
    @rule(key=keys_st, keys=st.lists(keys_st, max_size=4), opts=st.data())
    def read_in_txn(self, key, keys, opts):
        self.count("read_txn")
        self.check_reads(self.txn, self.view(), key, keys, opts.draw(scan_opts()))

    @precondition(lambda s: s.opened and s.txn is not None)
    @rule()
    def commit(self):
        self.count("commit")
        self.on(self.txn.commit)
        assert self.on(lambda: self.txn.is_closed)
        for k, v in self.pending.items():
            if v is TOMB:
                self.committed.pop(k, None)
            else:
                self.committed[k] = v
        self._end_txn()

    @precondition(lambda s: s.opened and s.txn is not None)
    @rule()
    def abort(self):
        self.count("abort")
        self.on(self.txn.abort)
        assert self.on(lambda: self.txn.is_closed)
        self._end_txn()

    @precondition(lambda s: s.opened and s.txn is not None)
    @rule()
    def drop_without_end(self):
        """Drop the transaction object. Its finalizer must free the slot."""
        self.count("drop")
        self._end_txn()
        gc.collect()
        t0 = time.monotonic()
        t = self.storage.begin_transaction()  # must land promptly on the freed slot
        dt = time.monotonic() - t0
        t.abort()
        assert dt < 0.2, f"begin after drop took {dt:.3f}s"

    @precondition(lambda s: s.opened and s.txn is not None)
    @rule()
    def ended_txn_refuses_ops(self):
        """Commit, then every op on an ended transaction raises StorageClosedError."""
        self.count("ended_refuses")
        self.commit()
        t = self.storage.begin_transaction()
        t.abort()
        for op in (
            lambda: t.get(("a",)),
            lambda: t.put(("a",), 1),
            lambda: t.delete(("a",)),
            lambda: list(t.scan(StorageScanOptions()).items()),
            t.commit,
        ):
            expect(StorageClosedError, op)

    # ------------------------------------------------------------------ rules: bulk rows (scan paging)
    @precondition(lambda s: s.opened and s.txn is not None and s.txn_mode == "sync")
    @rule(n=st.sampled_from([300, 1500]), deep=st.booleans())
    def bulk_put_and_scan(self, n, deep):
        """Many rows under one prefix in one txn, scanned across page boundaries."""
        self.count("bulk_put")
        for i in range(n):
            k = ("bulk", i, "x") if deep and i % 3 == 0 else ("bulk", i)
            self.txn.put(k, i)
            self.pending[k] = i
        pf = PrefixFilter(prefix=("bulk",))
        for opts in (
            StorageScanOptions(start=("bulk",), break_filter=pf),
            StorageScanOptions(start=("bulk", n), reverse=True, break_filter=pf),
            StorageScanOptions(start=("bulk",), break_filter=pf, max_depth=2),
            StorageScanOptions(start=("bulk", 100), break_filter=pf, limit=700),
            StorageScanOptions(start=("bulk", 100), reverse=True, break_filter=pf, limit=50),
            StorageScanOptions(start=("bulk",), break_filter=pf, filter=LengthFilter(length=3)),
        ):
            rows = list(self.txn.scan(opts).items())
            exp = self.expect_scan(self.view(), opts)
            assert rows == exp, f"bulk scan {opts}: got {len(rows)} rows, expected {len(exp)}"

    @precondition(
        lambda s: (
            s.opened
            and s.txn is not None
            and s.txn_mode == "sync"
            and any(k[0] == "bulk" for k in s.view())
        )
    )
    @rule(step=st.sampled_from([2, 3]))
    def bulk_delete_some(self, step):
        self.count("bulk_delete")
        for i, k in enumerate(sorted(k for k in self.view() if k[0] == "bulk")):
            if i % step == 0:
                self.txn.delete(k)
                self.pending[k] = TOMB
        opts = StorageScanOptions(start=("bulk",), break_filter=PrefixFilter(prefix=("bulk",)))
        rows = list(self.txn.scan(opts).items())
        assert rows == self.expect_scan(self.view(), opts)

    # ------------------------------------------------------------------ rules: odd sizes and characters
    @precondition(lambda s: s.opened and s.txn is not None)
    @rule(size=st.sampled_from(BIG_SIZES))
    def big_value(self, size):
        self.count("big_value")
        v = os.urandom(size)
        self.on(self.txn.put, ("big",), v)
        self.pending[("big",)] = v
        assert self.on(self.txn.get, ("big",)) == v

    @precondition(lambda s: s.opened and s.txn is not None)
    @rule(n=st.sampled_from(KEY_LENGTHS), which=st.sampled_from(["put", "delete"]))
    def oversized_key(self, n, which):
        """A key near or past the backend's limit: stored and read back, or a clean StorageError."""
        self.count("oversized_key")
        k = ("k" * n,)
        try:
            if which == "put":
                self.on(self.txn.put, k, n)
            else:
                self.on(self.txn.delete, k)
        except StorageError:
            # refused cleanly: the txn must still work and the key must not be there
            assert is_empty(self.on(self.txn.get, k))
            self.on(self.txn.put, ("after_oversized",), 1)
            self.pending[("after_oversized",)] = 1
            return
        if which == "put":
            self.pending[k] = n
            assert self.on(self.txn.get, k) == n
        else:
            self.pending[k] = TOMB

    @precondition(lambda s: s.opened and s.txn is not None)
    @rule(k=st.sampled_from(ODD_KEYS))
    def odd_key(self, k):
        self.count("odd_key")
        try:
            self.on(self.txn.put, k, "odd")
        except StorageError:
            assert is_empty(self.on(self.txn.get, k))
            return
        self.pending[k] = "odd"
        assert self.on(self.txn.get, k) == "odd"
        rows = self.on(lambda: list(self.txn.scan(StorageScanOptions(start=k, limit=1)).items()))
        assert rows and rows[0] == (k, "odd"), rows

    # ------------------------------------------------------------------ rules: nesting and contention
    @precondition(lambda s: s.opened and s.txn is not None and s.txn_mode == "sync")
    @rule(how=st.sampled_from(["sync", "batch", "to_thread", "asyncio_run"]))
    def nested_begin_raises(self, how):
        self.count("nested_" + how)

        async def reserve_and_begin():
            slot = await self.storage.areserve_write_slot()
            return self.storage.begin_transaction(write_slot=slot)

        def batch_write():
            b = self.storage.begin_write_batch()
            b.put(("nested",), 1)
            b.write()

        attempt = {
            "sync": self.storage.begin_transaction,
            "batch": batch_write,
            "to_thread": lambda: asyncio.run(asyncio.to_thread(self.storage.begin_transaction)),
            "asyncio_run": lambda: asyncio.run(reserve_and_begin()),
        }[how]
        expect(StorageNestedTransactionError, attempt)

    @precondition(lambda s: s.opened and s.txn is not None and s.txn_mode != "sync")
    @rule()
    def contended_sync_begin_times_out(self):
        """Main thread sync begin while the loop or worker thread holds: lock timeout, bounded."""
        self.count("contended_sync")
        t0 = time.monotonic()
        expect(StorageLockTimeoutError, lambda: self.storage.begin_transaction())
        dt = time.monotonic() - t0
        assert dt < self.busy + 1.0, dt

    @precondition(lambda s: s.opened and s.txn is not None and s.txn_mode == "sync")
    @rule(how=st.sampled_from(["sync_on_loop", "areserve_on_loop", "plain_thread"]))
    def contended_from_elsewhere(self, how):
        """Another task or thread begins while the main thread holds: lock timeout, never a hang."""
        self.count("contended_" + how)
        attempt = {
            "sync_on_loop": lambda: self.loop.call(self.storage.begin_transaction),
            "areserve_on_loop": lambda: self.loop.run(self.storage.areserve_write_slot),
            "plain_thread": lambda: self.worker.submit(self.storage.begin_transaction).result(30),
        }[how]
        t0 = time.monotonic()
        expect(StorageLockTimeoutError, attempt)
        dt = time.monotonic() - t0
        assert dt < self.busy + 1.0, dt

    # ------------------------------------------------------------------ rules: write batch
    @precondition(lambda s: s.opened and s.txn is None and s.batch is None)
    @rule()
    def batch_begin(self):
        self.count("batch_begin")
        self.batch = self.storage.begin_write_batch()
        self.batch_pending = {}

    @precondition(lambda s: s.opened and s.batch is not None)
    @rule(key=keys_st, value=values_st, delete=st.booleans())
    def batch_op(self, key, value, delete):
        self.count("batch_op")
        if delete:
            self.batch.delete(key)
            self.batch_pending[key] = TOMB
        else:
            self.batch.put(key, value)
            self.batch_pending[key] = value

    @precondition(lambda s: s.opened and s.batch is not None)
    @rule()
    def batch_read_is_interface_error(self):
        self.count("batch_read")
        for op in (lambda: self.batch.get(("a",)), lambda: self.batch.exists(("a",))):
            expect((StorageInterfaceError, AttributeError), op)

    @precondition(lambda s: s.opened and s.batch is not None)
    @rule(write=st.booleans())
    def batch_end(self, write):
        self.count("batch_write" if write else "batch_abort")
        if write:
            self.batch.write()
            for k, v in self.batch_pending.items():
                if v is TOMB:
                    self.committed.pop(k, None)
                else:
                    self.committed[k] = v
        else:
            self.batch.abort()
        assert self.batch.is_closed
        self.batch = None

    # ------------------------------------------------------------------ rules: snapshots
    @precondition(lambda s: s.opened and len(s.snaps) < 3)
    @rule()
    def snap_open(self):
        self.count("snap_open")
        self.snaps.append((self.storage.begin_snapshot(), dict(self.committed)))

    @precondition(lambda s: s.opened and s.snaps)
    @rule(i=st.integers(0, 2), key=keys_st, keys=st.lists(keys_st, max_size=4), opts=st.data())
    def snap_read(self, i, key, keys, opts):
        self.count("snap_read")
        snap, frozen = self.snaps[i % len(self.snaps)]
        saved, self.txn_mode = self.txn_mode, None
        try:
            self.check_reads(snap, frozen, key, keys, opts.draw(scan_opts()))
        finally:
            self.txn_mode = saved

    @precondition(lambda s: s.opened and s.snaps)
    @rule(i=st.integers(0, 2))
    def snap_write_is_interface_error(self, i):
        self.count("snap_write")
        snap, _ = self.snaps[i % len(self.snaps)]
        for op in (lambda: snap.put(("a",), 1), lambda: snap.delete(("a",))):
            expect((StorageInterfaceError, AttributeError), op)

    @precondition(lambda s: s.snaps)
    @rule(i=st.integers(0, 2))
    def snap_close(self, i):
        self.count("snap_close")
        snap, _ = self.snaps.pop(i % len(self.snaps))
        snap.close()
        assert snap.is_closed
        expect(StorageClosedError, lambda: snap.get(("a",)))

    # ------------------------------------------------------------------ rules: fresh reader sees committed only
    @precondition(lambda s: s.opened)
    @rule(key=keys_st, keys=st.lists(keys_st, max_size=4), opts=st.data())
    def fresh_snapshot_sees_committed(self, key, keys, opts):
        self.count("fresh_snap")
        saved, self.txn_mode = self.txn_mode, None
        try:
            with self.storage.snapshot() as snap:
                self.check_reads(snap, self.committed, key, keys, opts.draw(scan_opts()))
        finally:
            self.txn_mode = saved

    # ------------------------------------------------------------------ rules: close and reopen
    @precondition(lambda s: s.opened)
    @rule()
    def close_reopen(self):
        self.count("close_reopen")
        old_txn, old_snaps, old_batch = self.txn, list(self.snaps), self.batch
        self.storage.close()
        self.opened = False
        # open contexts are ended by close; their use raises StorageClosedError
        for ctx in [old_txn, old_batch, *[s for s, _ in old_snaps]]:
            if ctx is None:
                continue
            assert ctx.is_closed, f"{type(ctx).__name__} not closed by storage.close()"
            op = (ctx.get, ("a",)) if hasattr(ctx, "get") else (ctx.put, ("a",), 1)
            expect(StorageClosedError, lambda op=op: self.on(*op))
        self._end_txn()
        self.batch = None
        self.snaps = []
        expect(StorageClosedError, lambda: self.storage.begin_snapshot())
        self.storage.open()
        self.opened = True
        if self.spec.name == "mem":
            self.committed = {}
        with self.storage.snapshot() as snap:
            rows = list(snap.scan(StorageScanOptions()).items())
        assert rows == self.expect_scan(self.committed, StorageScanOptions()), (
            f"after reopen\n got={rows}\n exp={sorted(self.committed.items())}"
        )

    @invariant()
    def txn_and_batch_exclusive(self):
        assert not (self.txn is not None and self.batch is not None)

    def teardown(self) -> None:
        try:
            if self.opened:
                self.storage.close()
        finally:
            self.loop.stop()
            self.worker.shutdown(wait=False)
            shutil.rmtree(self.root, ignore_errors=True)


# ---------------------------------------------------------------------------- case


@pytest.mark.parametrize("tier", TIERS)
def test_s8_storage_model(backend: BackendSpec, tier: Tier, tmp_path):
    """Random transactions, batches, snapshots, reopen: the backend matches the model."""
    ops: Counter = Counter()
    cls = type(
        f"StorageMachine_{backend.name}_{tier.name}",
        (StorageMachine,),
        {"spec": backend, "tier": tier, "scratch": tmp_path, "ops": ops},
    )
    t0 = time.monotonic()
    try:
        run_state_machine_as_test(seed(tier.seed)(cls), settings=hypothesis_settings(tier))
    finally:
        dt = time.monotonic() - t0
        print(
            f"[s8 model {backend.name} {tier.name}] seed={tier.seed} examples={tier.examples} "
            f"steps={tier.steps} ops={sum(ops.values())} time={dt:.1f}s"
        )
        print("  " + ", ".join(f"{k}={v}" for k, v in sorted(ops.items())))
