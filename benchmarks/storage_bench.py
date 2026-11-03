"""Storage backend benchmark: mem vs sqlite (vs lmdb) over the public protocol.

Every backend gets the exact same calls: ``begin_snapshot`` / ``transaction``
/ ``batch_write``, ``get``, ``put``, ``scan(StorageScanOptions(...))`` and the
scan's ``items()`` / ``keys()``. No backend internals are touched.

Data shape: keys look like view keys, ``("views", "users", "u00042", "item017")``,
grouped 100 per bucket so a prefix ``("views", "users", "u00042")`` matches
exactly 100 keys. Values alternate between small ints and small dicts.

Codec: mem uses ``NoOpCodec`` (what nustd's mem preset uses: binary keys,
values stored by reference); sqlite and lmdb use ``BinaryCodec`` (binary keys,
pickled values), since they must store bytes. ``--codec binary`` forces
``BinaryCodec`` on mem too for an apples-to-apples encode cost.

Timing: each op is run in a loop of ``iters`` calls (sized from a probe so
one repeat costs about ``--budget`` seconds), GC disabled inside the loop.
Per-op latency = loop time / iters; the reported value is the median over
``--repeat`` loops. Slow-op rule: if a single probe call takes longer than
``--max-op-seconds`` the op is not looped; the cell shows that single call,
marked ``(1x)``. This is what keeps the old mem backend (which sorts the
whole store on every scan) usable at 1M keys.

Run (from the repo root, editable install):
    .venv/bin/python benchmarks/storage_bench.py \
        --backends mem,sqlite --sizes 1000,100000 --repeat 3
"""

# ruff: noqa: D101, D103, ANN001, ANN201, ANN202, ANN401, T201, S311, TC003

from __future__ import annotations

import argparse
import gc
import random
import shutil
import statistics
import sys
import tempfile
import threading
import time
import zlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from virtuals.codecs import BinaryCodec, NoOpCodec
from virtuals.tkv.filter import PrefixFilter
from virtuals.tkv.storage import StorageScanOptions


BUCKET = 100  # keys per prefix
LOAD_BATCH = 10_000


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def make_key(i: int) -> tuple:
    return ("views", "users", f"u{i // BUCKET:06d}", f"item{i % BUCKET:03d}")


def make_value(i: int) -> Any:
    if i % 2:
        return i
    return {"id": i, "name": f"user-{i}", "active": True, "score": i * 0.5}


def bucket_prefix(b: int) -> tuple:
    return ("views", "users", f"u{b:06d}")


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


@dataclass
class Backend:
    name: str
    storage: Any
    cleanup: Callable[[], None] = field(default=lambda: None)


def make_backend(name: str, codec_mode: str) -> Backend:
    if name == "mem":
        from virtuals.storages.mem import InMemoryStorage

        codec = BinaryCodec() if codec_mode == "binary" else NoOpCodec()
        st = InMemoryStorage(codec)
        st.open()
        return Backend(name, st, st.close)

    tmp = Path(tempfile.mkdtemp(prefix=f"storage-bench-{name}-"))

    def _cleanup(st: Any) -> Callable[[], None]:
        def run() -> None:
            st.close()
            shutil.rmtree(tmp, ignore_errors=True)

        return run

    if name == "sqlite":
        from virtuals.storages.sqlite import SQLiteStorage

        st = SQLiteStorage(tmp / "bench.db", BinaryCodec())  # backend defaults: WAL, NORMAL
        st.open()
        return Backend(name, st, _cleanup(st))
    if name == "lmdb":
        from virtuals.storages.lmdb import LMDBStorage

        st = LMDBStorage(tmp / "bench.lmdb", BinaryCodec())  # backend defaults
        st.open()
        return Backend(name, st, _cleanup(st))
    raise SystemExit(f"unknown backend {name!r}")


def preload(st: Any, n: int) -> None:
    for lo in range(0, n, LOAD_BATCH):
        with st.batch_write() as wb:
            for i in range(lo, min(n, lo + LOAD_BATCH)):
                wb.put(make_key(i), make_value(i))


# ---------------------------------------------------------------------------
# Ops: each factory returns (op_fn, setup, teardown). op_fn(i) does one op.
# ---------------------------------------------------------------------------


@dataclass
class Op:
    name: str
    unit: str  # what one "op" is, for the report
    make: Callable[[Any, int, random.Random], tuple[Callable[[int], Any], Callable[[], None]]]


def _rand_keys(n: int, rng: random.Random, count: int = 4096) -> list[tuple]:
    return [make_key(rng.randrange(n)) for _ in range(count)]


def _rand_prefixes(n: int, rng: random.Random, count: int = 1024) -> list[tuple]:
    buckets = max(1, n // BUCKET)
    return [bucket_prefix(rng.randrange(buckets)) for _ in range(count)]


def op_get_snapshot(st, n, rng):
    keys = _rand_keys(n, rng)
    snap = st.begin_snapshot()
    m = len(keys) - 1
    return (lambda i: snap.get(keys[i & m])), snap.close


def op_get_txn(st, n, rng):
    keys = _rand_keys(n, rng)
    tx = st.begin_transaction()
    m = len(keys) - 1
    return (lambda i: tx.get(keys[i & m])), tx.abort


def op_put_commit1(st, n, rng):
    keys = _rand_keys(n, rng)
    m = len(keys) - 1

    def run(i):
        with st.transaction() as tx:
            tx.put(keys[i & m], i)

    return run, lambda: None


def op_put_txn100(st, n, rng):
    keys = _rand_keys(n, rng)
    m = len(keys) - 1

    def run(i):
        base = i * 100
        with st.transaction() as tx:
            for j in range(base, base + 100):
                tx.put(keys[j & m], j)

    return run, lambda: None


def _scan_op(*, limit=None, keys_only=False, reverse=False):
    def make(st, n, rng):
        prefixes = _rand_prefixes(n, rng)
        m = len(prefixes) - 1
        codec = st.codec
        if reverse:
            opts = [
                StorageScanOptions(
                    start_encoded=codec.upper_bound_of_prefix(p),
                    reverse=True,
                    break_filter=PrefixFilter(prefix=p),
                    limit=limit,
                )
                for p in prefixes
            ]
        else:
            opts = [
                StorageScanOptions(start=p, break_filter=PrefixFilter(prefix=p), limit=limit)
                for p in prefixes
            ]
        snap = st.begin_snapshot()
        expected = min(BUCKET, n) if limit is None else min(limit, BUCKET, n)

        def run(i):
            sc = snap.scan(opts[i & m])
            it = sc.keys() if keys_only else sc.items()
            c = 0
            for _ in it:
                c += 1
            if c != expected:
                raise AssertionError(f"scan returned {c}, expected {expected}")

        return run, snap.close

    return make


CONTENTION_THREADS = 4
CONTENTION_COMMITS = 50  # per thread per call


def op_put_commit1_threads(st, n, rng):
    """4 threads each doing 1-put commits on their own keys; per-commit wall time."""
    keys = [_rand_keys(n, rng, 1024) for _ in range(CONTENTION_THREADS)]
    m = 1023
    start = threading.Barrier(CONTENTION_THREADS + 1)
    done = threading.Barrier(CONTENTION_THREADS + 1)
    state = {"i": 0, "stop": False}
    errors: list[BaseException] = []

    def worker(t):
        ks = keys[t]
        while True:
            start.wait()
            if state["stop"]:
                return
            base = state["i"] * CONTENTION_COMMITS
            try:
                for j in range(base, base + CONTENTION_COMMITS):
                    with st.transaction() as tx:
                        tx.put(ks[j & m], j)
            except BaseException as e:
                errors.append(e)
            done.wait()

    threads = [
        threading.Thread(target=worker, args=(t,), daemon=True) for t in range(CONTENTION_THREADS)
    ]
    for th in threads:
        th.start()

    def run(i):
        state["i"] = i
        start.wait()
        done.wait()
        if errors:
            raise errors[0]

    def close():
        state["stop"] = True
        start.wait()
        for th in threads:
            th.join()

    return run, close


def op_snapshot_open_close(st, n, rng):
    def run(i):
        st.begin_snapshot().close()

    return run, lambda: None


OPS: list[Op] = [
    Op("get (snapshot)", "get", op_get_snapshot),
    Op("get (write txn)", "get", op_get_txn),
    Op("put, 1 per txn (commit)", "txn", op_put_commit1),
    Op("put, 100 per txn (per put)", "put", op_put_txn100),
    Op("put, 1 per txn, 4 threads (per commit)", "txn", op_put_commit1_threads),
    Op("prefix scan 100 items", "scan", _scan_op()),
    Op("prefix scan 100 keys-only", "scan", _scan_op(keys_only=True)),
    Op("prefix scan limit=10", "scan", _scan_op(limit=10)),
    Op("first (fwd limit=1)", "scan", _scan_op(limit=1)),
    Op("last (rev limit=1)", "scan", _scan_op(limit=1, reverse=True)),
    Op("snapshot open+close", "snap", op_snapshot_open_close),
]
PER_CALL_DIVISOR = {
    "put, 100 per txn (per put)": 100,
    "put, 1 per txn, 4 threads (per commit)": CONTENTION_THREADS * CONTENTION_COMMITS,
}


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


@dataclass
class Result:
    us: float | None  # per-op latency, microseconds
    single: bool = False  # slow-op rule hit, one call measured
    note: str = ""


def _loop(fn: Callable[[int], Any], start: int, iters: int) -> float:
    gc.disable()
    try:
        t0 = time.perf_counter()
        for i in range(start, start + iters):
            fn(i)
        return time.perf_counter() - t0
    finally:
        gc.enable()


def measure(op: Op, st: Any, n: int, args: argparse.Namespace) -> Result:
    rng = random.Random(zlib.crc32(f"{op.name}|{n}".encode()))
    div = PER_CALL_DIVISOR.get(op.name, 1)
    gc.collect()
    fn, close = op.make(st, n, rng)
    try:
        probe = _loop(fn, 0, 1)
        if probe > args.max_op_seconds:
            return Result(probe * 1e6 / div, single=True)
        # second probe to get past first-call warmup
        k = 1
        while True:
            t = _loop(fn, 1, k)
            if t > 0.02 or k >= 1 << 20:
                break
            k *= 4
        per = t / k
        iters = max(3, min(args.max_iters, int(args.budget / per)))
        samples = []
        counter = 1 + k
        for _ in range(args.repeat):
            samples.append(_loop(fn, counter, iters) / iters)
            counter += iters
        return Result(statistics.median(samples) * 1e6 / div)
    finally:
        close()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def fmt_us(us: float) -> str:
    if us >= 1e6:
        return f"{us / 1e6:.2f} s"
    if us >= 1e3:
        return f"{us / 1e3:.2f} ms"
    if us >= 10:
        return f"{us:.1f} us"
    return f"{us:.2f} us"


def fmt_rate(us: float) -> str:
    r = 1e6 / us
    if r >= 1e6:
        return f"{r / 1e6:.1f}M/s"
    if r >= 1e3:
        return f"{r / 1e3:.0f}k/s"
    return f"{r:.0f}/s" if r >= 10 else f"{r:.1f}/s"


def fmt_size(n: int) -> str:
    if n >= 1_000_000 and n % 1_000_000 == 0:
        return f"{n // 1_000_000}M"
    if n >= 1000 and n % 1000 == 0:
        return f"{n // 1000}k"
    return str(n)


def cell(r: Result | None) -> str:
    if r is None:
        return "-"
    if r.us is None:
        return r.note or "skip"
    s = f"{fmt_us(r.us)} ({fmt_rate(r.us)})"
    return s + " (1x)" if r.single else s


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--backends", default="mem,sqlite", help="comma list: mem,sqlite,lmdb")
    p.add_argument("--sizes", default="1000,100000,1000000", help="comma list of key counts")
    p.add_argument("--repeat", type=int, default=3, help="timed loops per op (median taken)")
    p.add_argument("--budget", type=float, default=0.25, help="target seconds per timed loop")
    p.add_argument("--max-iters", type=int, default=200_000)
    p.add_argument(
        "--max-op-seconds",
        type=float,
        default=0.25,
        help="slow-op rule: a probe call slower than this is reported once, not looped",
    )
    p.add_argument("--codec", choices=["default", "binary"], default="default")
    p.add_argument("--ops", default="", help="substring filter on op names (comma list)")
    args = p.parse_args()

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    ops = OPS
    if args.ops:
        wanted = [w.strip() for w in args.ops.split(",")]
        ops = [o for o in OPS if any(w in o.name for w in wanted)]

    import virtuals

    print(f"virtuals: {virtuals.__file__}", file=sys.stderr)
    t_all = time.perf_counter()
    results: dict[tuple[str, int], dict[str, Result]] = {}
    for n in sizes:
        for b in backends:
            be = make_backend(b, args.codec)
            try:
                t0 = time.perf_counter()
                preload(be.storage, n)
                load_s = time.perf_counter() - t0
                print(f"[{b} {fmt_size(n)}] loaded in {load_s:.1f}s", file=sys.stderr)
                col: dict[str, Result] = {}
                for op in ops:
                    t0 = time.perf_counter()
                    col[op.name] = measure(op, be.storage, n, args)
                    print(
                        f"[{b} {fmt_size(n)}] {op.name}: {cell(col[op.name])}"
                        f" [{time.perf_counter() - t0:.1f}s]",
                        file=sys.stderr,
                    )
                col["(preload, not timed per op)"] = Result(load_s * 1e6, single=True)
                results[(b, n)] = col
            finally:
                be.cleanup()

    cols = [(b, n) for n in sizes for b in backends]
    print()
    print("| op | " + " | ".join(f"{b} {fmt_size(n)}" for b, n in cols) + " |")
    print("|---|" + "---|" * len(cols))
    for op in ops:
        print(f"| {op.name} | " + " | ".join(cell(results[c].get(op.name)) for c in cols) + " |")
    print(
        "| bulk load (batch_write, 10k/batch) | "
        + " | ".join(fmt_us(results[c]["(preload, not timed per op)"].us) for c in cols)
        + " |"
    )
    print()
    print(
        f"median per-op latency (ops/s), repeat={args.repeat}, codec={args.codec},"
        f" total {time.perf_counter() - t_all:.0f}s. (1x) = slow-op rule, single call."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
