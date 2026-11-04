"""Sections 1 and 4 under random concurrent load, across threads, coroutines and processes.

One storage, N threads, M coroutines (one loop on its own thread, a write
slot reserved with an ``await`` between read and write) and K spawned
processes, each looping over three random bodies:

- bank transfer between two of eight accounts: the sum is constant
- claim: scan the ``("slot", *)`` range, take the first free index, write
  the worker id and bump a claims counter (insert race on a scanned range)
- snapshot check: the sum holds and the slot row count equals the claims
  counter (atomic visibility of a multi-key commit)

Every write transaction bumps an ops counter. After the run the storage is
closed and reopened (mem: checked live) and the invariants are checked
against what the workers recorded: ops equals the recorded commits (a lost
update would show), every recorded claim is present under its recorder (a
lost insert race would show), slot indices are contiguous (a scan skew would
leave a gap). Workers retry only the two retryable error classes; any other
class fails the case. Process workers fsync each commit to a log after the
commit returns, so a SIGKILLed process still proves what it committed.

Three cases per backend: mixed load with the default busy timeout, mixed load
with a 20ms busy timeout (the retry paths run), and mixed load with SIGKILL
of writer processes (multi-process backends only). Each case has a ``fast``
tier (about a second, default suite) and a ``slow`` tier (``-m slow``, see
``tests/README.md``). ``CONFORMANCE_MODEL_SECONDS`` scales the slow tier.
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import random
import signal
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest

from tests.conformance.backends import BACKENDS, BackendSpec
from tests.conformance.helpers import RETRYABLE
from virtuals.tkv.filter import PrefixFilter
from virtuals.tkv.storage import StorageScanOptions
from virtuals.tkv.types import is_empty


# ---------------------------------------------------------------------------- tiers


@dataclass(frozen=True)
class Load:
    name: str
    threads: int
    coros: int
    procs: int  # 0 on single-process backends
    seconds: float
    kills: int = 0
    busy: float | None = None  # busy_timeout override, None is the backend default
    pause: float = 0.0005  # held between the read and the dependent write in sync workers
    seed: int = 7


_SLOW_SCALE = float(os.environ.get("CONFORMANCE_MODEL_SECONDS", 30)) / 30

FAST_MIXED = Load("fast", threads=2, coros=2, procs=1, seconds=0.8)
SLOW_MIXED = Load("slow", threads=4, coros=4, procs=3, seconds=30 * _SLOW_SCALE)
# The busy cases hold each write 10ms, so with four threads queued the slot
# wait passes the 20ms busy timeout and the retry paths run on every backend,
# in the fast tier too (4ms was not enough on mem and rocksdb).
FAST_BUSY = Load("fast", threads=4, coros=4, procs=1, seconds=0.8, busy=0.02, pause=0.01, seed=99)
SLOW_BUSY = replace(SLOW_MIXED, seconds=15 * _SLOW_SCALE, busy=0.02, pause=0.01, seed=99)
FAST_KILL = Load("fast", threads=2, coros=2, procs=2, seconds=2.0, kills=1, seed=11)
SLOW_KILL = Load("slow", threads=2, coros=2, procs=4, seconds=25 * _SLOW_SCALE, kills=3, seed=11)


def tiers(fast: Load, slow: Load) -> list:
    return [
        pytest.param(fast, id="fast", marks=pytest.mark.timeout(60)),
        pytest.param(slow, id="slow", marks=[pytest.mark.slow, pytest.mark.timeout(600)]),
    ]


ACCOUNTS = 8
NSLOTS = 100_000
TOTAL = ACCOUNTS * 1000
KILL_WARMUP = 0.8  # seconds before the first kill; a victim is only a process that reported ready


# ---------------------------------------------------------------------------- bodies


def _acct(i: int) -> tuple:
    return ("acct", i)


def _slot(j: int) -> tuple:
    return ("slot", j)


def slots_scan() -> StorageScanOptions:
    return StorageScanOptions(start=("slot",), break_filter=PrefixFilter(prefix=("slot",)))


def _counter(ctx, key: tuple) -> int:
    v = ctx.get(key)
    return 0 if is_empty(v) else v


def check_snapshot(snap) -> None:
    s = sum(snap.get(_acct(i)) for i in range(ACCOUNTS))
    assert s == TOTAL, f"torn or lost: sum={s} expected={TOTAL}"
    rows = list(snap.scan(slots_scan()).items())
    claims = _counter(snap, ("claims",))
    assert len(rows) == claims, f"slot rows={len(rows)} claims counter={claims}"
    assert len({k for k, _ in rows}) == len(rows)


def transfer(t, rnd: random.Random, pause) -> tuple:
    a, b = rnd.sample(range(ACCOUNTS), 2)
    va, vb = t.get(_acct(a)), t.get(_acct(b))
    amt = rnd.randint(1, 50)
    pause()
    t.put(_acct(a), va - amt)
    t.put(_acct(b), vb + amt)
    t.put(("ops",), _counter(t, ("ops",)) + 1)
    return "transfer", None


def claim(t, wid: str, pause) -> tuple:
    taken = {k[1] for k in t.scan(slots_scan()).keys()}
    free = next((j for j in range(NSLOTS) if j not in taken), None)
    pause()
    if free is None:
        return "claim", False
    assert is_empty(t.get(_slot(free)))
    t.put(_slot(free), wid)
    t.put(("claims",), _counter(t, ("claims",)) + 1)
    t.put(("ops",), _counter(t, ("ops",)) + 1)
    return "claim", free


def pick(rnd: random.Random) -> str:
    return rnd.choice(["transfer", "transfer", "claim"])


def rng(seed: int) -> random.Random:
    return random.Random(seed)  # noqa: S311


class Recorder:
    """What one worker committed. Process workers also fsync each commit to ``path``."""

    def __init__(self, path: Path | None = None) -> None:
        self.f = path.open("a", buffering=1) if path else None
        self.commits = 0
        self.claims: list[int] = []
        self.retries = 0
        self.snaps = 0

    def commit(self, kind: str, slot) -> None:
        self.commits += 1
        if kind == "claim" and slot is not False:
            self.claims.append(slot)
        if self.f:
            self.f.write(f"commit {kind} {slot}\n")
            self.f.flush()
            os.fsync(self.f.fileno())

    def close(self) -> None:
        if self.f:
            self.f.close()


class StillConflictingError(AssertionError):
    pass


def sync_worker(
    storage, wid: str, deadline: float, rnd: random.Random, rec: Recorder, hold: float
) -> None:
    def pause() -> None:
        time.sleep(hold)

    while time.monotonic() < deadline:
        if rnd.random() < 0.25:
            with storage.snapshot() as snap:
                check_snapshot(snap)
            rec.snaps += 1
            continue
        kind = pick(rnd)
        tries = 0
        while True:
            try:
                with storage.transaction() as t:
                    res = transfer(t, rnd, pause) if kind == "transfer" else claim(t, wid, pause)
                break
            except RETRYABLE:
                tries += 1
                if time.monotonic() > deadline + 5:
                    raise StillConflictingError(
                        f"{wid}: {tries} retries past the deadline"
                    ) from None
                time.sleep(min(0.0002 * tries, 0.01))
        rec.retries += tries
        rec.commit(*res)


async def coro_worker(storage, wid: str, deadline: float, rnd: random.Random, rec: Recorder):
    async def body() -> tuple:
        slot = await storage.areserve_write_slot()
        t = storage.begin_transaction(write_slot=slot)
        try:
            kind = pick(rnd)
            if kind == "transfer":
                a, b = rnd.sample(range(ACCOUNTS), 2)
                va, vb = t.get(_acct(a)), t.get(_acct(b))
                await asyncio.sleep(0)  # yield mid transaction
                amt = rnd.randint(1, 50)
                t.put(_acct(a), va - amt)
                t.put(_acct(b), vb + amt)
                t.put(("ops",), _counter(t, ("ops",)) + 1)
                res = (kind, None)
            else:
                taken = {k[1] for k in t.scan(slots_scan()).keys()}
                free = next((j for j in range(NSLOTS) if j not in taken), None)
                await asyncio.sleep(0)
                if free is None:
                    res = (kind, False)
                else:
                    assert is_empty(t.get(_slot(free)))
                    t.put(_slot(free), wid)
                    t.put(("claims",), _counter(t, ("claims",)) + 1)
                    t.put(("ops",), _counter(t, ("ops",)) + 1)
                    res = (kind, free)
            t.commit()
            return res
        except BaseException:
            if not t.is_closed:
                t.abort()
            raise

    while time.monotonic() < deadline:
        if rnd.random() < 0.25:
            with storage.snapshot() as snap:
                check_snapshot(snap)
            rec.snaps += 1
            await asyncio.sleep(0)
            continue
        tries = 0
        while True:
            try:
                res = await body()
                break
            except RETRYABLE:
                tries += 1
                if time.monotonic() > deadline + 5:
                    raise StillConflictingError(
                        f"{wid}: {tries} retries past the deadline"
                    ) from None
                await asyncio.sleep(min(0.0002 * tries, 0.01))
        rec.retries += tries
        rec.commit(*res)


def run_coros(storage, n: int, deadline: float, seed: int, recs: list[Recorder]) -> None:
    async def main() -> None:
        await asyncio.gather(
            *[
                coro_worker(storage, f"c{i}", deadline, rng(seed * 100 + i), recs[i])
                for i in range(n)
            ]
        )

    asyncio.run(main())


# Process worker. Module level so spawn can import it.
def proc_worker(
    backend: str,
    root: str,
    wid: str,
    deadline_epoch: float,
    out: str,
    seed: int,
    busy: float | None,
    hold: float,
    ready,
) -> None:
    st = BACKENDS[backend].make(Path(root), lock_timeout=busy)
    st.open()
    rec = Recorder(Path(out))
    deadline = time.monotonic() + max(0.0, deadline_epoch - time.time())
    try:
        ready.set()
        sync_worker(st, wid, deadline, rng(seed), rec, hold)
    finally:
        rec.close()
        st.close()


# ---------------------------------------------------------------------------- the run


@dataclass
class Procs:
    """Spawned writer processes, each with a ready event and a commit log."""

    ctx: Any
    spec: BackendSpec
    root: Path
    load: Load
    deadline_epoch: float
    seq: int = 0
    alive: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self.outdir = self.root / "out"
        self.outdir.mkdir(parents=True, exist_ok=True)

    def spawn(self) -> None:
        wid = f"p{self.seq}"
        ready = self.ctx.Event()
        p = self.ctx.Process(
            target=proc_worker,
            args=(
                self.spec.name,
                str(self.root),
                wid,
                self.deadline_epoch,
                str(self.outdir / f"{wid}.log"),
                self.load.seed * 1000 + self.seq,
                self.load.busy,
                self.load.pause,
                ready,
            ),
            daemon=True,
        )
        p.start()
        self.seq += 1
        self.alive.append((p, ready))

    def kill_one(self, rnd: random.Random) -> bool:
        """SIGKILL a random process that has opened the storage, spawn a replacement."""
        candidates = [
            i for i, (p, ready) in enumerate(self.alive) if p.is_alive() and ready.is_set()
        ]
        if not candidates:
            return False
        i = rnd.choice(candidates)
        p, _ = self.alive[i]
        os.kill(p.pid, signal.SIGKILL)
        p.join(10)
        del self.alive[i]
        self.spawn()
        return True

    def failed(self) -> list[str]:
        return [
            f"proc {p.pid} exit {p.exitcode}"
            for p, _ in self.alive
            if not p.is_alive() and p.exitcode not in (0, -signal.SIGKILL)
        ]

    def join(self, timeout: float) -> list[str]:
        problems = []
        for p, _ in self.alive:
            p.join(timeout)
            if p.is_alive():
                p.kill()
                problems.append(f"proc {p.pid} stalled")
            elif p.exitcode not in (0, -signal.SIGKILL):
                problems.append(f"proc {p.pid} exit {p.exitcode}")
        return problems

    def recorded(self) -> tuple[int, dict[int, str]]:
        """Commits and claims every process logged, by worker id."""
        commits, claims = 0, {}
        for f in sorted(self.outdir.glob("p*.log")):
            for line in f.read_text().splitlines():
                parts = line.split()
                if parts[:1] != ["commit"]:
                    continue
                commits += 1
                if parts[1] == "claim" and parts[2] not in ("None", "False"):
                    claims[int(parts[2])] = f.stem
        return commits, claims


def run_load(spec: BackendSpec, root: Path, load: Load) -> dict:
    """Run ``load`` against a fresh storage under ``root``, verify, return the stats."""
    procs_n = load.procs if spec.process_model == "multi" else 0
    kills_n = load.kills if procs_n else 0
    st = spec.make(root, lock_timeout=load.busy)
    st.open()
    with st.transaction() as t:
        for i in range(ACCOUNTS):
            t.put(_acct(i), 1000)
        t.put(("ops",), 0)
        t.put(("claims",), 0)

    t0 = time.monotonic()
    deadline = t0 + load.seconds
    rnd = rng(load.seed)
    errors: list[BaseException] = []

    def guarded(fn, *args) -> None:
        try:
            fn(*args)
        except BaseException as e:
            errors.append(e)

    thread_recs = [Recorder() for _ in range(load.threads)]
    coro_recs = [Recorder() for _ in range(load.coros)]
    threads = [
        threading.Thread(
            target=guarded,
            args=(
                sync_worker,
                st,
                f"t{i}",
                deadline,
                rng(load.seed * 10 + i),
                rec,
                load.pause,
            ),
            daemon=True,
            name=f"t{i}",
        )
        for i, rec in enumerate(thread_recs)
    ]
    if load.coros:
        threads.append(
            threading.Thread(
                target=guarded,
                args=(run_coros, st, load.coros, deadline, load.seed, coro_recs),
                daemon=True,
                name="coros",
            )
        )
    procs = Procs(mp.get_context("spawn"), spec, root, load, time.time() + load.seconds)
    for _ in range(procs_n):
        procs.spawn()
    for th in threads:
        th.start()

    kills = 0
    next_kill = t0 + KILL_WARMUP
    kill_every = max(0.2, (load.seconds - KILL_WARMUP) / max(1, kills_n))
    while time.monotonic() < deadline and not errors:
        time.sleep(0.05)
        if kills < kills_n and time.monotonic() >= next_kill and procs.kill_one(rnd):
            kills += 1
            next_kill += kill_every
        if procs.failed():
            break

    for th in threads:
        th.join(60)
    problems = (
        [f"stalled workers: {[th.name for th in threads if th.is_alive()]}"]
        if any(th.is_alive() for th in threads)
        else []
    )
    problems += procs.join(60)
    if errors:
        raise errors[0]
    assert not problems, problems

    # verify after reopen (mem keeps nothing across close: verify live)
    if spec.name != "mem":
        st.close()
        st = spec.make(root, lock_timeout=load.busy)
        st.open()
    recorded_commits = sum(r.commits for r in thread_recs + coro_recs)
    recorded_claims: dict[int, str] = {}
    for prefix, recs in (("t", thread_recs), ("c", coro_recs)):
        for i, r in enumerate(recs):
            for s in r.claims:
                recorded_claims[s] = f"{prefix}{i}"
    proc_commits, proc_claims = procs.recorded()
    recorded_commits += proc_commits
    recorded_claims.update(proc_claims)
    try:
        with st.snapshot() as snap:
            check_snapshot(snap)
            ops = snap.get(("ops",))
            rows = dict(snap.scan(slots_scan()).items())
            claims = snap.get(("claims",))
    finally:
        st.close()

    # A killed process may have committed after its last log line, never before.
    if kills:
        assert ops >= recorded_commits, f"ops counter {ops} < recorded commits {recorded_commits}"
        assert len(rows) >= len(recorded_claims), "fewer slot rows than recorded claims"
    else:
        assert ops == recorded_commits, f"ops counter {ops} != recorded commits {recorded_commits}"
        assert len(rows) == len(recorded_claims), "slot rows != recorded claims"
    lost = [
        (s, w, rows.get(_slot(s))) for s, w in recorded_claims.items() if rows.get(_slot(s)) != w
    ]
    assert not lost, f"recorded claims missing or overwritten (slot, recorder, stored): {lost[:5]}"
    idx = sorted(k[1] for k in rows)
    assert idx == list(range(len(idx))), "slot indices not contiguous: a scan was skewed"

    stats = {
        "commits": ops,
        "claims": claims,
        "retries": sum(r.retries for r in thread_recs + coro_recs),
        "snapshots": sum(r.snaps for r in thread_recs + coro_recs),
        "proc_commits": proc_commits,
        "kills": kills,
        "time": round(time.monotonic() - t0, 1),
    }
    print(f"[s1 randomized {spec.name} {load.name}] seed={load.seed} procs={procs_n} {stats}")
    # A kill needs a ready process in its window; a slow spawn under load is not a storage bug,
    # so the fast tier settles for at least one kill.
    wanted = kills_n if load.name == "slow" else min(1, kills_n)
    assert kills >= wanted, f"only {kills} of {kills_n} kills landed on a ready process"
    assert ops > 0 and stats["snapshots"] > 0
    return stats


# ---------------------------------------------------------------------------- cases


@pytest.mark.parametrize("load", tiers(FAST_MIXED, SLOW_MIXED))
def test_s1_randomized_mixed(backend: BackendSpec, load: Load, tmp_path):
    """Threads, coroutines and processes under random load: no lost update, no lost claim."""
    run_load(backend, tmp_path, load)


@pytest.mark.parametrize("load", tiers(FAST_BUSY, SLOW_BUSY))
def test_s1_randomized_busy_timeout(backend: BackendSpec, load: Load, tmp_path):
    """Same with a 20ms busy timeout: the retry paths run, only retryable classes appear."""
    stats = run_load(backend, tmp_path, load)
    assert stats["retries"] > 0, "a 20ms busy timeout never made a writer retry"


@pytest.mark.parametrize("load", tiers(FAST_KILL, SLOW_KILL))
def test_s1_randomized_kill(multi_process: BackendSpec, load: Load, tmp_path):
    """Writer processes SIGKILLed mid run: the survivors keep going, nothing committed is lost."""
    run_load(multi_process, tmp_path, load)
