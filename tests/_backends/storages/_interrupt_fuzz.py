# ruff: noqa: S311, S603
"""Interrupt fuzz for the sqlite and lmdb storages: a signal handler raising at random times.

Run as a script (``python -m tests._backends.storages._interrupt_fuzz <backend>
<n> <seed> <lo> <hi> <side> <mode>``) in its own process: the handler must run
on the main thread, and a wedged process must not be the test's. Prints one
JSON object with what the caller saw per transaction, checked against what
landed. ``mode`` is ``base`` (the handler raises a ``BaseException``, what
Ctrl-C does) or ``exc`` (a plain ``Exception``, what a timeout helper's own
class does).

A sender thread delivers SIGUSR1 every ``lo`` to ``hi`` seconds; the handler
raises only when its frame is not this file, so every interrupt lands inside
virtuals or the stdlib below it. The main loop runs ``n`` transactions of
four shapes (read then write, write only, a batch, explicit begin, read,
commit, abort on failure) with values from 10B to 2MB, ``sync`` on, while a
peer process writes throughout (lock contention across processes) and, with
``side``, an in-process thread too (slot contention). Then: the slot is
free, a write from the main thread works, the peers never failed, every
key is present iff its caller saw success (a caller that saw an exception
has an absent key, or a present one only through an interrupt the backend
reported after the commit), values are intact, and every present key was
notified exactly once.
"""

from __future__ import annotations

import collections
import json
import os
import random
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path


HERE = str(Path(__file__).resolve())
SRC = str(Path(HERE).resolve().parents[3] / "src")


class InterruptBase(BaseException):
    pass


class InterruptExc(Exception):  # noqa: N818 (what a handler's own class looks like)
    pass


def _open(backend: str, root: Path, publisher=None):
    from virtuals.codecs import BinaryCodec

    if backend == "lmdb":
        from virtuals._backends.storages.lmdb import LMDBStorage

        st = LMDBStorage(
            path=root / "db",
            codec=BinaryCodec(),
            publisher=publisher,
            map_size=1 << 30,
            sync=True,
            metasync=False,
            busy_timeout=5.0,
        )
    else:
        from virtuals._backends.storages.sqlite import SQLiteStorage

        st = SQLiteStorage(
            path=root / "db.sqlite",
            codec=BinaryCodec(),
            publisher=publisher,
            synchronous="FULL",
            busy_timeout=5.0,
        )
    st.open()
    return st


PEER = """
import sys, time, os
sys.path.insert(0, {src!r})
sys.path.insert(0, {tests!r})
from pathlib import Path
from tests._backends.storages._interrupt_fuzz import _open
st = _open({backend!r}, Path({root!r}))
ok = err = 0
end = time.monotonic() + float(sys.argv[1])
parent = os.getppid()
while time.monotonic() < end and not os.path.exists({stop!r}) and os.getppid() == parent:
    try:
        with st.transaction() as tx:
            tx.put(({prefix!r}, ok), b"y" * 1000)
        ok += 1
    except Exception as e:
        err += 1
        print("peer err", type(e).__name__, e, flush=True)
    time.sleep(0.002)
print("PEER", ok, err, flush=True)
st.close()
"""


def _peer(backend: str, root: Path, prefix: str, seconds: float) -> subprocess.Popen:
    code = PEER.format(
        src=SRC,
        tests=str(Path(HERE).resolve().parents[3]),
        backend=backend,
        root=str(root),
        stop=str(root / f"stop-{prefix}"),
        prefix=prefix,
    )
    return subprocess.Popen(
        [sys.executable, "-c", code, str(seconds)], stdout=subprocess.PIPE, text=True
    )


def _frames(x: BaseException | None) -> str:
    import traceback

    if x is None:
        return "-"
    fs = [f for f in traceback.extract_tb(x.__traceback__) if "/virtuals/" in f.filename]
    tail = " < ".join(f"{Path(f.filename).name}:{f.lineno}:{f.name}" for f in fs[-5:][::-1])
    return f"{type(x).__name__} @ {tail}"


def _note_leak(st, error: BaseException, sigs: dict[str, int], i: int, kind: int) -> None:
    """Record where the interrupt landed when the slot is left held by nobody.

    With a side writer this can be its hold between the acquire and its
    bookkeeping (a false positive): a true leak repeats from some txn on.
    """

    writer = st._writer
    if not (writer._lock.locked() and writer._owner is None) and not getattr(
        st, "_gate_held", False
    ):
        return
    if writer._lock.locked() and writer._owner is not None:
        return

    key = f"txn {i} kind {kind}: {_frames(error)} || context: {_frames(error.__context__)}"
    sigs[key] = sigs.get(key, 0) + 1


def run(backend: str, n: int, seed: int, lo: float, hi: float, side: bool, mode: str) -> dict:
    """Run the fuzz; returns the report (see the module docstring)."""
    random.seed(seed)
    interrupt = InterruptBase if mode == "base" else InterruptExc
    notified: collections.Counter = collections.Counter()

    class Recorder:
        def notify(self, keys) -> None:
            for key in keys:
                notified[key] += 1

    root = Path(tempfile.mkdtemp(prefix="interrupt-fuzz-"))
    st = _open(backend, root, Recorder())
    stats: collections.Counter = collections.Counter()
    armed = False

    def handler(_signum, frame) -> None:
        if not armed or frame.f_code.co_filename == HERE:
            stats["skipped"] += 1
            return
        stats["raised"] += 1
        raise interrupt

    previous = signal.signal(signal.SIGUSR1, handler)
    stop = threading.Event()

    def sender() -> None:
        pid = os.getpid()
        rnd = random.Random(seed + 1)
        while not stop.is_set():
            time.sleep(rnd.uniform(lo, hi))
            os.kill(pid, signal.SIGUSR1)

    def side_writer() -> None:
        j = 0
        while not stop.is_set():
            try:
                with st.transaction() as tx:
                    tx.put(("t", j), b"x" * 100)
                j += 1
            except Exception as e:
                stats["side_err_" + type(e).__name__] += 1
            time.sleep(0.001)
        stats["side_ok"] = j

    peer = _peer(backend, root, "p", 600)
    time.sleep(0.5)
    outcome: dict[int, tuple] = {}

    def one(i: int, kind: int, val: bytes) -> None:
        key = ("m", i)
        if kind == 0:
            with st.transaction() as tx:
                tx.get(("m", i - 1))
                tx.put(key, val)
        elif kind == 1:
            with st.transaction() as tx:
                tx.put(key, val)
        elif kind == 2:
            with st.batch_write() as batch:
                batch.put(key, val)
        else:
            tx = st.begin_transaction()
            try:
                tx.exists(("m", 0))
                tx.put(key, val)
                tx.commit()
            except BaseException:
                tx.abort()
                raise

    trace = bool(os.environ.get("FUZZ_TRACE"))
    leak_sigs: dict[str, int] = {}
    fail_sigs: dict[int, str] = {}
    sender_thread = threading.Thread(target=sender, daemon=True)
    sender_thread.start()
    side_thread = threading.Thread(target=side_writer, daemon=True)
    if side:
        side_thread.start()
    t0 = time.monotonic()
    try:
        for i in range(n):
            kind = random.randrange(4)
            size = random.choice([10, 1000, 100_000, 2_000_000])
            val = bytes([i % 251]) * size
            t1 = time.monotonic()
            try:
                armed = True
                one(i, kind, val)
                armed = False
                outcome[i] = ("ok", kind, None)
            except BaseException as e:
                armed = False
                name = type(e).__name__
                if e.__cause__ is not None:
                    name += "<-" + type(e.__cause__).__name__
                outcome[i] = ("fail", kind, name)
                fail_sigs[i] = _frames(e) + " || context: " + _frames(e.__context__)
                _note_leak(st, e, leak_sigs, i, kind)
            if trace:
                took = time.monotonic() - t1
                print(
                    f"{i} kind={kind} size={size} {took:.3f}s {outcome[i][2]}",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        armed = False
        stop.set()
        sender_thread.join(10)  # no signal after the handler is gone
        signal.signal(signal.SIGUSR1, previous)
    elapsed = time.monotonic() - t0
    if side:
        side_thread.join(10)
    time.sleep(0.3)

    report: dict = {"backend": backend, "mode": mode, "n": n, "seed": seed}
    report["elapsed"] = round(elapsed, 1)
    report["interrupts"] = stats["raised"]
    report["ok"] = sum(1 for o in outcome.values() if o[0] == "ok")
    report["slot_locked"] = st._writer.lock.locked()
    report["slot_owner"] = repr(st._writer._owner)
    report["slot_sync_waiting"] = st._writer._sync_waiting
    if backend == "lmdb":
        report["live_writes"] = len(st._live_writes)
        report["gate_held"] = st._gate_held
        report["gate_waiting"] = st._gate._waiting
    try:
        t1 = time.monotonic()
        with st.transaction() as tx:
            tx.put(("final",), b"1")
        report["final_write"] = f"ok {time.monotonic() - t1:.3f}s"
    except Exception as e:
        report["final_write"] = f"FAIL {type(e).__name__}: {e}"
        import faulthandler

        faulthandler.dump_traceback(all_threads=True)  # to stderr: who holds what
    (root / "stop-p").touch()
    out, _ = peer.communicate(timeout=60)
    report["peer"] = out.strip().splitlines()[-1:]
    report["peer_errors"] = [line for line in out.splitlines() if line.startswith("peer err")][:5]
    fresh = _peer(backend, root, "q", 2.0)
    out, _ = fresh.communicate(timeout=60)
    report["fresh_peer"] = out.strip().splitlines()[-1:]

    with st.snapshot() as snap:
        present = {i: snap.exists(("m", i)) for i in range(n)}
        values_ok = all(
            snap.get(("m", i)) == bytes([i % 251]) * len(snap.get(("m", i)))
            for i in range(n)
            if present[i]
        )
    report["values_consistent"] = values_ok
    report["lost"] = [i for i, o in outcome.items() if o[0] == "ok" and not present[i]]
    # A present key whose caller saw an exception: only the interrupt itself,
    # raised after the commit was reported (never a storage error).
    interrupt_names = {InterruptBase.__name__, InterruptExc.__name__}
    report["present_after_error"] = collections.Counter(
        o[2] for i, o in outcome.items() if o[0] == "fail" and present[i]
    )
    report["present_misreported"] = [
        i
        for i, o in outcome.items()
        if o[0] == "fail" and present[i] and o[2].split("<-")[0] not in interrupt_names
    ]
    report["fail_kinds"] = collections.Counter(o[2] for o in outcome.values() if o[0] == "fail")
    report["notify_not_once"] = [i for i in range(n) if present[i] and notified[("m", i)] != 1]
    report["notify_not_once_detail"] = {
        i: (notified[("m", i)], outcome[i], fail_sigs.get(i)) for i in report["notify_not_once"][:5]
    }
    report["notify_for_absent"] = [i for i in range(n) if not present[i] and notified[("m", i)]]
    st.close()
    report["stats"] = dict(stats)
    report["leak_sigs"] = leak_sigs
    if leak_sigs:
        first = min(int(k.split()[1].rstrip(":")) for k in leak_sigs)
        report["before_first_leak"] = {
            i: fail_sigs[i] for i in range(first - 6, first + 1) if i in fail_sigs
        }
    return report


def check(report: dict) -> list[str]:
    """The invariants a run must hold; returns what failed."""
    bad = []
    if report["slot_locked"]:
        bad.append("slot still locked")
    if report.get("live_writes"):
        bad.append(f"live writes: {report['live_writes']}")
    if report.get("gate_held") or report.get("gate_waiting"):
        bad.append("gate held or waiting")
    if not str(report["final_write"]).startswith("ok"):
        bad.append(f"final write: {report['final_write']}")
    for name in ("peer", "fresh_peer"):
        line = report[name][0] if report[name] else ""
        if not line.startswith("PEER") or line.split()[-1] != "0" or line.split()[-2] == "0":
            bad.append(f"{name}: {line!r}")
    if not report["values_consistent"]:
        bad.append("values inconsistent")
    if report["lost"]:
        bad.append(f"lost: {report['lost'][:10]}")
    if report["present_misreported"]:
        bad.append(f"present but reported as a failure: {report['present_misreported'][:10]}")
    if report["notify_not_once"]:
        bad.append(f"notified not once: {report['notify_not_once'][:10]}")
    if report["notify_for_absent"]:
        bad.append(f"notified absent: {report['notify_for_absent'][:10]}")
    side_errors = {k: v for k, v in report["stats"].items() if k.startswith("side_err_")}
    if side_errors:
        bad.append(f"side writer errors: {side_errors}")
    return bad


if __name__ == "__main__":
    backend, n, seed, lo, hi, side, mode = sys.argv[1:8]
    result = run(backend, int(n), int(seed), float(lo), float(hi), side == "1", mode)
    result["bad"] = check(result)
    print(json.dumps(result, default=str))
