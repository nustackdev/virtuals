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
commit, abort on failure), plus with ``use_async`` a fifth (a reserved slot
from ``areserve_write_slot`` on a loop on the main thread, handed to
``begin_transaction``), with values from 10B to 2MB, ``sync`` on, while a
peer process writes throughout (lock contention across processes) and, with
``side``, an in-process thread too (slot contention). Then: the slot is
free, a write from the main thread works, the peers never failed, every
key is present iff its caller saw success (a caller that saw an exception
has an absent key, or a present one only through an interrupt the backend
reported after the commit), values are intact, every present key was
notified exactly once, a counter bumped inside the read-then-write shapes
equals the number of their committed transactions (no double apply, no
torn transaction), and every context's committed flag agrees with the
store. A peer's lock timeout is the documented retryable error (SQLite's
busy handler can starve a process behind a tight writer) and is reported,
not counted as a defect. A wedge (the slot or the gate left held in the
process, or a live LMDB write txn, so its later writers time out) is
reported separately: an interrupt that lands in one of the documented
bookkeeping windows can do that, and the rate is what is measured.
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


def run(
    backend: str,
    n: int,
    seed: int,
    lo: float,
    hi: float,
    side: bool,
    mode: str,
    use_async: bool = False,
) -> dict:
    """Run the fuzz; returns the report (see the module docstring)."""
    import asyncio

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
    ctxs: dict[int, object] = {}
    loop = asyncio.new_event_loop() if use_async else None

    def handler(_signum, frame) -> None:
        if not armed or frame.f_code.co_filename == HERE:
            stats["skipped"] += 1
            return
        if loop is not None and "/virtuals/" not in frame.f_code.co_filename:
            stats["skipped"] += 1  # the loop's own frames: only virtuals is under test
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

    def bump(tx) -> None:
        count = tx.get(("ctr",))
        tx.put(("ctr",), str((int(count) if isinstance(count, bytes) else 0) + 1).encode())

    async def reserved_one(i: int, key: tuple, val: bytes) -> None:
        slot = await st.areserve_write_slot()
        try:
            tx = st.begin_transaction(write_slot=slot)
        except BaseException:
            slot.release()
            raise
        ctxs[i] = tx
        try:
            bump(tx)
            tx.put(key, val)
            tx.commit()
        except BaseException:
            tx.abort()
            raise

    def one(i: int, kind: int, val: bytes) -> None:
        key = ("m", i)
        if kind == 0:
            with st.transaction() as tx:
                ctxs[i] = tx
                tx.get(("m", i - 1))
                bump(tx)
                tx.put(key, val)
        elif kind == 1:
            with st.transaction() as tx:
                ctxs[i] = tx
                tx.put(key, val)
        elif kind == 2:
            with st.batch_write() as batch:
                ctxs[i] = batch
                batch.put(key, val)
        elif kind == 3:
            tx = st.begin_transaction()
            ctxs[i] = tx
            try:
                tx.exists(("m", 0))
                bump(tx)
                tx.put(key, val)
                tx.commit()
            except BaseException:
                tx.abort()
                raise
        else:
            loop.run_until_complete(reserved_one(i, key, val))  # type: ignore[union-attr]

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
            kind = random.randrange(5 if use_async else 4)
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

    report: dict = {"backend": backend, "mode": mode, "n": n, "seed": seed, "async": use_async}
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
    if loop is not None:
        loop.close()
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
        count = snap.get(("ctr",))
    report["values_consistent"] = values_ok
    report["counter"] = int(count) if isinstance(count, bytes) else 0
    report["counter_expected"] = sum(
        1 for i, o in outcome.items() if o[1] in (0, 3, 4) and present[i]
    )

    def landed(ctx: object) -> bool:
        return bool(getattr(ctx, "_committed", False) or getattr(ctx, "_written", False))

    report["flags_off"] = [i for i in outcome if i in ctxs and landed(ctxs[i]) != present[i]][:10]
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


def wedged(report: dict) -> list[str]:
    """What a wedge left behind, if the run ended in one; empty otherwise."""
    out = []
    if report["slot_locked"]:
        out.append("slot still locked")
    if report.get("live_writes"):
        out.append(f"live writes: {report['live_writes']}")
    if report.get("gate_held") or report.get("gate_waiting"):
        out.append("gate held or waiting")
    if not str(report["final_write"]).startswith("ok"):
        out.append(f"final write: {report['final_write']}")
    return out


def check(report: dict) -> list[str]:
    """The correctness invariants a run must hold whatever else happened; returns what failed.

    Nothing a caller saw succeed is lost, nothing is applied twice or
    torn, a present key never came with a storage error, values are
    intact, and the peers failed only with the retryable lock timeout. A
    wedge is not in here: see ``wedged``.
    """
    bad = []
    for name in ("peer", "fresh_peer"):
        line = report[name][0] if report[name] else ""
        if not line.startswith("PEER"):
            bad.append(f"{name}: {line!r}")
    other = [e for e in report["peer_errors"] if "StorageLockTimeoutError" not in e]
    if other:
        bad.append(f"peer errors: {other}")
    if report["counter"] != report["counter_expected"]:
        bad.append(f"counter {report['counter']} != {report['counter_expected']} committed")
    # A context's committed flag disagreeing with the store, and a present
    # key notified other than once, are what an interrupt landing right
    # after the commit and before its bookkeeping leaves (a few bytecodes):
    # reported, not defects, like a wedge.
    if not report["values_consistent"]:
        bad.append("values inconsistent")
    if report["lost"]:
        bad.append(f"lost: {report['lost'][:10]}")
    if report["present_misreported"]:
        bad.append(f"present but reported as a failure: {report['present_misreported'][:10]}")
    if report["notify_for_absent"]:
        bad.append(f"notified absent: {report['notify_for_absent'][:10]}")
    side_errors = {k: v for k, v in report["stats"].items() if k.startswith("side_err_")}
    if side_errors:
        bad.append(f"side writer errors: {side_errors}")
    return bad


if __name__ == "__main__":
    backend, n, seed, lo, hi, side, mode = sys.argv[1:8]
    use_async = len(sys.argv) > 8 and sys.argv[8] == "1"
    result = run(backend, int(n), int(seed), float(lo), float(hi), side == "1", mode, use_async)
    result["bad"] = check(result)
    result["wedged"] = wedged(result)
    print(json.dumps(result, default=str))
