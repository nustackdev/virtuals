"""Contract section 4: close() never crashes the process, whatever other threads are doing."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
ROUNDS = 20

# Runs in a subprocess with faulthandler: a backend that crashes here kills
# the child, not the test run. Each round: a worker thread hammers get,
# put and scan on a transaction (or get and scan on a snapshot) while the
# main thread closes the storage at a varied offset. close() may wait for
# the in-flight call or refuse with a StorageError while a call is in
# flight; the worker may finish or see a StorageError. Nothing else.
_CLOSE_BUSY = textwrap.dedent(
    """
    import json, sys, threading, time
    sys.path.insert(0, sys.argv[1])
    from pathlib import Path
    from tests.conformance.backends import BACKENDS
    from virtuals.tkv import StorageScanOptions
    from virtuals.tkv.storage import StorageError

    backend, root, rounds, mode = sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
    outcomes = {}
    for i in range(rounds):
        s = BACKENDS[backend].make(Path(root) / f"r{i}", lock_timeout=2.0)
        s.open()
        with s.transaction() as tx:
            for k in range(300):
                tx.put(("k", k), b"v" * 50)
        go = threading.Event()
        result = []

        def worker():
            ctx = s.begin_snapshot() if mode == "snap" else s.begin_transaction()
            go.set()
            try:
                for r in range(20000):
                    if mode == "txn":
                        ctx.put(("w", r), b"x")
                    ctx.get(("k", r % 300))
                    if r % 50 == 0:
                        list(ctx.scan(StorageScanOptions(limit=40)).keys())
                result.append("finished")
            except StorageError as e:
                result.append("storage-error")
            except BaseException as e:
                result.append(type(e).__name__)

        th = threading.Thread(target=worker, daemon=True)
        th.start()
        go.wait()
        time.sleep(0.0005 * (i % 7))
        t0 = time.perf_counter()
        try:
            s.close()
            closed = "closed"
        except StorageError:
            closed = "refused"
        took = time.perf_counter() - t0
        th.join()  # bounded by the test timeout; a close under a live worker is not a verdict
        while closed == "refused":
            try:
                s.close()
                break
            except StorageError:
                time.sleep(0.01)
        key = closed + "/" + (result[0] if result else "hung") + (" slow-close" if took > 1 else "")
        outcomes[key] = outcomes.get(key, 0) + 1
        s = BACKENDS[backend].make(Path(root) / f"r{i}")
        s.open()
        s.begin_transaction().commit()
        s.close()
    print(json.dumps(outcomes))
    """
)

OK_CLOSE = {"closed", "refused"}
OK_WORKER = {"finished", "storage-error"}


@pytest.mark.timeout(120)
@pytest.mark.parametrize("mode", ["txn", "snap"])
def test_s4_close_during_in_flight_call_never_crashes(backend, tmp_path, mode):
    """close() racing a thread inside a call on a transaction or snapshot never crashes."""
    out = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-X",
            "faulthandler",
            "-c",
            _CLOSE_BUSY,
            str(ROOT),
            backend.name,
            str(tmp_path),
            str(ROUNDS),
            mode,
        ],
        capture_output=True,
        text=True,
        timeout=110,
        check=False,
    )
    assert out.returncode == 0, (
        f"{backend.name}-{mode}: child died (rc={out.returncode}):\n{out.stderr[-3000:]}"
    )
    outcomes = json.loads(out.stdout.strip().splitlines()[-1])
    bad = {
        k: v
        for k, v in outcomes.items()
        if k.split("/")[0] not in OK_CLOSE
        or k.split("/")[1].split(" ")[0] not in OK_WORKER
        or "slow-close" in k
    }
    assert not bad, f"{backend.name}-{mode}: close() under an in-flight call: {outcomes}"
    assert sum(outcomes.values()) == ROUNDS
