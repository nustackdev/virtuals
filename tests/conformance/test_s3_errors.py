"""Contract section 3: only conflict and lock timeout are retryable."""

from __future__ import annotations

import json
import multiprocessing as mp
import threading
import time

import pytest

from tests.conformance.helpers import PROCS, RETRYABLE, THREADS, proc_contend, run_threads
from virtuals.tkv.storage import StorageError


K = ("counter",)
HOLD = 0.03  # a writer holds its lock this long
LOCK_TIMEOUT = 0.01  # well under HOLD, so waiting writers give up


@pytest.mark.timeout(20)
def test_s3_contention_raises_only_retryable(backend, tmp_path):
    """Writers aborted by conflict or busy raise only RETRYABLE classes.

    The storage is opened with a lock timeout shorter than the time each
    writer holds its lock, so a backend with a timeout knob must surface
    contention. Backends that queue writers raise nothing, which also passes.
    """
    st = backend.make(tmp_path, lock_timeout=LOCK_TIMEOUT)
    st.open()
    try:
        with st.transaction() as t:
            t.put(K, 0)
        errors: list[BaseException] = []
        lock = threading.Lock()

        def writer() -> None:
            for _ in range(3):
                try:
                    with st.transaction() as t:
                        v = t.get(K)
                        t.put(K, v + 1)
                        time.sleep(HOLD)
                except Exception as e:
                    with lock:
                        errors.append(e)

        run_threads(*[writer] * THREADS)
    finally:
        st.close()
    wrong = sorted(
        {f"{type(e).__name__}: {str(e)[:80]}" for e in errors if not isinstance(e, RETRYABLE)}
    )
    assert not wrong, f"contention raised non-retryable errors: {wrong}"


@pytest.mark.timeout(10)
def test_s3_closed_storage_error_not_retryable(backend, tmp_path):
    """Writing to a closed storage raises a StorageError that is not retryable."""
    st = backend.make(tmp_path)
    st.open()
    st.close()
    with pytest.raises(StorageError) as info:
        with st.transaction() as t:
            t.put(K, 1)
    assert not isinstance(info.value, RETRYABLE), type(info.value).__name__


XP_ROUNDS = 5


@pytest.mark.timeout(60)
def test_s3_cross_process_contention_raises_retryable(multi_process, tmp_path):
    """Writer processes contending past a short lock timeout get the backend's busy signal as RETRYABLE.

    Each process opens the storage with LOCK_TIMEOUT and holds every write
    txn for HOLD, so the backend's own cross-process lock must give up and
    surface. The case fails if no error surfaces (the wait was unbounded) or
    if any error is not RETRYABLE. Commits that did land are all counted.
    """
    st = multi_process.make(tmp_path)
    st.open()
    with st.transaction() as t:
        t.put(K, 0)
    st.close()

    ctx = mp.get_context("spawn")
    start = ctx.Barrier(PROCS)
    outs = [tmp_path / f"contend-{i}.json" for i in range(PROCS)]
    procs = [
        ctx.Process(
            target=proc_contend,
            args=(
                multi_process.name,
                str(tmp_path),
                str(out),
                LOCK_TIMEOUT,
                HOLD,
                XP_ROUNDS,
                start,
            ),
        )
        for out in outs
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(50)
    stuck = [p for p in procs if p.is_alive()]
    for p in stuck:
        p.kill()
    assert not stuck, "writer process stalled"
    assert [p.exitcode for p in procs] == [0] * PROCS, "writer process failed"

    results = [json.loads(out.read_text()) for out in outs]
    errors = [e for r in results for e in r["errors"]]
    wrong = sorted({f"{name}: {msg}" for name, retryable, msg in errors if not retryable})
    assert not wrong, f"cross-process contention raised non-retryable errors: {wrong}"
    assert errors, (
        f"{PROCS} processes holding the lock {HOLD * 1000:.0f}ms past a "
        f"{LOCK_TIMEOUT * 1000:.0f}ms lock timeout never got a busy signal"
    )
    st = multi_process.make(tmp_path)
    st.open()
    try:
        with st.snapshot() as s:
            assert s.get(K) == sum(r["commits"] for r in results)
    finally:
        st.close()
