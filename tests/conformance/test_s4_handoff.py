"""Contract section 4: a transaction may be ended on another thread, and close() from one thread
never wedges writers elsewhere.

A write transaction is begun on one thread and committed or aborted on
another (nustd opens transactions lazily, so the first touch may land on a
pool worker and the commit on the loop). Afterwards the next writer in
this process, and a peer process on a multi-process backend, must get
through within the lock timeout. close() from the main thread while a
worker thread still holds a transaction must leave peers writable too.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import threading
import time
from pathlib import Path

import pytest

from virtuals.tkv.types import EMPTY


K = ("k",)
LOCK_TIMEOUT = 2.0
PROMPT = 1.0  # a writer behind an ended transaction must get through this fast


def _begin_on_thread(storage):
    box: dict = {}

    def run() -> None:
        try:
            box["tx"] = storage.begin_transaction()
            box["tx"].put(K, 1)
        except BaseException as e:
            box["error"] = e

    t = threading.Thread(target=run)
    t.start()
    t.join(10)
    if "error" in box:
        raise box["error"]
    return box["tx"]


def _write_promptly(storage, key) -> None:
    t0 = time.monotonic()
    with storage.transaction() as t:
        t.put(key, 1)
    took = time.monotonic() - t0
    assert took < PROMPT, f"writer took {took:.2f}s behind an ended transaction"


# Peer process: opens the storage first, writes when told. Module level so spawn can import it.
def proc_peer_write(backend: str, root: str, ready, go, out: str) -> None:
    from tests.conformance.backends import BACKENDS

    st = BACKENDS[backend].make(root, lock_timeout=LOCK_TIMEOUT)
    st.open()
    ready.set()
    go.wait(30)
    t0 = time.monotonic()
    try:
        with st.transaction() as t:
            t.put(("peer",), 1)
        result = "ok"
    except Exception as e:
        result = f"{type(e).__name__}: {str(e)[:80]}"
    took = time.monotonic() - t0
    st.close()
    Path(out).write_text(json.dumps({"result": result, "took": took}))


@pytest.fixture
def peer(multi_process, tmp_path):
    """A spawned peer that has the storage open before the case touches the writer lock."""
    ctx = mp.get_context("spawn")
    ready, go = ctx.Event(), ctx.Event()
    out = tmp_path / "peer.json"
    p = ctx.Process(
        target=proc_peer_write, args=(multi_process.name, str(tmp_path), ready, go, str(out))
    )
    p.start()
    assert ready.wait(20), "peer did not open the storage"

    def write_now() -> dict:
        go.set()
        p.join(20)
        assert p.exitcode == 0, "peer process failed"
        report = json.loads(out.read_text())
        assert report["result"] == "ok", f"peer: {report['result']}"
        assert report["took"] < PROMPT, f"peer took {report['took']:.2f}s"
        return report

    yield write_now
    if p.is_alive():
        p.kill()


@pytest.mark.timeout(20)
@pytest.mark.parametrize("end", ["commit", "abort"])
def test_s4_end_on_another_thread_frees_writers_in_process(backend, tmp_path, end):
    """Begin on a worker thread, end on the main thread, then a new thread writes promptly."""
    st = backend.make(tmp_path, lock_timeout=LOCK_TIMEOUT)
    st.open()
    try:
        tx = _begin_on_thread(st)
        getattr(tx, end)()
        t = threading.Thread(target=_write_promptly, args=(st, ("after",)))
        t.start()
        t.join(10)
        assert not t.is_alive(), "writer thread stalled behind an ended transaction"
        with st.snapshot() as s:
            assert s.get(("after",)) == 1
            assert s.get(K) == (1 if end == "commit" else EMPTY)
    finally:
        st.close()


@pytest.mark.timeout(30)
@pytest.mark.parametrize("end", ["commit", "abort"])
def test_s4_end_on_another_thread_frees_peer_process(multi_process, peer, tmp_path, end):
    """Begin on a worker thread, end on the main thread, then a peer process writes promptly."""
    st = multi_process.make(tmp_path, lock_timeout=LOCK_TIMEOUT)
    st.open()
    try:
        tx = _begin_on_thread(st)
        getattr(tx, end)()
        peer()
        with st.snapshot() as s:
            assert s.get(("peer",)) == 1
    finally:
        st.close()


@pytest.mark.timeout(30)
def test_s4_close_with_live_worker_transaction_frees_peer_process(multi_process, peer, tmp_path):
    """close() on the main thread while a worker thread holds a transaction: a peer still writes."""
    st = multi_process.make(tmp_path, lock_timeout=LOCK_TIMEOUT)
    st.open()
    begun, keep = threading.Event(), threading.Event()
    box: dict = {}

    def worker() -> None:
        try:
            box["tx"] = st.begin_transaction()
            box["tx"].put(K, 1)
        except BaseException as e:
            box["error"] = e
        begun.set()
        keep.wait(20)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    assert begun.wait(10)
    try:
        if "error" in box:
            raise box["error"]
        t0 = time.monotonic()
        st.close()
        assert time.monotonic() - t0 < PROMPT, "close() stalled behind the worker's transaction"
        peer()
    finally:
        keep.set()
    st = multi_process.make(tmp_path)
    st.open()
    try:
        with st.snapshot() as s:
            assert s.get(("peer",)) == 1
            assert s.get(K) is EMPTY
    finally:
        st.close()
