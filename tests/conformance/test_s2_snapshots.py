"""Contract section 2: a snapshot is a true point-in-time view."""

from __future__ import annotations

import sys
import threading

import pytest

from tests.conformance.helpers import ROUNDS, put_committed, run_threads
from virtuals.tkv import StorageScanOptions
from virtuals.tkv.filter import PrefixFilter


X, Y = ("x",), ("y",)


@pytest.mark.timeout(10)
def test_s2_repeatable_read_in_snapshot(storage):
    """Re-reading a key in an open snapshot returns the same value after a concurrent commit."""
    put_committed(storage, {X: 0})
    changed = []
    for i in range(1, ROUNDS + 1):
        with storage.snapshot() as s:
            first = s.get(X)
            run_threads(lambda i=i: put_committed(storage, {X: i}))
            second = s.get(X)
        if first != second:
            changed.append((first, second))
    assert not changed, f"snapshot re-read changed: {changed}"


@pytest.mark.timeout(10)
def test_s2_snapshot_never_torn(storage):
    """A snapshot never shows half of another txn's commit.

    The writer moves x and y together keeping x + y == 0. The snapshot reads
    x, the commit lands, then the snapshot reads y.
    """
    put_committed(storage, {X: 0, Y: 0})
    torn = []
    for i in range(1, ROUNDS + 1):
        with storage.snapshot() as s:
            x = s.get(X)
            run_threads(lambda i=i: put_committed(storage, {X: i, Y: -i}))
            y = s.get(Y)
        if x + y != 0:
            torn.append((x, y))
    assert not torn, f"torn snapshot reads (x, y): {torn}"


SNAPSHOT_BUDGET = 0.1  # seconds a snapshot open and read may take with writers in flight


@pytest.mark.timeout(10)
def test_s2_snapshot_never_waits_on_writer(storage):
    """A snapshot opened while a writer holds a write txn and a second writer waits never waits.

    Writer A has written x and holds its txn open. Writer B is started on a
    thread and is blocked at begin (or not, on a backend without a writer
    slot). The snapshot must open and read within the budget and see the
    value from before A began.
    """
    import time

    put_committed(storage, {X: 0})
    a_wrote, release_a, b_started = threading.Event(), threading.Event(), threading.Event()

    def writer_a() -> None:
        t = storage.begin_transaction()
        try:
            t.put(X, 1)
            a_wrote.set()
            release_a.wait(5.0)
        finally:
            t.abort()

    def writer_b() -> None:
        b_started.set()
        put_committed(storage, {Y: 1})

    ta = threading.Thread(target=writer_a, daemon=True)
    tb = threading.Thread(target=writer_b, daemon=True)
    try:
        ta.start()
        assert a_wrote.wait(5.0), "writer A never wrote"
        tb.start()
        assert b_started.wait(5.0)
        time.sleep(0.02)  # let B reach its wait at begin
        t0 = time.monotonic()
        with storage.snapshot() as s:
            seen = s.get(X)
        elapsed = time.monotonic() - t0
    finally:
        release_a.set()
        ta.join(5.0)
        tb.join(5.0)
    assert not ta.is_alive() and not tb.is_alive(), "writers stalled"
    assert seen == 0, f"snapshot saw pending write: {seen!r}"
    assert elapsed < SNAPSHOT_BUDGET, f"snapshot took {elapsed * 1000:.0f}ms with writers in flight"


@pytest.mark.timeout(10)
def test_s2_snapshot_binds_at_open_get(storage):
    """A snapshot sees storage as of open: a commit between open and the first get is invisible."""
    put_committed(storage, {X: 0})
    late = []
    for i in range(1, ROUNDS + 1):
        with storage.snapshot() as s:
            run_threads(lambda i=i: put_committed(storage, {X: i}))
            first = s.get(X)
        if first != i - 1:
            late.append((i - 1, first))
    assert not late, f"snapshot bound at first read, not at open (expected, saw): {late}"


def _scan_keys(ctx, prefix):
    return list(
        ctx.scan(StorageScanOptions(start=prefix, break_filter=PrefixFilter(prefix=prefix))).keys()
    )


@pytest.mark.timeout(10)
def test_s2_snapshot_binds_at_open_scan(storage):
    """A commit between snapshot open and the first scan is invisible to that scan."""
    prefix = ("bind",)
    late = []
    for i in range(ROUNDS):
        with storage.snapshot() as s:
            run_threads(lambda i=i: put_committed(storage, {(*prefix, i): i}))
            seen = _scan_keys(s, prefix)
        if len(seen) != i:
            late.append((i, len(seen)))
    assert not late, f"snapshot scan saw later commits (expected keys, saw keys): {late}"


ATOM_KEYS = [("v", i) for i in range(200)]
ATOM_COMMITS = 50
ATOM_READERS = 3
SWITCH = 1e-5  # seconds; a short GIL switch interval interleaves readers with the commit


@pytest.mark.timeout(20)
def test_s2_commit_atomic_to_concurrent_snapshots(storage):
    """Snapshots opened while commits land see every key of a commit, or none of it.

    The writer commits ATOM_COMMITS txns, each stamping all ATOM_KEYS with one
    version. Readers open snapshots back to back during the commits and read
    every key. A snapshot holding two versions saw a commit half applied.
    """
    put_committed(storage, dict.fromkeys(ATOM_KEYS, 0))
    done = threading.Event()
    torn: list = []
    reads = [0] * ATOM_READERS

    def writer() -> None:
        try:
            for v in range(1, ATOM_COMMITS + 1):
                put_committed(storage, dict.fromkeys(ATOM_KEYS, v))
        finally:
            done.set()

    def reader(r: int) -> None:
        while not done.is_set():
            with storage.snapshot() as s:
                versions = {s.get(k) for k in ATOM_KEYS}
            reads[r] += 1
            if len(versions) > 1:
                torn.append(sorted(versions))

    old = sys.getswitchinterval()
    sys.setswitchinterval(SWITCH)
    try:
        run_threads(writer, *(lambda r=r: reader(r) for r in range(ATOM_READERS)))
    finally:
        sys.setswitchinterval(old)
    assert sum(reads) >= ATOM_READERS, "readers never ran during the commits"
    assert not torn, f"{len(torn)}/{sum(reads)} snapshots held several versions, eg {torn[0]}"
