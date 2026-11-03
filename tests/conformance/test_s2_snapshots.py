"""Contract section 2: a snapshot is a true point-in-time view."""

from __future__ import annotations

import pytest

from tests.conformance.helpers import ROUNDS, put_committed, run_threads


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
    import threading
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
