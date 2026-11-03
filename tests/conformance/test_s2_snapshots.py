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
