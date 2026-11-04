# ruff: noqa: S603
"""A signal handler raising at random times into sqlite and lmdb writes (see ``_interrupt_fuzz``).

Each run is its own process, so a wedged storage never wedges the suite.
The short runs are the smoke; the long ones carry the ``slow`` marker. A
run that ends wedged (an interrupt in one of the documented bookkeeping
windows) still has to be correct about what landed; the wedge rate is
measured from the reports, not asserted. The side writer's lock timeouts
are measured the same way: they come from a loaded box or from a wedge
starving it, and both are retryable to a caller.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests._backends.storages._interrupt_fuzz import SIDE_LOCK_TIMEOUT, check


def _measured(report: dict) -> dict:
    """What to print when a run fails: the verdict and what was measured."""
    return {
        "bad": report["bad"],
        "wedged": report["wedged"],
        "interrupts": report["interrupts"],
        "side_lock_timeouts": report.get("side_lock_timeouts"),
        "stats": report["stats"],
    }


def _fuzz(
    backend: str, n: int, seed: int, side: bool, mode: str, timeout: float, use_async: bool = False
) -> dict:
    root = Path(__file__).resolve().parents[3]
    cmd = [
        sys.executable,
        "-m",
        "tests._backends.storages._interrupt_fuzz",
        backend,
        str(n),
        str(seed),
        "0.0005",
        "0.01",
        "1" if side else "0",
        mode,
        "1" if use_async else "0",
    ]
    proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=timeout)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.timeout(180)
@pytest.mark.parametrize("mode", ["base", "exc"])
@pytest.mark.parametrize("backend", ["lmdb", "sqlite"])
def test_interrupts_never_wedge_or_misreport(backend: str, mode: str) -> None:
    report = _fuzz(backend, 200, 7, False, mode, timeout=150, use_async=True)
    assert report["interrupts"] > 3, _measured(report)  # few land in virtuals frames
    assert report["bad"] == [], _measured(report)  # a wedge is measured, not asserted


@pytest.mark.slow
@pytest.mark.timeout(2400)
@pytest.mark.parametrize("mode", ["base", "exc"])
@pytest.mark.parametrize("backend", ["lmdb", "sqlite"])
def test_interrupts_never_wedge_or_misreport_long(backend: str, mode: str) -> None:
    """Three seeds of 2000 with the side writer, two with the async shape.

    The interrupt count scales with the box and its load (713 to 896 per
    run on an idle red, 493 under a parallel slow tier), so a tenth of the
    transactions is the floor: it only proves interrupts were delivered.
    """
    for seed, use_async in ((1, False), (2, True), (3, True)):
        report = _fuzz(backend, 2000, seed, True, mode, timeout=900, use_async=use_async)
        assert report["interrupts"] > 2000 / 10, _measured(report)
        # a wedge and the side writer's lock timeouts are measured, not asserted
        assert report["bad"] == [], _measured(report)


def _clean_report(**stats: int) -> dict:
    return {
        "peer": ["PEER ok"],
        "fresh_peer": ["PEER ok"],
        "peer_errors": ["StorageLockTimeoutError: busy"],
        "counter": 3,
        "counter_expected": 3,
        "values_consistent": True,
        "lost": [],
        "present_misreported": [],
        "notify_for_absent": [],
        "stats": {"side_ok": 10, **stats},
    }


def test_check_counts_side_writer_lock_timeouts_as_retryable() -> None:
    """A side writer that timed out on the lock is load or a wedge, not a defect.

    Any other side writer error class still fails the run.
    """
    assert check(_clean_report()) == []
    assert check(_clean_report(**{SIDE_LOCK_TIMEOUT: 12})) == []
    bad = check(_clean_report(**{SIDE_LOCK_TIMEOUT: 12, "side_err_StorageOperationError": 1}))
    assert bad == ["side writer errors: {'side_err_StorageOperationError': 1}"], bad
