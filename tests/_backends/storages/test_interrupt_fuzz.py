# ruff: noqa: S603
"""A signal handler raising at random times into sqlite and lmdb writes (see ``_interrupt_fuzz``).

Each run is its own process, so a wedged storage never wedges the suite.
The short runs are the smoke; the long ones carry the ``slow`` marker. A
run that ends wedged (an interrupt in one of the documented bookkeeping
windows) still has to be correct about what landed; the wedge rate is
measured from the reports, not asserted.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


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
    assert report["interrupts"] > 3, report  # few land in virtuals frames in a short run
    assert report["bad"] == [], report  # a wedge (report["wedged"]) is measured, not asserted


@pytest.mark.slow
@pytest.mark.timeout(2400)
@pytest.mark.parametrize("mode", ["base", "exc"])
@pytest.mark.parametrize("backend", ["lmdb", "sqlite"])
def test_interrupts_never_wedge_or_misreport_long(backend: str, mode: str) -> None:
    """Three seeds of 2000 with the side writer, two with the async shape.

    The interrupt count scales with the box (713 to 896 per run on red),
    so a quarter of the transactions is the floor.
    """
    for seed, use_async in ((1, False), (2, True), (3, True)):
        report = _fuzz(backend, 2000, seed, True, mode, timeout=900, use_async=use_async)
        assert report["interrupts"] > 2000 / 4, report
        assert report["bad"] == [], report  # a wedge (report["wedged"]) is measured, not asserted
