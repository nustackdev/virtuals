# ruff: noqa: S603
"""A signal handler raising at random times into sqlite and lmdb writes (see ``_interrupt_fuzz``).

Each run is its own process, so a wedged storage never wedges the suite.
The short runs are the smoke; the long ones carry the ``slow`` marker.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def _fuzz(backend: str, n: int, seed: int, side: bool, mode: str, timeout: float) -> dict:
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
    ]
    proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=timeout)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.timeout(180)
@pytest.mark.parametrize("mode", ["base", "exc"])
@pytest.mark.parametrize("backend", ["lmdb", "sqlite"])
def test_interrupts_never_wedge_or_misreport(backend: str, mode: str) -> None:
    report = _fuzz(backend, 150, 7, False, mode, timeout=150)
    assert report["interrupts"] > 10, report
    assert report["bad"] == [], report


@pytest.mark.slow
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mode", ["base", "exc"])
@pytest.mark.parametrize("backend", ["lmdb", "sqlite"])
def test_interrupts_never_wedge_or_misreport_long(backend: str, mode: str) -> None:
    for seed in (1, 2, 3):
        report = _fuzz(backend, 2000, seed, True, mode, timeout=600)
        assert report["interrupts"] > 1000, report
        assert report["bad"] == [], report
