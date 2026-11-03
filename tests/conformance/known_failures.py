"""Known contract failures, one row per (case, backend).

A row turns the case into ``xfail``. ``strict=True`` means an unexpected
pass fails the run, so a fix flips loudly: delete the row in the same commit.
"""

from __future__ import annotations


LMDB_RA = "lmdb follow-up for RA (round 2 audit X5/B3, B11): "

# case name -> backend (or "backend-param" for one parametrization) -> (strict, reason)
KNOWN_FAILURES: dict[str, dict[str, tuple[bool, str]]] = {
    "test_s4_cross_process_sync_begin_on_loop_is_bounded": {
        "lmdb": (True, LMDB_RA + "cap the gate and begin hop wait on a loop thread"),
    },
    "test_s4_backend_declares_process_model": {
        "lmdb": (True, LMDB_RA + "declare process_model = 'multi' on LMDBStorage"),
    },
}
