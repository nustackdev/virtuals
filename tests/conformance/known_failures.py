"""Known contract failures, one row per (case, backend).

A row turns the case into ``xfail``. ``strict=True`` means an unexpected
pass fails the run, so a fix flips loudly: delete the row in the same commit.
"""

from __future__ import annotations


LMDB_XPROC = (
    "lmdb cross-process writer wait is unbounded; contract §4; owner: lmdb writer slot follow-up"
)

# case name -> backend -> (strict, reason)
KNOWN_FAILURES: dict[str, dict[str, tuple[bool, str]]] = {
    "test_s3_cross_process_contention_raises_retryable": {
        "lmdb": (True, LMDB_XPROC),
    },
}
