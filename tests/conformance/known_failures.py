"""Known contract failures, one row per (case, backend).

A row turns the case into ``xfail``. ``strict=True`` means an unexpected
pass fails the run, so a fix flips loudly: delete the row in the same commit.
"""

from __future__ import annotations


MEM_SCAN_CLOSED = (
    "mem scan keeps yielding its buffered page after close; owner: P2 mem (audit 06 F10)"
)

# case name -> backend (or "backend-param" for one parametrization) -> (strict, reason)
KNOWN_FAILURES: dict[str, dict[str, tuple[bool, str]]] = {
    "test_s3_scan_after_context_close_raises_closed": {
        "mem": (True, MEM_SCAN_CLOSED),
    },
    "test_s3_scan_after_storage_close_raises_closed": {
        "mem": (True, MEM_SCAN_CLOSED),
    },
}
