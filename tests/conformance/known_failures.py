"""Known contract failures, one row per (case, backend).

A row turns the case into ``xfail``. ``strict=True`` means an unexpected
pass fails the run, so a fix flips loudly: delete the row in the same commit.
"""

from __future__ import annotations


LMDB_XPROC = (
    "lmdb cross-process writer wait is unbounded; contract §4; owner: lmdb writer slot follow-up"
)
LMDB_XPROC_LOOP = (
    "lmdb begin with a slot waits for the cross-process writer inside lmdb, on the loop; "
    "contract §4; owner: P1 lmdb (audit 06 F7)"
)
LMDB_CLOSE_CRASH = (
    "lmdb close() aborts a txn another thread is inside, segfault (probabilistic, so not "
    "strict); owner: P1 lmdb (audit 06 F2)"
)
LMDB_SCAN_CLOSED = (
    "lmdb scan after close raises StorageOperationError, not StorageClosedError; "
    "owner: P1 lmdb (audit 06 F10)"
)
MEM_SCAN_CLOSED = (
    "mem scan keeps yielding its buffered page after close; owner: P2 mem (audit 06 F10)"
)

DROPPED_TXN = (
    "a transaction dropped without commit or abort holds the writer slot until close: "
    "the active set is strong and the transaction has no finalizer; contract §1; owner: {owner} "
    "(weak active set, __weakref__ in the context slots, a __del__ that ends an unended "
    "transaction by lock identity, as rocksdb and mem do)"
)

# case name -> backend (or "backend-param" for one parametrization) -> (strict, reason)
KNOWN_FAILURES: dict[str, dict[str, tuple[bool, str]]] = {
    "test_s1_dropped_transaction_frees_slot": {
        "lmdb": (True, DROPPED_TXN.format(owner="P1 lmdb")),
    },
    "test_s4_late_release_after_close_and_reopen_frees_nothing": {
        "lmdb": (True, "fixed in P1, release by identity"),
    },
    "test_s4_cross_process_lock_wait_does_not_block_loop": {
        "lmdb": (True, LMDB_XPROC_LOOP),
    },
    "test_s4_close_during_in_flight_call_never_crashes": {
        "lmdb": (False, LMDB_CLOSE_CRASH),
    },
    "test_s3_scan_after_context_close_raises_closed": {
        "lmdb": (True, LMDB_SCAN_CLOSED),
        "mem": (True, MEM_SCAN_CLOSED),
    },
    "test_s3_scan_after_storage_close_raises_closed": {
        "lmdb": (True, LMDB_SCAN_CLOSED),
        "mem": (True, MEM_SCAN_CLOSED),
    },
}
