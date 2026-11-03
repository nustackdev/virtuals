"""Known contract failures, one row per (case, backend).

A row turns the case into ``xfail``. ``strict=True`` means an unexpected
pass fails the run, so a fix flips loudly: delete the row in the same commit.
"""

from __future__ import annotations


LANE_K = "fixed by lane K (rocksdb rewrite)"
LMDB_XPROC = (
    "lmdb cross-process writer wait is unbounded; contract §4; owner: lmdb writer slot follow-up"
)

# case name -> backend -> (strict, reason)
KNOWN_FAILURES: dict[str, dict[str, tuple[bool, str]]] = {
    "test_s1_lost_update_threads": {
        "rocksdb": (True, f"reads ignore concurrent commits, last writer wins; {LANE_K}"),
    },
    "test_s1_write_skew_disjoint_keys": {
        "rocksdb": (True, f"read-committed, no read validation; {LANE_K}"),
    },
    "test_s1_insert_race": {
        "rocksdb": (True, f"absent read not validated; {LANE_K}"),
    },
    "test_s1_scan_write_skew": {
        "rocksdb": (True, f"scanned range not protected from inserts; {LANE_K}"),
    },
    "test_s1_nested_transaction_raises_distinct_error": {
        "rocksdb": (True, f"inner write waits on its own lock, then lock timeout; {LANE_K}"),
    },
    "test_s1_nested_batch_write_raises_distinct_error": {
        "rocksdb": (True, f"batch waits on the txn's key lock, then a generic txn error; {LANE_K}"),
    },
    "test_s2_repeatable_read_in_snapshot": {
        "rocksdb": (True, f"snapshot reads see later commits; {LANE_K}"),
    },
    "test_s2_snapshot_never_torn": {
        "rocksdb": (True, f"snapshot reads see later commits; {LANE_K}"),
    },
    "test_s2_snapshot_binds_at_open_get": {
        "rocksdb": (True, f"snapshot reads see commits after open; {LANE_K}"),
    },
    "test_s2_snapshot_binds_at_open_scan": {
        "rocksdb": (True, f"snapshot scans see commits after open; {LANE_K}"),
    },
    "test_s2_commit_atomic_to_concurrent_snapshots": {
        "rocksdb": (True, f"snapshot reads see commits landing between its reads; {LANE_K}"),
    },
    "test_s3_cross_process_contention_raises_retryable": {
        "lmdb": (True, LMDB_XPROC),
    },
    "test_s4_coroutine_writers_serialize": {
        "rocksdb": (True, f"no writer slot, coroutines lose updates; {LANE_K}"),
    },
    "test_s4_coroutine_writers_do_not_block_loop": {
        "rocksdb": (True, f"lock wait blocks the event loop thread; {LANE_K}"),
    },
}
