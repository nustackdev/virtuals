"""Known contract failures, one row per (case, backend).

A row turns the case into ``xfail``. ``strict=True`` means an unexpected
pass fails the run, so a fix flips loudly: delete the row in the same commit.
"""

from __future__ import annotations


LMDB_XPROC = (
    "lmdb cross-process writer wait is unbounded; contract §4; owner: lmdb writer slot follow-up"
)

DROPPED_TXN = (
    "a transaction dropped without commit or abort holds the writer slot until close: "
    "the active set is strong and the transaction has no finalizer; contract §1; owner: {owner} "
    "(weak active set, __weakref__ in the context slots, a __del__ that ends an unended "
    "transaction by lock identity, as rocksdb and mem do)"
)

# case name -> backend -> (strict, reason)
KNOWN_FAILURES: dict[str, dict[str, tuple[bool, str]]] = {
    "test_s3_cross_process_contention_raises_retryable": {
        "lmdb": (True, LMDB_XPROC),
    },
    "test_s1_dropped_transaction_frees_slot": {
        "sqlite": (True, DROPPED_TXN.format(owner="P3 sqlite")),
        "lmdb": (True, DROPPED_TXN.format(owner="P1 lmdb")),
    },
}
