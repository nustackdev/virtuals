# storage conformance suite

The kv contract (task-161 `contract-kv.md`) as executable cases. Every case runs once per backend: sqlite, lmdb, rocksdb, mem. A backend variant ships only if it passes here.

## run

```sh
.venv/bin/python -m pytest -q -p no:cacheprovider tests/conformance -rxXs            # all backends
.venv/bin/python -m pytest -q -p no:cacheprovider tests/conformance -rxXs -k "[lmdb]" # one backend
CONFORMANCE_ROUNDS=50 CONFORMANCE_ITERS=50 .venv/bin/python -m pytest -q tests/conformance # soak
```

| env var | default | drives |
|---|---|---|
| `CONFORMANCE_ROUNDS` | 5 | rounds of each race or interleaving |
| `CONFORMANCE_THREADS` | 4 | writer threads in lost update and contention |
| `CONFORMANCE_ITERS` | 10 | increments per thread or process |
| `CONFORMANCE_PROCS` | 3 | writer processes in cross-process lost update |
| `CONFORMANCE_CORO_N` | 10 | coroutine writers in section 4 |

Races are forced, not hoped for. In lost update a pause sits between each read and its dependent write. In write skew, insert race and scan write skew the two racers also meet at a 50ms barrier after reading; on a serializing backend the peer is blocked at begin, the barrier times out and the round runs serially. The atomic commit case shortens the GIL switch interval so readers interleave with each commit.

## layout

- `backends.py`: one spec per backend (factory into a tmp dir, process model, writer slot check)
- `known_failures.py`: the xfail table, one row per (case, backend)
- `helpers.py`: knobs, `RETRYABLE`, retry loops, thread runner, cross-process workers, `CountingCodec`
- `test_s<N>_*.py`: cases, named by contract section

## cases

| case | section | proves |
|---|---|---|
| `test_s1_lost_update_threads` | 1 | concurrent read-modify-write from threads loses no increment |
| `test_s1_lost_update_processes` | 1, 4 | same across processes that each open the storage (multi-process backends only) |
| `test_s1_write_skew_disjoint_keys` | 1 | two txns reading each other's key cannot both act on the stale read |
| `test_s1_insert_race` | 1 | create-if-absent on one key: exactly one creator wins |
| `test_s1_scan_write_skew` | 1 | a decision from a range scan is not skewed by a concurrent insert into the range |
| `test_s1_read_your_own_writes` | 1 | get and scan in a txn see its pending set and delete |
| `test_s1_atomic_commit_pending_invisible` | 1 | a pending multi-key txn is invisible to other readers, then visible whole |
| `test_s1_atomic_commit_abort_leaves_none` | 1 | abort (explicit or by exception) leaves no key |
| `test_s1_nested_transaction_raises_distinct_error` | 1, 3 | a second write txn on the same storage raises `StorageNestedTransactionError`, not a retryable class |
| `test_s1_nested_batch_write_raises_distinct_error` | 1, 3 | a write batch inside an open write txn on the same storage raises `StorageNestedTransactionError` |
| `test_s1_interrupted_writer_frees_storage` | 1, 4 | a txn or batch cancelled or hit by a `BaseException` inside its block does not land, and the next writer (other thread, then same thread) commits within 200ms |
| `test_s2_repeatable_read_in_snapshot` | 2 | a re-read in an open snapshot ignores a concurrent commit |
| `test_s2_snapshot_never_torn` | 2 | a snapshot never shows half of a commit that moves x and y together |
| `test_s2_snapshot_binds_at_open_get` | 2 | a commit between snapshot open and its first get is invisible |
| `test_s2_snapshot_binds_at_open_scan` | 2 | a commit between snapshot open and its first scan is invisible |
| `test_s2_commit_atomic_to_concurrent_snapshots` | 1, 2 | snapshots opened during commits of 200 keys stamped with one version never hold two versions |
| `test_s2_snapshot_never_waits_on_writer` | 2, 4 | with one writer holding a pending write and a second writer waiting at begin, a snapshot opens and reads in under 100ms and sees the pre-transaction value |
| `test_s3_contention_raises_only_retryable` | 3 | in-process writers aborted by conflict or busy raise only conflict or lock timeout |
| `test_s3_cross_process_contention_raises_retryable` | 3, 4 | writer processes past a 10ms lock timeout get the backend's own busy signal, and only as a retryable class (multi-process backends only) |
| `test_s3_closed_storage_error_not_retryable` | 3 | a write to a closed storage raises a non-retryable `StorageError` |
| `test_s4_coroutine_writers_serialize` | 4 | N coroutines doing read, await, write all land, in bounded time |
| `test_s4_coroutine_writers_do_not_block_loop` | 4 | writers holding a write across an await never block the loop for 200ms |
| `test_s7_storage_delete_is_point_delete` | 7 | `txn.delete` of one key handles one key inside the backend at 10 and 1000 siblings |
| `test_s7_storage_delete_time_flat` | 7 | median `txn.delete` time grows under 10x from 100 to 10k siblings |
| `test_s7_view_dict_delete_is_point_delete` | 7 | deleting one dict view key handles as many keys at 10 as at 1000 siblings |

Concurrency cases retry only `RETRYABLE` errors; any other class fails the case. Every case carries a `pytest.mark.timeout`, so a stall fails instead of hanging. Both section 4 cases run twice: `slot` waits in the backend's `areserve_write_slot()` where it has one, `plain` calls `begin_transaction()` directly and retries.

Section 7 counts keys below the transaction API: storages are built with `CountingCodec`, which counts every key the backend encodes or decodes, so a delete that walks the store shows up however it is wrapped. A walk over raw bytes that never decodes is caught by the timing case.

Out of scope: multi-store transactions (one Transaction over several storages, contract section 1) live in nustd, not virtuals, and are not covered here.

## known failures

Strict xfails: an unexpected pass fails the run, so delete the row in the commit that fixes it.

| case | backend | owner |
|---|---|---|
| lost update threads, write skew, insert race, scan write skew | rocksdb, mem | lane K, lane D |
| nested transaction, nested batch write | rocksdb, mem | lane K, lane D |
| repeatable read, never torn, binds at open (get, scan), commit atomic to snapshots | rocksdb, mem | lane K, lane D |
| cross-process contention | lmdb | lmdb writer slot follow-up: a cross-process writer waits inside lmdb unbounded (contract section 4) |
| coroutine writers serialize (slot, plain) | rocksdb, mem | lane K, lane D |
| coroutine writers do not block loop (slot, plain) | rocksdb | lane K |
| view dict delete is point delete | all | lane B |

Not marked: `test_s1_interrupted_writer_frees_storage` fails on sqlite (transaction) and lmdb (transaction, batch write) until the writer slot fix that aborts on `BaseException` merges. The sqlite batch takes its lock only at write, so it holds nothing when interrupted.

On rocksdb and mem the second writer in the never-waits case does not block, and the case still checks only the snapshot. On backends without a writer slot the `slot` and `plain` section 4 variants run the same path.
