# storage conformance suite

The kv contract (task-161 `contract-kv.md`) as executable cases. Every case runs once per backend: sqlite, lmdb, rocksdb, mem; section 5 also once per transport. A backend variant ships only if it passes here.

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
- `known_failures.py`: the xfail table, one row per (case, backend), or per (case, backend-param) for one parametrization
- `helpers.py`: knobs, `RETRYABLE`, retry loops, thread runner, cross-process workers, `CountingCodec`
- `test_s<N>_*.py`: cases, named by contract section (`test_s5_writer_slot.py` spans sections 1 and 4: the writer slot across drops, loop threads, close and nesting)
- `test_s8_model.py`, `test_s1_randomized.py`: the model-based tiers (`fast` by default, `slow` with `-m slow`), see `tests/README.md`

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
| `test_s3_scan_after_context_close_raises_closed` | 3 | a scan resumed after its snapshot or transaction closed raises `StorageClosedError`, never a buffered row |
| `test_s3_scan_after_storage_close_raises_closed` | 3 | same after the storage closed |
| `test_s4_coroutine_writers_serialize` | 4 | N coroutines doing read, await, write all land, in bounded time |
| `test_s4_coroutine_writers_do_not_block_loop` | 4 | writers holding a write across an await never block the loop for 200ms |
| `test_s4_cross_process_sync_begin_on_loop_is_bounded` | 4 | with another process holding the write lock, a sync begin or batch write from a coroutine gives up within the in-process cap, not the whole `busy_timeout` (multi-process backends only) |
| `test_s4_backend_declares_process_model` | 4 | every backend class declares the process model its spec claims |
| `test_s1_dropped_transaction_frees_slot` | 1 | a txn dropped without commit or abort (other thread, then same thread) does not hold the slot once collected: the next writer lands within 200ms |
| `test_s1_nested_through_worker_or_nested_loop_raises_distinct_error` | 1, 3 | a write begun in a `to_thread` worker below the slot holder, or inside an `asyncio.run` within a sync txn, raises `StorageNestedTransactionError` at once |
| `test_s4_sync_begin_on_loop_thread_is_bounded` | 4 | a sync `begin_transaction` or `batch.write` from a coroutine while a thread holds the slot gives up within 300ms with the retryable lock timeout, loop gap under 200ms |
| `test_s4_async_writer_lands_next_to_sync_writer_threads` | 4 | an `areserve_write_slot` writer on the loop gets the slot in FIFO order while two sync threads write back to back |
| `test_s4_late_release_after_close_and_reopen_frees_nothing` | 4 | close and reopen while a commit is about to give the slot back, a new writer takes the fresh slot: the old release neither frees it nor raises |
| `test_s4_reserved_slot_does_not_survive_close` | 4 | a slot reserved and never taken is gone after close and reopen: the next writer lands within 200ms, the stale slot is refused with a non-retryable error |
| `test_s4_cross_process_lock_wait_does_not_block_loop` | 4 | a slot writer waiting for a write lock held by another process never blocks the loop for 200ms (multi-process backends only) |
| `test_s4_close_during_in_flight_call_never_crashes` | 4 | close() racing a thread inside a call on a txn or snapshot (20 rounds in a faulthandler subprocess) never crashes: close waits or refuses, the worker finishes or gets a `StorageError` |
| `test_s4_end_on_another_thread_frees_writers_in_process` | 4 | a txn begun on a worker thread and ended on the main thread frees the slot: a new thread writes promptly |
| `test_s4_end_on_another_thread_frees_peer_process` | 4 | same, and a peer process writes promptly (multi-process backends only) |
| `test_s4_close_with_live_worker_transaction_frees_peer_process` | 4 | close() on the main thread while a worker holds a txn: a peer process still writes (multi-process backends only) |
| `test_s5_wake_follows_the_commit` | 5 | a receiver woken by a change reads that commit or a later one |
| `test_s5_nothing_on_rollback` | 5 | abort, an exception in the body and an empty transaction wake nobody |
| `test_s5_write_batch_notifies` | 5 | a write batch's puts and deletes are delivered like a transaction's |
| `test_s5_one_wake_per_subscription_per_commit` | 5 | a key matching two subscriptions with different filters wakes each once |
| `test_s5_subscribe_ready_on_return` | 5 | the very next commit after subscribe() returns is delivered, no window |
| `test_s5_watch_covers_the_subtree` | 5 | a prefix watch gets the row and everything under it, not a sibling whose name only starts the same |
| `test_s5_untyped_child_watch_covers_the_child_subtree` | 5 | the watch a dict view builds for an untyped child fires on a write deep inside the compound value stored there, not on a sibling (once per dict view kind) |
| `test_s5_resync_on_reconnect` | 5 | writes made while the observer is cut off are recovered by one keyless wake (redis transport only) |
| `test_s7_storage_delete_is_point_delete` | 7 | `txn.delete` of one key handles one key inside the backend at 10 and 1000 siblings |
| `test_s7_storage_delete_time_flat` | 7 | median `txn.delete` time grows under 10x from 100 to 10k siblings |
| `test_s7_view_dict_delete_is_point_delete` | 7 | deleting one dict view key handles as many keys at 10 as at 1000 siblings |
| `test_s1_randomized_mixed` | 1, 4 | threads, coroutines and processes under random load keep the bank sum, lose no claim on a scanned range and no ops count; fast and slow tier |
| `test_s1_randomized_busy_timeout` | 1, 3, 4 | same with a 20ms busy timeout: the retry paths run and only retryable classes appear |
| `test_s1_randomized_kill` | 1, 4 | same with writer processes SIGKILLed mid run: survivors keep going, every logged commit is present (multi-process backends only) |
| `test_s8_storage_model` | 1, 2, 3, 4 | a Hypothesis state machine of random transactions, batches, snapshots, drops, reopen, bulk rows, big values and odd keys matches a reference model |

Concurrency cases retry only `RETRYABLE` errors; any other class fails the case. Every case in sections 1 to 4 and 7 carries a `pytest.mark.timeout`, so a stall fails instead of hanging; the section 5 cases wait on `flush()` barriers with their own timeouts, and the process model case is a plain assertion. There is no suite-wide timeout. The two coroutine writer cases of section 4 run twice: `slot` waits in the backend's `areserve_write_slot()` where it has one, `plain` calls `begin_transaction()` directly and retries.

Section 7 counts keys below the transaction API: storages are built with `CountingCodec`, which counts every key the backend encodes or decodes, so a delete that walks the store shows up however it is wrapped. A walk over raw bytes that never decodes is caught by the timing case.

Out of scope: multi-store transactions (one Transaction over several storages, contract section 1) live in nustd, not virtuals, and are not covered here.

## known failures

Strict xfails: an unexpected pass fails the run, so delete the row in the commit that fixes it. `known_failures.py` is the source of truth and is empty as of this commit: every case passes on every backend. A row names the case, the backend (or one parametrization of it), and an owner.

Section 5 runs every case on both transports (in-memory, redis). The resync case is skipped on the in-memory transport, which has no connection to lose. The multi-process cases skip on rocksdb and mem, which declare a single-process model.

On rocksdb and mem the second writer in the never-waits case does not block, and the case still checks only the snapshot. On backends without a writer slot the `slot` and `plain` section 4 variants run the same path.
