# Storage Layer: Contract

What every backend promises, and what the views built on it may rely on. This mirrors the kv contract the backends are tested against (`tests/conformance`). The class docstrings of `SQLiteStorage`, `LMDBStorage`, `RocksDBStorage` and `InMemoryStorage` carry the per backend details.

Backends: sqlite, lmdb, rocksdb, mem. A backend ships only if it passes the conformance suite.

## 1. Transactions

- A transaction is serializable. Point reads and scans inside it see one state, and committed transactions are equivalent to some serial order. No lost update, no write skew, no phantom.
- Read your own writes: a get or scan inside a transaction sees its pending puts and deletes. This holds for scans started after the write. A scan already in flight may or may not see a write the same transaction makes after it started (lmdb's live cursor does, the other three yield the page they already fetched): unspecified, do not rely on either outcome. Deleting keys a scan has already yielded is safe on every backend.
- Commit is atomic. A pending multi-key transaction is invisible to other readers, then visible whole. Abort, by `abort()` or by an exception inside the block, leaves no key.
- One writer at a time per storage. Writers in one process queue on an in-process writer slot before they reach the backend. Threads hand off directly, coroutines wait without blocking their loop.
- A second write transaction from the holder of the slot (the same task or thread, a `to_thread` worker below it, or an `asyncio.run` inside it) raises `StorageNestedTransactionError`. It is not retryable.

## 2. Snapshots

- A snapshot binds at open. A commit landing between open and the first read is invisible. Re-reads inside the snapshot ignore later commits.
- A snapshot is never torn. It never shows half of a commit.
- A snapshot never waits on a writer.

## 3. Errors and retry

- Two classes are retryable: `StorageTransactionConflictError` (the backend found a conflict at commit, or a snapshot lost its view, as when a rocksdb secondary caught up with the primary; retry by opening a new snapshot) and `StorageLockTimeoutError` (a wait for the writer slot or a lock ran out). Contention surfaces as one of these two, never as a backend-specific error.
- Everything else is not retryable: `StorageNestedTransactionError`, `StorageClosedError`, `StorageOperationError`.
- A scan resumed after its snapshot, transaction or storage closed raises `StorageClosedError`, never a buffered row.
- The sync wait for the writer slot is `busy_timeout` (default 60s). On an event loop thread a sync begin waits at most `LOOP_THREAD_WAIT_CAP` (50ms) for a holder on another thread or in another process, then raises `StorageLockTimeoutError`. The loop is never stalled for long; the caller retries. A sync begin whose own loop has another task holding the slot raises at once, since that holder could never run to release it. Coroutines use `areserve_write_slot()` and pass the slot to `begin_transaction(write_slot=...)`; that path waits without blocking the loop, including for a lock held by another process on sqlite and lmdb. Bound that wait with `busy_timeout`, not `asyncio.wait_for`: on Python 3.10 and 3.11 `wait_for` can drop a slot that was granted as it timed out, which leaves it held.

## 4. Process model

| backend | process model | notes |
|---|---|---|
| sqlite | multi-process | one writer per file across processes, OS file locks. WAL mode: local disk only, not NFS. A killed writer releases its lock |
| lmdb | multi-process | one writer per environment. A cross-process gate (`flock` on `writer.lock`) bounds the wait by `busy_timeout`. Local disk only: LMDB requires it, and the gate is never relied on across hosts |
| rocksdb | single-process | RocksDB locks its directory. Others open it read-only as a secondary or go through the owning process |
| mem | single-process | one process by nature |

- Open the storage in each process (the `spawn` start method is the supported path). Reusing an open storage across `os.fork` is best effort, one rule per backend. sqlite parks the inherited connections and the child opens fresh ones, but if the parent held the write lock at the fork, every write from the child on that file raises `StorageError` for good while reads still work: fork only while no write is in progress. lmdb refuses every use in the child with a clear error. rocksdb refuses every begin in the child. mem resets its locks in fork hooks and keeps working.
- LMDB thread affinity: LMDB's writer lock belongs to the thread that took it. The storage runs every begin, commit and abort on one writer thread per storage, so a transaction may be begun on one thread and ended on another. Snapshots never touch the writer thread.
- LMDB begins lazily: the LMDB write transaction starts at the first `get`, `exists` or `scan` of a write transaction, or at its commit. That first read is where the wait for the writer thread happens, so it can raise the retryable `StorageLockTimeoutError` (after `busy_timeout`, or `LOOP_THREAD_WAIT_CAP` on an event loop thread), not only `begin_transaction`. The transaction stays pending and may be retried or aborted.
- LMDB writer slot: threads compete for the slot's lock and take it the instant it is free, so a thread writing in a loop barges ahead of other threads; a coroutine is handed the slot at a release instead. Threads and coroutines take turns at each release, so a queued coroutine waits at most one thread transaction per coroutine ahead of it.
- An interrupt (a signal handler raising, `KeyboardInterrupt`) that lands while a commit is in flight never changes the commit's outcome or how it is reported, on sqlite and lmdb: the commit is waited out, the transaction ends committed and notified, and the interrupt is then raised as it is, so a caller never sees a failure for a commit that happened and never applies it twice. One that lands before the commit started aborts the transaction and is raised the same way. The waits (for the slot, the gate, a begin) undo themselves when interrupted. The bookkeeping around them is ordered so that the windows are a few bytecodes, but they are not zero: a `KeyboardInterrupt` or a signal handler's exception raised inside a storage call can, in such a window, leave the write slot (on lmdb also the gate, or a begun write transaction) held in a process that keeps running, so every later writer in it times out and peers wait on it. The storage never touches signal handlers, so it cannot defer them; restart the process (or the kernel, in a notebook) when that happens. An interrupt in the few bytecodes right after a commit can also cost that commit its notification, so a subscriber may miss it until its next change or resync; the data is committed either way. Task cancellation never does any of this: a commit has no await point, and the async waits give back what they took.
- A transaction, snapshot or scan on rocksdb belongs to one thread.
- `close()` ends open contexts. A context whose owner is inside a call is ended after that call returns. The owner's next call raises `StorageClosedError`. A storage closed and reopened does not honor slots reserved before.

## 5. Notifications

See [notify](notify.md). In short: notify after commit, subscribe ready on return, a watch covers its subtree, at most one wake per subscription per key per commit, keyless resync wakes on reconnect and after drops.

## 6. Existence and EMPTY

- `EMPTY` is the absent sentinel. A read of a missing key returns it. Doors that check: every public view write (`__setitem__`, `store`, `update`, `append`, `prepend`, `insert`, `insert_before`, `insert_after`, `extend`, `add`, `set_primitive`, `set_child_container_as`), and the container's `put_child_primitive`, `put_child_primitive_unsafe` and `put_metadata`. They raise `ValueError` on an EMPTY value, nested EMPTY included, before anything is written; a lazy iterable is materialized first. A rejected write on an absent view leaves it absent. Delete the key instead. The raw storage doors check too: `tx.put` and `batch.put` raise `ValueError` on `EMPTY` itself before anything is written, on every backend; `Empty` is a true singleton, so a copy or a pickle round trip is still `EMPTY` and is refused the same way. These raw doors and the view's private `_unsafe_primitive_write` (one raw put) refuse `EMPTY` at the top only: they do not walk a compound value, so a nested `EMPTY` is caught by the view doors above them, not here.
- Access mimics Python's dict, with `EMPTY` as the absent default. `view[key]` on an absent key raises `KeyError`. `view.get(key)` returns `EMPTY` for an absent key, never `None`, and `view.get(key, default)` returns the given default. `get` returns the default only for an absent key: any other error propagates. `pop` follows `get`. The same holds on an absent view, and a read never creates it. Sequence views (`ListView`, `FlatListView`, `TupleView`, `ByteArrayView`) keep Python's rule: an out-of-range index raises `IndexError`, on an absent view too. Every dict-like view and facet is pinned to this by one test (`tests/views/test_dict_views_hardening.py`, `ACCESS_VIEWS`), and a guard test fails when a view with `get` is exported from `_views` but missing there.
- A key the view cannot hold is absent, the way a dict holds no key of a type it was never given. Dict views take `str` and `int` (bool included, `True` is key `1`); any other key reads as absent (`[]` KeyError, `get` its default, `in` False, `del` as for a missing key) and a write under it raises `TypeError` before anything is written. `Kh57View` holds non-negative ints under 57 bits and reads with Python's int-key semantics: `1.0` and `True` name key `1`, `1.5`, `-1` and `1 << 57` are absent; writes take ints only. An int past int64 is a storage limit, not a view one: it raises `StorageOperationError` on every op.
- `None` is a stored value, never a marker for absent: `view[key]`, `view.get(key)` and `view.get(key, default)` all return `None` for a key that holds it. An empty container is not absent: primitive and flat views store `{}` and `[]` as values, collection views exist once their marker row does, even with no children.
- Navigation is lazy. Building `a.b.c` through `open_child` touches no storage.
- Existence is one read: `view.container.exists()` reads the marker row. An absent container reads as empty, writes into it create it, subscriptions on it work. A `ListView` watch on a non-negative index names a slot and works before the list or the slot exists; a negative index needs the length. `ByteArrayView` still resolves every index against its length.
- Writes into an absent container create it. A write that raises or changes nothing leaves it absent: an out-of-range index on a list or bytearray, `remove` or `discard` of a non-member, `extend([])`, `update({})`, `clear()`.
- A key keeps its kind. Writing a compound value over a primitive key, or a primitive over a compound key, raises `ContainerTypeError` and changes nothing. Delete the key first. Compound over compound replaces.

## 7. Cost

- A point access by key is O(1) for the contract's purposes (one b-tree lookup).
- No write op of a dict view scans or shifts siblings. Delete is a point delete. Dicts keep no stored length: `len()` on DictView, FlatDictView, LightDictView, LogIndexedDictView and FractionalIndexedDictView is a live count of the children's rows, O(children), and skips each child's subtree. `bool()` is one seek. Kh57View and IndexedDictView read a stored count instead (see below).
- Child iteration is O(children), not O(descendants).
- Paging returns exactly `limit`: `keys(after=, limit=)` and `keys_reverse(before=, limit=)` on `LogIndexedDictView` and `FractionalIndexedDictView` yield `limit` keys when that many exist and read at most `limit + 2` rows. `LogIndexedDictView` pages on whether or not the cursor row still exists. `FractionalIndexedDictView` raises `KeyError` on a cursor that was deleted: its position went with the row, so there is nothing honest to resume from.
- Known exceptions, pinned by tests: `IndexedDictView.__delitem__` rewrites its key list (O(n)) and its `len()` reads a stored length. `Kh57View` keeps a stored count too (`len()` and `bool()` are one get). `ListView` and `ByteArrayView` insert and delete in the middle shift the tail (O(n)); `ListView` and `SetView` read a stored count. Use `FractionalIndexedDictView` for ordered collections.

## Key limits

| backend | key limit |
|---|---|
| sqlite | none in practice (a BLOB primary key) |
| lmdb | 511 bytes for the encoded key, the whole site included. A longer key raises `StorageOperationError` on put |
| rocksdb | none in practice |
| mem | none |

Int key segments must fit int64 on every backend that uses the binary codec. Deep nesting and long segments hit the lmdb limit first.

## Backend status

| | sqlite | lmdb | rocksdb | mem |
|---|---|---|---|---|
| serializable transactions | yes | yes | yes (one writer, snapshot validation) | yes |
| real snapshots | yes | yes | yes | yes |
| coroutine writers (writer slot) | yes | yes | yes | yes |
| process model | multi | multi | single | single |
| notifications | any publisher | any publisher | any publisher | any publisher |
