# Storage Layer: Contract

What every backend promises, and what the views built on it may rely on. This mirrors the kv contract the backends are tested against (`tests/conformance`). The class docstrings of `SQLiteStorage`, `LMDBStorage`, `RocksDBStorage` and `InMemoryStorage` carry the per backend details.

Backends: sqlite, lmdb, rocksdb, mem. A backend ships only if it passes the conformance suite.

## 1. Transactions

- A transaction is serializable. Point reads and scans inside it see one state, and committed transactions are equivalent to some serial order. No lost update, no write skew, no phantom.
- Read your own writes: a get or scan inside a transaction sees its pending puts and deletes.
- Commit is atomic. A pending multi-key transaction is invisible to other readers, then visible whole. Abort, by `abort()` or by an exception inside the block, leaves no key.
- One writer at a time per storage. Writers in one process queue on an in-process writer slot before they reach the backend. Threads hand off directly, coroutines wait without blocking their loop.
- A second write transaction from the holder of the slot (the same task or thread, a `to_thread` worker below it, or an `asyncio.run` inside it) raises `StorageNestedTransactionError`. It is not retryable.

## 2. Snapshots

- A snapshot binds at open. A commit landing between open and the first read is invisible. Re-reads inside the snapshot ignore later commits.
- A snapshot is never torn. It never shows half of a commit.
- A snapshot never waits on a writer.

## 3. Errors and retry

- Two classes are retryable: `StorageTransactionConflictError` (the backend found a conflict at commit) and `StorageLockTimeoutError` (a wait for the writer slot or a lock ran out). Contention surfaces as one of these two, never as a backend-specific error.
- Everything else is not retryable: `StorageNestedTransactionError`, `StorageClosedError`, `StorageOperationError`.
- A scan resumed after its snapshot, transaction or storage closed raises `StorageClosedError`, never a buffered row.
- The sync wait for the writer slot is `busy_timeout` (default 60s). On an event loop thread a sync begin waits at most `LOOP_THREAD_WAIT_CAP` (50ms) for a holder on another thread or in another process, then raises `StorageLockTimeoutError`. The loop is never stalled for long; the caller retries. A sync begin whose own loop has another task holding the slot raises at once, since that holder could never run to release it. Coroutines use `areserve_write_slot()` and pass the slot to `begin_transaction(write_slot=...)`; that path waits without blocking the loop, including for a lock held by another process on sqlite and lmdb.

## 4. Process model

| backend | process model | notes |
|---|---|---|
| sqlite | multi-process | one writer per file across processes, OS file locks. WAL mode: local disk only, not NFS. A killed writer releases its lock |
| lmdb | multi-process | one writer per environment. A cross-process gate (`flock` on `writer.lock`) bounds the wait by `busy_timeout`. Local disk only: LMDB requires it, and the gate is never relied on across hosts |
| rocksdb | single-process | RocksDB locks its directory. Others open it read-only as a secondary or go through the owning process |
| mem | single-process | one process by nature |

- Open the storage in each process (the `spawn` start method is the supported path). Reusing an open storage across `os.fork` is best effort and documented per backend.
- LMDB thread affinity: LMDB's writer lock belongs to the thread that took it. The storage runs every begin, commit and abort on one writer thread per storage, so a transaction may be begun on one thread and ended on another. Snapshots never touch the writer thread.
- A transaction, snapshot or scan on rocksdb belongs to one thread.
- `close()` ends open contexts. A context whose owner is inside a call is ended after that call returns. The owner's next call raises `StorageClosedError`. A storage closed and reopened does not honor slots reserved before.

## 5. Notifications

See [notify](notify.md). In short: notify after commit, subscribe ready on return, a watch covers its subtree, at most one wake per subscription per key per commit, keyless resync wakes on reconnect and after drops.

## 6. Existence and EMPTY

- `EMPTY` is the absent sentinel. A read of a missing key returns it. It is never stored: every write door, the container's unsafe put and metadata put included, raises `ValueError` on an EMPTY value, nested EMPTY included, before anything is written. A rejected write on an absent view leaves it absent. Delete the key instead.
- `None` is a stored value. An empty container is not absent: primitive and flat views store `{}` and `[]` as values, collection views exist once their marker row does, even with no children.
- Navigation is lazy. Building `a.b.c` through `open_child` touches no storage.
- Existence is one read: `view.container.exists()` reads the marker row. An absent container reads as empty, writes into it create it, subscriptions on it work. A list watch on a non-negative index names a slot and works before the list or the slot exists; a negative index needs the length.
- Writes into an absent container create it.
- A key keeps its kind. Writing a compound value over a primitive key, or a primitive over a compound key, raises `ContainerTypeError` and changes nothing. Delete the key first. Compound over compound replaces.

## 7. Cost

- A point access by key is O(1) for the contract's purposes (one b-tree lookup).
- No write op of a dict view scans or shifts siblings. Delete is a point delete. Dicts keep no stored length: `len()` is a live count of the children's rows, O(children), and skips each child's subtree. `bool()` is one seek.
- Child iteration is O(children), not O(descendants).
- Paging returns exactly `limit`: `keys(after=, limit=)` and `keys_reverse(before=, limit=)` on `LogIndexedDictView` and `FractionalIndexedDictView` yield `limit` keys when that many exist, whether or not the cursor row still exists, and read at most `limit + 2` rows.
- Known exceptions, pinned by tests: `IndexedDictView.__delitem__` rewrites its key list (O(n)) and its `len()` reads a stored length. `ListView` and `ByteArrayView` insert and delete in the middle shift the tail (O(n)). Use `FractionalIndexedDictView` for ordered collections.

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
