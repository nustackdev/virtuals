# Layer 1: Storage

## What It Is

A generic key-value store where keys are tuples instead of strings.

```python
# Traditional KV store
db["user:alice:name"] = "Alice"

# KV store
db[("user", "alice", "name")] = "Alice"
```

## Key Properties

### Tuple Keys

Keys are tuples of components: `("users", "alice", "profile")`

- Components can be strings, ints, bytes, etc.
- Tuples are immutable and hashable

### Lexicographic Ordering

Keys are ordered like words in a dictionary:

```python
("a",) < ("a", "b") < ("a", "b", "c") < ("a", "c") < ("b",)
```

This ordering makes a prefix range query one seek.

### Flat Storage

All keys are equal. No concept of hierarchy:

- `("users",)` and `("users", "alice")` are two keys
- No relationship between them at this layer
- No concept of "parent", "child", "container", "value"

### Backend Agnostic

Works with any ordered KV store. Shipped backends:

- SQLite (one `WITHOUT ROWID` table, multi-process)
- LMDB (B-tree, multi-process, local disk only)
- RocksDB (LSM-tree, single-process)
- In-memory sorted dict (single-process)

Key size limits are the backend's own. LMDB rejects an encoded key above
511 bytes (the whole site, every segment included), which surfaces as a
`StorageOperationError` on put; SQLite, RocksDB and the in-memory store take
longer keys. Int key segments must fit int64 on every backend that uses the
binary codec. Deeply nested sites and long key segments hit the LMDB limit
first. The per backend table is in [contract](contract.md).

## Operations

Every operation runs inside a context: a `Snapshot` (read only), a `Transaction` (read and write) or a `WriteBatch` (write only). See [context](context.md).

```python
with storage.snapshot() as snap:        # read only, binds at open
    value = snap.get(key)               # Value or EMPTY
    found = snap.exists(key)
    many = snap.multiget([k1, k2])      # dict, missing keys are omitted

with storage.transaction() as tx:       # read and write, commit on exit
    tx.put(key, value)
    tx.delete(key)                      # silent when the key is missing
    tx.get(key)                         # sees the pending put

with storage.batch_write() as batch:    # write only, applied on exit
    batch.put(key, value)
    batch.delete(other)
```

The plain forms are `begin_snapshot()`, `begin_transaction()` and `begin_write_batch()`; a snapshot has `close()`, a transaction `commit()` and `abort()`, a write batch `write()` and `abort()`. A transaction aborted by an exception inside the block leaves no key.

### Scan

```python
from virtuals.tkv.filter import PrefixFilter
from virtuals.tkv.storage import StorageScanOptions

prefix = PrefixFilter(prefix=("users",))
options = StorageScanOptions(
    start=("users",),
    break_filter=prefix,    # stop once past the prefix
    filter=prefix,          # skip rows that do not match
    reverse=False,
    limit=100,
)
for key, value in tx.scan(options).items():
    ...
```

The backend seeks to `start` and streams rows in key order. `break_filter` ends the scan, `filter` skips rows, `max_depth` jumps over whole subtrees so a child listing costs O(children). See [scan](scan.md) and [filtering](filtering.md).

### Transactions

Transactions are serializable, snapshots bind at open and are never torn. One writer at a time per storage; a second writer waits on an in-process slot, or for the file lock across processes, up to `busy_timeout`. Waits that run out raise `StorageLockTimeoutError`, commit conflicts raise `StorageTransactionConflictError`; both are retryable. On an event loop, coroutines reserve the slot with `areserve_write_slot()` so the loop is never blocked. The full rules, including the process model per backend, are in [contract](contract.md). Notifications after commit are in [notify](notify.md).

## Why Tuples?

### Prefix Queries

Want all keys starting with `("users",)`?

```python
# Traditional KV: scan everything, filter in memory
for key in db.all_keys():
    if key.startswith("users:"):
        yield key

# KV: one range scan
prefix = PrefixFilter(prefix=("users",))
scan = tx.scan(StorageScanOptions(start=("users",), break_filter=prefix, filter=prefix))
for key in scan.keys():
    yield key
```

Backend seeks directly to `("users",)` and stops at the first key past the prefix.

### Natural Hierarchical Encoding

Tuples represent paths:

```python
("users", "alice", "profile")
```

But Storage doesn't KNOW this is hierarchical. Higher layers interpret it.

### Type Safety

```python
# Traditional KV: all strings, easy to make mistakes
db["user:alice:age"] = "30"  # Should be int?

# KV: tuples preserve types
db[("users", "alice", "age")] = 30  # Clear
```

## What Storage Does NOT Do

### No Hierarchy

Doesn't know:

- That `("users",)` is "parent" of `("users", "alice")`
- That certain keys are "containers"
- That parent should exist before children

These are Container (Layer 2) concepts.

### No Validation

Doesn't check:

- Parent exists before creating child
- Key types are consistent
- Structure is valid

These are Container (Layer 2) responsibilities.

### No Semantics

Doesn't know:

- What values mean
- What a "DictView" or "ListView" is
- Application logic

These are View (Layer 3) responsibilities.

## Implementation Notes

### Encoding

Storage keys must be encoded to backend format:

```python
tuple → bytes (eg for LMDB)
```

Encoding must preserve lexicographic order:

```python
encode(("a",)) < encode(("a", "b"))  # Must be true
```

Custom encoder used.

### Backend Requirements

Backend must provide:

- Ordered key-value storage
- Range iteration (seek to position, iterate forward/backward)
- Serializable transactions and snapshots that bind at open

Most modern KV stores provide this (LMDB, RocksDB, etc).

### Performance Characteristics

- Point access: O(log N) - tree lookup
- Range scan: O(log N + M) - seek + iterate M results
- multiget: O(K log N) - K lookups

Where N = total keys, M = results, K = batch size.
