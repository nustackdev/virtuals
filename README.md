# Virtuals

Virtual Python collections over any storage.

Dict, list, set, indexed dict, tree — they look and feel like native Python, but they don't physically exist as in-memory collections. They're virtual: lazy views that compose data structure logic over flat tuple-key storage. Any backend that implements the storage protocol gets every data structure for free.

Like SQLAlchemy for Python collections. No SQL, no specific backend. Define your structure, plug in a store.

PyPI: `virtuals-py` | Import: `virtuals`

## What It Does

```python
from virtuals import Navigator
from virtuals.codecs import BinaryCodec
from virtuals.storages.mem import InMemoryStorage
from virtuals.views import EagerDictView

storage = InMemoryStorage(codec=BinaryCodec())
storage.open()

with storage.transaction() as tx:
    users = Navigator(storage).root(tx).open_child("users", EagerDictView)
    users["alice"] = {"name": "Alice", "age": 30}
    users["bob"] = {"name": "Bob", "age": 25}

    # Navigate naturally
    for user_id, profile in users.items():
        print(f"{user_id}: {profile['name']}")

# Under the hood: flat KV rows with tuple keys, under the "/" root
# ("/", "users", "alice", "name") -> "Alice"
# ("/", "users", "alice", "age")  -> 30
```

## Three Layers

### Layer 1: Storage
Generic tuple-key KV store with lexicographic ordering. Reads and writes go through a context: a transaction, a snapshot or a write batch.

```python
from virtuals.tkv.types import EMPTY

with storage.transaction() as tx:
    tx.put(("raw", "alice", "name"), "Alice")

with storage.snapshot() as snap:
    assert snap.get(("raw", "alice", "name")) == "Alice"
    assert snap.get(("raw", "nobody")) is EMPTY  # absent, never None
```

### Layer 2: Container
Hierarchy and parent-child relationships over flat keys.

```python
from virtuals import Container

with storage.snapshot() as snap:
    alice = Container(ctx=snap, site=("/", "users", "alice"))
    assert alice.exists()
    assert sorted(alice.iter_child_keys()) == ["age", "name"]
```

### Layer 3: View
Data structure abstractions (dict, list, set) over containers.

```python
with storage.transaction() as tx:
    users = Navigator(storage).root(tx).open_child("users", EagerDictView)
    users["carol"] = {"name": "Carol", "age": 41}
    assert len(users) == 3

storage.close()
```

The storage pages under `docs/storage/` carry the contract every backend meets; `tests/README.md` says how to run the suite and its tiers.

## Installation

```bash
pip install virtuals-py
pip install virtuals-py[rocksdb]  # with RocksDB backend
```

## Features

- **Virtual collections**: Work with dicts, lists, sets over any KV backend
- **Tuple keys**: Natural hierarchical addressing with lexicographic ordering
- **Backend agnostic**: Works with any ordered KV store (RocksDB, LMDB, SQLite, in-memory)
- **Observable**: Watch for changes at any level of the hierarchy
- **Transactional**: Full ACID support when the backend provides it
- **Lazy**: Nothing materializes until accessed

## License

Apache-2.0
