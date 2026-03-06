# Virtuals

Virtual Python collections over any storage.

Dict, list, set, indexed dict, tree — they look and feel like native Python, but they don't physically exist as in-memory collections. They're virtual: lazy views that compose data structure logic over flat tuple-key storage. Any backend that implements the storage protocol gets every data structure for free.

Like SQLAlchemy for Python collections. No SQL, no specific backend. Define your structure, plug in a store.

PyPI: `virtuals-py` | Import: `virtuals`

## What It Does

```python
from virtuals import View, Container
from virtuals.storages.mem import InMemoryStorage
from virtuals.codecs import NoOpCodec

storage = InMemoryStorage(codec=NoOpCodec())
storage.open()

with storage.transaction() as tx:
    users = DictView.open_root(tx)
    users["alice"] = {"name": "Alice", "age": 30}
    users["bob"] = {"name": "Bob", "age": 25}

    # Navigate naturally
    for user_id, profile in users.items():
        print(f"{user_id}: {profile['name']}")

# Under the hood: flat KV pairs with tuple keys
# ("users", "alice", "name") -> "Alice"
# ("users", "alice", "age")  -> 30
```

## Three Layers

### Layer 1: Storage
Generic tuple-key KV store with lexicographic ordering.

```python
storage.put(("users", "alice", "name"), "Alice")
value = storage.get(("users", "alice", "name"))
```

### Layer 2: Container
Hierarchy and parent-child relationships over flat keys.

```python
container = Container(storage, ("users", "alice"))
container.create()
children = container.children()  # ["name", "age"]
```

### Layer 3: View
Data structure abstractions (dict, list, set) over containers.

```python
users = DictView.open_root(tx)
users["alice"] = "data"
```

## Installation

```bash
pip install virtuals-py
pip install virtuals-py[rocksdb]  # with RocksDB backend
```

## Features

- **Virtual collections**: Work with dicts, lists, sets over any KV backend
- **Tuple keys**: Natural hierarchical addressing with lexicographic ordering
- **Backend agnostic**: Works with any ordered KV store (RocksDB, LMDB, in-memory)
- **Observable**: Watch for changes at any level of the hierarchy
- **Transactional**: Full ACID support when the backend provides it
- **Lazy**: Nothing materializes until accessed

## Documentation

See `docs/` for detailed documentation:

- `docs/layers/layer1_storage.md` — Storage layer
- `docs/layers/layer2_container.md` — Container layer
- `docs/layers/layer3_view.md` — View layer
- `docs/general/architecture.md` — Architecture overview
- `docs/general/philosophy.md` — Design philosophy

## License

MIT
