# PV - Polymorphic Views

**KV stores it. PV shapes it.**

PV provides polymorphic views over key-value stores, letting you work with familiar data structures (dicts, lists, sets) while the underlying storage remains flat KV pairs.

## What It Does

```python
from pv import View, Container

# Your flat KV storage becomes structured data
users = View.dict(storage, ("users",))
users["alice"] = {"name": "Alice", "age": 30}
users["bob"] = {"name": "Bob", "age": 25}

# Navigate naturally
for user_id, profile in users.items():
    print(f"{user_id}: {profile['name']}")

# Under the hood: flat KV pairs with tuple keys
# ("users", "alice", "name") -> "Alice"
# ("users", "alice", "age")  -> 30
# ("users", "bob", "name")   -> "Bob"
# ("users", "bob", "age")    -> 25
```

## Three Layers

PV is built in three layers:

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
container.create()  # Ensures parent exists
children = container.children()  # ["name", "age"]
```

### Layer 3: View
Data structure abstractions (dict, list, set) over containers.

```python
users = View.dict(storage, ("users",))
users["alice"] = "data"  # DictView interface
```

## Installation

```bash
pip install pv
```

## Features

- **Polymorphic Views**: Work with dicts, lists, sets over any KV backend
- **Tuple Keys**: Natural hierarchical addressing with lexicographic ordering
- **Backend Agnostic**: Works with any ordered KV store (RocksDB, LMDB, in-memory)
- **Observable**: Watch for changes at any level of the hierarchy
- **Transactional**: Full ACID support when the backend provides it

## Documentation

See the `docs/` directory for detailed documentation on each layer:

- `docs/layers/layer1_storage.md` - Storage layer
- `docs/layers/layer2_container.md` - Container layer
- `docs/layers/layer3_view.md` - View layer

## License

MIT
