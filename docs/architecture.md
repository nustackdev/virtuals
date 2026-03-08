# Virtuals — Architecture

## Overview

Virtuals is a layered data system. Each layer builds on the one below, adding one concept at a time.

```text
┌─────────────────────────────────────┐
│ Layer 3: Views                      │  Data structures (Dict, List, Set, Queue)
├─────────────────────────────────────┤
│ Layer 2: Container                  │  Hierarchical semantics, parent-child
├─────────────────────────────────────┤
│ Layer 1: Storage (TKV)              │  Flat tuple key-value store
│ + Backend: RocksDB / LMDB / etc     │
└─────────────────────────────────────┘
```

## The Layers

### Layer 1: Storage (TKV)

- Generic key-value store where keys are tuples
- No hierarchy, no semantics
- Provides: point access, range scans, transactions
- Example: `get(("users", "alice"))` returns a value, no concept of "parent" or "child"

Lives in `virtuals.tkv`. Defines protocols for storage, codecs, observers, filters.

### Layer 2: Container

- Interprets tuples as hierarchical paths
- Introduces containers (can have children) vs primitives (leaves)
- Enforces rules: parent must exist before children
- Example: `("users",)` is a container, `("users", "alice")` is its child

Lives in `virtuals.container`. Also `virtuals.loc` (key/path/site navigation).

### Layer 3: Views

- Data structure abstractions built on containers
- DictView, ListView, SetView, IndexedDictView, etc.
- Auto-population and extraction of Python objects
- Example: `users["alice"] = {"name": "Alice"}` stores as tree structure

Lives in `virtuals._views` (implementations) and `virtuals.views` (public API).

## Key Design Principles

### Separation of Concerns

Each layer knows ONLY its own concept:

- Storage: doesn't know what "container" means
- Container: doesn't know what "DictView" means
- Views: don't know about application semantics

### One Concept Per Layer

- Layer 1 adds: tuple keys + ordering
- Layer 2 adds: hierarchy + containers
- Layer 3 adds: data structures

### Bottom-Up Composition

Higher layers use lower layer primitives:

- Container uses Storage's `scan()` to implement `list_children()`
- DictView uses Container's operations to implement dict semantics

## Data Flow Example

Storing a user:

```python
# Layer 3 (View)
users["alice"] = {"name": "Alice", "age": 30}

# ↓ uses Layer 2 (Container)
container.create_child_container("alice")
container.set_child_value("name", "Alice")
container.set_child_value("age", 30)

# ↓ uses Layer 1 (Storage)
tx.put(("users", "alice"), <CONTAINER_SENTINEL>)
tx.put(("users", "alice", "name"), "Alice")
tx.put(("users", "alice", "age"), 30)

# ↓ uses Backend
rocksdb.put(encode(("users", "alice")), encode(<SENTINEL>))
rocksdb.put(encode(("users", "alice", "name")), encode("Alice"))
rocksdb.put(encode(("users", "alice", "age")), encode(30))
```

Reading a user:

```python
# Layer 3 (View)
data = users["alice"]

# ↓ uses Layer 2 (Container)
container.list_children()

# ↓ uses Layer 1 (Storage)
scan = tx.scan(ScanOptions(start=("users", "alice"), end=("users", "alice", "￿")))
for _ in scan.items():
    pass
```

## Package Layout

Pattern: `_private/` for implementations, top-level for public re-exports.

```
virtuals/
├── tkv/                    # core protocols (storage, codec, observer, filter, types)
├── _backends/              # private implementations
│   ├── storages/           #   mem, rocksdb, textdb
│   ├── codecs/             #   json, msgpack, pickle, passthrough
│   ├── observers/          #   mem, redis_pubsub
│   └── key_codecs/         #   binary, string
├── storages/               # public alias
├── codecs/                 # public alias
├── observers/              # public alias
├── _views/                 # private view implementations
├── views/                  # public alias
├── view/                   # base view machinery (registry, capability bases)
├── container/              # hierarchical container model
├── loc/                    # location/navigation (key, path, site)
├── types/                  # core types (Empty, NotSet, Value)
├── collections/            # collection bases, protocols, hierarchy
└── testing/                # compliance test suites
```

## Views

| View | What |
| ---- | ---- |
| `DictView` | Dict — keys + nested values |
| `ListView` | Ordered list with index-based access |
| `SetView` | Unique unordered set |
| `IndexedDictView` | Dict with explicit key ordering |
| `FlatDictView` | Dict with primitive-only values |
| `FlatListView` | List with primitive-only values |
| `LightDictView` | Minimal dict (no metadata) |
| `TupleView` | Immutable ordered sequence |
| `FrozensetView` | Immutable set |
| `BytearrayView` | Binary data |

## Backends

| Backend | When | Install |
| ------- | ---- | ------- |
| In-memory | Testing, ephemeral, caching | built-in |
| RocksDB | Persistent, high-performance | `virtuals-py[rocksdb]` |
| TextDB | Human-readable, debugging | built-in |

Adding a backend = implementing ~4 methods.

## Why This Architecture?

**Flexibility**: Each layer can be replaced independently

- Swap LMDB for RocksDB (Backend)
- Add new View types (Layer 3)
- Add application logic on top

**Testability**: Each layer has clear boundaries

- Test Storage without Container logic
- Test Container without View logic
- Test Views in isolation

**Performance**: Lower layers are optimized

- Storage maps directly to backend primitives
- No unnecessary abstractions
- Each layer adds minimal overhead

**Simplicity**: Each layer is simple in isolation

- Storage: just a KV store with tuple keys
- Container: just adds hierarchy concept
- Views: just data structure patterns
