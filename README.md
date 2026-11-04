# virtuals

Native Python dicts, lists and sets that live in a KV store.

[![PyPI - Python Version](https://img.shields.io/badge/python-%3E%3D%203.10-blue)](https://pypi.org/project/virtuals-py/)
[![PyPI Package](https://img.shields.io/pypi/v/virtuals-py?color=yellow)](https://pypi.org/project/virtuals-py/)
[![PyPI Downloads](https://img.shields.io/pypi/dw/virtuals-py?color=green)](https://pypi.org/project/virtuals-py/)
[![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](https://opensource.org/licenses/Apache-2.0)

## About

Dicts, lists, sets and ordered collections that look and feel like native Python, but sit in a KV store.
Each one is a lazy view over an ordered key-value store: reads stream in from disk on access, writes land as rows, and a nested dict of a billion entries costs nothing until you touch it.

Current backends: SQLite, LMDB, RocksDB, in-memory.

Notificaitons and observabiltiy, serializable transactions, real snapshots.


```bash
pip install virtuals-py
pip install "virtuals-py[lmdb]"     # LMDB backend
pip install "virtuals-py[rocksdb]"  # RocksDB backend
pip install "virtuals-py[redis]"    # cross-process notifications
```

## A quick look

```python
from virtuals import Navigator
from virtuals.codecs import BinaryCodec
from virtuals.storages.mem import InMemoryStorage
from virtuals.views import EagerDictView

storage = InMemoryStorage(codec=BinaryCodec())
storage.open()
nav = Navigator(storage)

with storage.transaction() as tx:
    users = nav.root(tx).open_child("users", EagerDictView)
    users["alice"] = {"name": "Alice", "age": 30, "tags": ["admin"]}
    users["bob"] = {"name": "Bob", "age": 25, "tags": []}

    for user_id, profile in users.items():
        print(f"{user_id}: {profile['name']}")

# Under the hood: flat rows with tuple keys, under the "/" root
# ("/", "users", "alice", "name")    -> "Alice"
# ("/", "users", "alice", "age")     -> 30
# ("/", "users", "alice", "tags", 0) -> "admin"
```

Swap `InMemoryStorage` for `SQLiteStorage`, `LMDBStorage` or `RocksDBStorage` and nothing else changes. The data is now on disk, shared between processes, and survives a restart.

## What you get

- **Native shapes.** `[]`, `in`, `len`, `del`, `items()`, `append`, set algebra. Nested values are stored as nested structure and read back with their types.
- **Lazy all the way down.** Opening `a.b.c` touches no storage. A read fetches exactly the rows it needs.
- **Many structures, one store.** Dict, list, set, frozenset, tuple, bytearray, insertion-ordered dict with paging, fractional-indexed ordered collection with O(1) insert and move, int-keyed map with range sampling. Mix them freely under one root.
- **Real transactions.** Serializable, read your own writes, atomic commit, abort on exception. Snapshots bind at open and are never torn.
- **Concurrency that holds up.** One writer at a time, queued across threads, coroutines and processes. Coroutines wait without blocking the loop. Contention surfaces as two retryable errors, never as backend noise.
- **Watch anything.** Subscribe to a view, a child, its children or a pattern of descendants. Wakes arrive after commit, in-process or across processes over Redis or Valkey, with resync on reconnect and no silent drops.
- **Extensible both ways.** Write a new view and it runs on every backend. Write a new backend, run the shipped compliance suites, and every view works on it.

## Eager and lazy

Every view that can hold nested containers has two facets over the same rows. Eager reads return Python values, lazy reads return child views you can keep navigating.

```python
from itertools import islice

with storage.snapshot() as snap:
    users = nav.root(snap).open_child("users", EagerDictView)

    assert users["alice"]["tags"] == ["admin"]       # a plain dict

    alice = users.lazy["alice"]                      # a live view, nothing read yet
    assert alice["name"] == "Alice"                  # one row

    admins = [k for k, v in users.lazy.items() if "admin" in v["tags"]]
    first = list(islice(users.lazy.keys(), 1))
    assert admins == ["alice"] and first == ["alice"]
```

`islice`, comprehensions and generators compose with lazy views directly, so there is no separate slice or window type to learn.

## Ordered collections

`FractionalIndexedDictView` is a collection keyed by id where order is a first-class operation. Insert, move and delete touch a constant number of rows however large the collection grows, and two writers inserting at the same spot both land.

```python
from virtuals.views import EagerFractionalIndexedDictView

with storage.transaction() as tx:
    board = nav.root(tx).open_child("board", EagerFractionalIndexedDictView)
    board.append("design", {"title": "Design"})
    board.append("ship", {"title": "Ship"})
    board.insert_before("ship", "build", {"title": "Build"})
    board.move_first("ship")

    assert list(board) == ["ship", "design", "build"]
    assert board.next_after("ship") == "design"
```

`LogIndexedDictView` keeps insertion order and pages with `keys(after=, limit=)`. `Kh57View` maps sparse ints and samples uniformly from a key range.

## Watching changes

Attach a publisher to the storage and an observer to the same transport. Views build the subscription filter. A watch on a container that does not exist yet still works, and its first write fires it.

```python
from virtuals.observers.mem import InMemoryObserver
from virtuals.publishers.mem import InMemoryPublisher
from virtuals.tkv.transport import InMemoryTransport

transport = InMemoryTransport()
publisher = InMemoryPublisher(transport)
observer = InMemoryObserver(transport)
live = InMemoryStorage(codec=BinaryCodec(), publisher=publisher)
publisher.connect()
observer.connect()
live.open()
live_nav = Navigator(live)

with live.snapshot() as snap:
    watch = live_nav.root(snap).open_child("orders", EagerDictView).on_child_change("o1")

woken = []
observer.subscribe(watch).bind(woken.append)

with live.transaction() as tx:
    live_nav.root(tx).open_child("orders", EagerDictView)["o1"] = {"status": "paid"}
publisher.flush()
observer.flush()
assert woken

publisher.disconnect()
observer.disconnect()
live.close()
```

Swap the in-memory pair for the Redis publisher and observer and the same watch fires in every process. Receivers re-read state on wake, so repeats and resync wakes are harmless.

## Backends

| Backend | Process model | Good for | Install |
| ------- | ------------- | -------- | ------- |
| In-memory | single | tests, caches, ephemeral state | built-in |
| SQLite | multi | persistent state shared by several processes | built-in |
| LMDB | multi | persistent, local disk, fastest reads | `virtuals-py[lmdb]` |
| RocksDB | single | persistent, write heavy, one owning process | `virtuals-py[rocksdb]` |

Every backend passes the same conformance suite: transactions, snapshots, errors, process model, notifications and cost, plus model-based state machines, randomized multi-process load with SIGKILLed writers, and interrupt fuzzing into live commits. A backend ships only if it passes.

## Three layers

Each layer adds one concept and knows nothing about the one above it.

```text
View       structure: dict, list, set, ordered, ...
Container  hierarchy: parents, children, kinds
Storage    flat tuple keys in lexicographic order
```

**Storage** is a tuple-key KV store. Every read and write goes through a context: a transaction, a snapshot or a write batch. A missing key reads as `EMPTY`, never `None`, since `None` is a value.

```python
from virtuals.tkv.types import EMPTY

with storage.transaction() as tx:
    tx.put(("raw", "alice", "name"), "Alice")

with storage.snapshot() as snap:
    assert snap.get(("raw", "alice", "name")) == "Alice"
    assert snap.get(("raw", "nobody")) is EMPTY
```

**Container** reads tuples as paths and keeps the tree honest: a key is a container or a value, never both, and keeps its kind until deleted.

```python
from virtuals import Container

with storage.snapshot() as snap:
    alice = Container(ctx=snap, site=("/", "users", "alice"))
    assert alice.exists()
    assert sorted(alice.iter_child_keys()) == ["age", "name", "tags"]
```

**View** turns a container into a data structure, and the registry maps Python types to views so nested values round-trip.

```python
with storage.transaction() as tx:
    users = nav.root(tx).open_child("users", EagerDictView)
    users["carol"] = {"name": "Carol", "age": 41, "tags": {"ops"}}
    assert users["carol"]["tags"] == {"ops"}
    assert len(users) == 3

storage.close()
```

## Building your own

A view is a class over a container. Subclass the bases in `virtuals.view` and `virtuals.collections` (they follow the `collections.abc` pattern: implement a few abstracts, get the rest), register it, and it works on every backend with observability included.

A backend implements the storage protocol. `virtuals.testing` ships `StorageProtocolCompliance`, `ObserverCompliance` and the codec suites as pytest classes: subclass, provide a fixture, run.

## Part of Nu

Virtuals is the storage engine under [Nu](https://github.com/nustackdev/nu). `nustd.kv` binds Nu refs to virtuals views, so a Nu program reads and writes durable, observable state with the same calls it uses for in-memory state. Virtuals has no Nu dependency and stands on its own.

## Docs

- [philosophy](docs/philosophy.md) and [architecture](docs/architecture.md): the idea and the layers
- [storage contract](docs/storage/contract.md): what every backend promises
- [notifications](docs/storage/notify.md): delivery guarantees and timeouts
- [views](docs/view/overview.md), [eager and lazy](docs/collections/eager-lazy.md), [collections](docs/collections/overview.md)
- [tests](tests/README.md): running the suite and its tiers

## License

Apache-2.0
