# Collections

The `virtuals.collections` package provides the abstract base hierarchy for all collection types in virtuals. It mirrors Python's `collections.abc` pattern — minimal abstract cores with rich default implementations derived from them.

## Hierarchy

```text
ContainerBase               abstract __contains__
IterableBase[V]             abstract __iter__
SizedBase                   abstract __len__
    └─ CollectionBase[V]    combines all three
        │
        ├─ MappingBase[K, V]
        │   └─ MutableMappingBase[K, V]
        │       └─ ReactiveMappingBase[K, V]
        │
        ├─ SequenceBase[V]
        │   └─ MutableSequenceBase[V]
        │       └─ ReactiveSequenceBase[V]
        │
        └─ SetBase[V]
            └─ MutableSetBase[V]
                └─ ReactiveSetBase[V]
```

Each collection type follows a three-tier pattern:

- **Read-only base** — minimal abstract methods, rich derived defaults
- **Mutable base** — adds mutation abstracts, derives convenience methods
- **Reactive base** — adds observation abstracts (`on_change`, `on_child_change`, etc.)

## How It Works

### Minimal abstracts, maximum defaults

Like Python's `collections.abc.MutableMapping`, each base requires only a few abstract methods and derives everything else:

**MappingBase** — implement `__getitem__`, `__iter__`, `__len__`:
```python
# You get for free:
__contains__    # try __getitem__, catch KeyError
keys()          # yield from __iter__
values()        # yield self[key] for each key
items()         # yield (key, self[key]) for each key
get()           # __getitem__ with default fallback
```

**MutableMappingBase** — additionally implement `__setitem__`, `__delitem__`:
```python
# You get for free:
pop()           # __getitem__ + __delitem__
update()        # __setitem__ in a loop
clear()         # __delitem__ in a loop
```

**SequenceBase** — implement `__getitem__`, `__len__`:
```python
# You get for free:
__iter__        # yield self[i] for i in range(len)
__reversed__    # same, reversed
__contains__    # linear scan
index()         # linear search
count()         # linear count
```

**SetBase** — all three from CollectionBase remain abstract:
```python
# You get for free:
isdisjoint()    # any(value in self for value in other)
issubset()      # all(value in other_set for value in self)
issuperset()    # all(value in self for value in other)
__le__, __ge__  # subset/superset operators
__or__, __and__,
__sub__, __xor__ # set algebra operators
```

## Protocols vs Bases

The package contains two kinds of abstractions:

### Atomic protocols (`protocols.py`)

Fine-grained capability checks. Used for isinstance testing and type narrowing:

```python
from virtuals.collections import Subscriptable, is_subscriptable

def read(view):
    if is_subscriptable(view):
        return view["key"]  # type-narrowed
```

| Protocol | Method | Purpose |
|----------|--------|---------|
| `Convertible` | `extract()` | Container → Python value |
| `Initializable` | `store(value)` | Python value → Container |
| `Nestable` | `open_child(addr, view)` | Navigate to child view |
| `Subscriptable` | `__getitem__` | Read by address |
| `Assignable` | `__setitem__` | Write by address |
| `Containable` | `__contains__` | Membership test |
| `Sizeable` | `__len__` | Item count |
| `Deletable` | `__delitem__` | Remove by address |
| `Clearable` | `clear()` | Remove all |
| `Observable` | `on_change()` | Watch all changes |
| `ChildObservable` | `on_child_change(addr)` | Watch specific child |
| `DescendantsObservable` | `on_descendents_change(*pattern)` | Watch pattern-matched descendants |

### Reactive protocols (per collection module)

Structural protocols for type-checking reactive views:

```python
from virtuals.collections import ReactiveMappingProtocol

def bind(view: ReactiveMappingProtocol[str, object]):
    sub = view.on_change()
    sub.bind(callback)
```

| Protocol | Combines |
|----------|----------|
| `ReactiveMappingProtocol[K, V]` | MutableMapping + Observable + ChildObservable |
| `ReactiveSequenceProtocol[V]` | MutableSequence + Observable + ChildObservable |
| `ReactiveSetProtocol[V]` | MutableSet + Observable |

## Relationship to Views

Collections provide the **abstract shape**. Views provide the **concrete implementation** backed by storage:

```text
Collections (abstract)          Views (concrete)
─────────────────────           ────────────────
ReactiveMappingBase   ←───────  DictViewBase → EagerDictView / LazyDictView
ReactiveSequenceBase  ←───────  ListViewBase → EagerListView / LazyListView
ReactiveSetBase       ←───────  SetView
```

Views that support nested containers have two **facets** — eager (reads return Python values) and lazy (reads return child Views). Both share a common base with mutations, keys, and lifecycle. See [Eager vs Lazy Access](eager-lazy.md) for details.

Views compose collection behavior with storage machinery (container access, observation wiring, metadata management) through the mixin bases in `virtuals.view`.
