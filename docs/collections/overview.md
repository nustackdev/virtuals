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

Each base is its `collections.abc` class: the same abstract core, the same mixin methods with the same semantics, and it is registered with that ABC, so `isinstance(view, MutableMapping)` holds for every view built on it. Built-in-only methods (`list +`, `dict |`, `set.union`, `copy`, `sort`) are not on the bases: a view standing in for a built-in adds what it needs.

**MappingBase** (`Mapping`): implement `__getitem__`, `__iter__`, `__len__`:
```python
# You get for free:
__contains__    # try __getitem__, catch KeyError
keys()          # KeysView over the mapping
values()        # ValuesView
items()         # ItemsView
get()           # __getitem__, default None as dict.get
__eq__          # equal to any mapping with the same items; not hashable
```

**MutableMappingBase** (`MutableMapping`): additionally implement `__setitem__`, `__delitem__`:
```python
# You get for free:
pop()           # with an optional default, else KeyError
popitem()       # the last pair in order (dict.popitem); the first without __reversed__
setdefault()    # the value, or store the default and read it back
update()        # a mapping, an object with keys(), or pairs, then kwargs
clear()         # __delitem__ in a loop
```

**SequenceBase** (`Sequence`): implement `__getitem__`, `__len__`:
```python
# You get for free:
__iter__        # yield self[i] for i in range(len)
__reversed__    # same, reversed
__contains__    # linear scan
index()         # linear search, with start and stop
count()         # linear count
```

**MutableSequenceBase** (`MutableSequence`): additionally implement `__setitem__`, `__delitem__`, `insert`:
```python
# You get for free:
append(), extend(), pop(), remove(), reverse(), clear(), __iadd__
```

**SetBase** (`Set`): all three from CollectionBase remain abstract:
```python
# You get for free:
__le__, __lt__, __eq__, __ge__, __gt__   # comparisons; not hashable
__and__, __or__, __sub__, __xor__        # and their reflected forms, a plain set back
isdisjoint()
```

**MutableSetBase** (`MutableSet`): additionally implement `add`, `discard`:
```python
# You get for free:
remove(), pop(), clear(), __ior__, __iand__, __isub__, __ixor__
```

`pop`, `popitem` and `reverse` read items through `_plain_value`, a plain read by default. A view overrides it to read through its eager facet, so a lazy view hands back data, not a view of a removed child.

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
| `ChildObservable` | `on_child_change(addr)` | Watch one child and everything under it |
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
ReactiveMappingBase   ←───────  DictViewBase, IndexedDictViewBase, FractionalIndexedDictViewBase,
                                LogIndexedDictViewBase, Kh57ViewBase (each → Eager / Lazy facet)
MutableMappingBase    ←───────  FlatDictView, LightDictView
ReactiveSequenceBase  ←───────  ListViewBase (→ Eager / Lazy), ByteArrayView
MutableSequenceBase   ←───────  FlatListView
SequenceBase          ←───────  TupleView, RingView (+ append / extend / clear)
ReactiveSetBase       ←───────  SetView
SetBase               ←───────  FrozenSetView
```

A view lists its collection base right after `ViewBase`, so the view's own methods and the `virtuals.view` mixins win over the base defaults. Where a layout cannot do what the built-in does, it raises: a kh57 view is not reversible (`reversed()` raises TypeError, as on a plain `Mapping`), a ring has no write at a position, a list view takes no slice writes. `tests/views/test_abc_conformance.py` runs every ABC method on every built-in view and facet against the built-in.

Views that support nested containers have two **facets** — eager (reads return Python values) and lazy (reads return child Views). Both share a common base with mutations, keys, and lifecycle. See [Eager vs Lazy Access](eager-lazy.md) for details.

Views compose collection behavior with storage machinery (container access, observation wiring, metadata management) through the mixin bases in `virtuals.view`.
