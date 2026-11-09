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
        │       ├─ ReactiveMappingBase[K, V]
        │       └─ DictMethods[K, V]              dict
        │
        ├─ SequenceBase[V]
        │   ├─ TupleMethods[V]                    tuple
        │   └─ MutableSequenceBase[V]
        │       ├─ ReactiveSequenceBase[V]
        │       ├─ ListMethods[V]                 list
        │       └─ ByteArrayMethods               bytearray
        │
        └─ SetBase[V]
            ├─ FrozenSetMethods[V]                frozenset
            └─ MutableSetBase[V]
                ├─ ReactiveSetBase[V]
                └─ SetMethods[V]                  set (+ FrozenSetMethods)
```

Each collection type follows a three-tier pattern:

- **Read-only base** — minimal abstract methods, rich derived defaults
- **Mutable base** — adds mutation abstracts, derives convenience methods
- **Reactive base** — adds observation abstracts (`on_change`, `on_child_change`, etc.)

Beside them, one `*Methods` class per built-in adds what that built-in has beyond its ABC. A view that mixes one in is that built-in in full.

## How It Works

### Minimal abstracts, maximum defaults

Each base is its `collections.abc` class: the same abstract core, the same mixin methods with the same semantics, and it is registered with that ABC, so `isinstance(view, MutableMapping)` holds for every view built on it. Built-in-only methods (`list +`, `dict |`, `set.union`, `copy`, `sort`) are not on the ABC bases: they sit on the `*Methods` classes below.

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

### Built-in methods

A view that models a built-in acts exactly like it: every method, same results, same errors. One that cannot implement a built-in's whole API sits at its ABC instead (Kh57View at `MutableMapping`, RingView at `Sequence`). Methods that return a new value return a plain built-in (`copy()` is a `dict`, `view + [1]` is a `list`), holding the items as the facet reads them, a shallow copy as on the built-in.

**DictMethods** (`dict`): `copy()`, `|` and its reflected form (a dict or dict view, as `dict | dict`), `|=` (anything `update` takes); requires `__reversed__`.

**ListMethods** (`list`): `copy()`, `sort(key=, reverse=)`, `+` (lists only), `*`, `*=`, ordering against lists, and slice writes through `_set_slice` / `_del_slice`.

**TupleMethods** (`tuple`): `+` (tuples only), `*`, ordering against tuples.

**FrozenSetMethods** (`frozenset`): `copy()`, `union`, `intersection`, `difference`, `symmetric_difference`, `issubset`, `issuperset` over any iterables; `& | - ^` take sets only, as on `frozenset`, and a reflected one gives the left operand's type.

**SetMethods** (`set`): the above plus `update`, `intersection_update`, `difference_update`, `symmetric_difference_update`; `|= &= -= ^=` take sets only.

**ByteArrayMethods** (`bytearray`): `copy()`, `+`, `*`, `%`, ordering against bytes-likes, slice writes, and every bytes method (`find`, `split`, `upper`, `decode`, `hex`, ...).

`sort`, `reverse` and the slice writes read the items whole and store the result back (`_plain_values`, `_replace_all`): O(n) rows.

Not provided, because a view cannot do them honestly: `hash()` of a tuple or frozenset view (the stored value can be replaced under it), `dict.fromkeys` and `bytearray.fromhex` (constructors; a view needs a place in storage), and the buffer protocol on a bytearray view (`memoryview(view)`; `bytes(view)` gives a copy).

`pop`, `popitem` and `reverse` read items through `_plain_value`, a plain read by default. A `Faceted` view overrides it to read through its eager facet, so a lazy view hands back data, not a view of a removed child.

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
ReactiveMappingBase   ←───────  Kh57ViewBase (→ Eager / Lazy facet)
  + DictMethods       ←───────  DictViewBase, IndexedDictViewBase, FractionalIndexedDictViewBase,
                                LogIndexedDictViewBase (each → Eager / Lazy facet)
DictMethods           ←───────  FlatDictView, LightDictView
ReactiveSequenceBase
  + ListMethods       ←───────  ListViewBase (→ Eager / Lazy)
  + ByteArrayMethods  ←───────  ByteArrayView
ListMethods           ←───────  FlatListView
TupleMethods          ←───────  TupleView
SequenceBase          ←───────  RingView (+ append / extend / clear)
ReactiveSetBase
  + SetMethods        ←───────  SetView
FrozenSetMethods      ←───────  FrozenSetView
```

A view lists its collection base right after `ViewBase`, so the view's own methods and the `virtuals.view` mixins win over the base defaults. Where a layout cannot do what the built-in does, it raises: a kh57 view is not reversible (`reversed()` raises TypeError, as on a plain `Mapping`), a ring has no write at a position. `tests/views/test_abc_conformance.py` runs every ABC method on every built-in view and facet against the built-in; `tests/views/test_builtin_conformance.py` does the same for the rest of each built-in's API.

Views that support nested containers have two **facets** — eager (reads return Python values) and lazy (reads return child Views). Both share a common base with mutations, keys, and lifecycle. See [Eager vs Lazy Access](eager-lazy.md) for details.

Views compose collection behavior with storage machinery (container access, observation wiring, metadata management) through the mixin bases in `virtuals.view`.
