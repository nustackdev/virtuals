# Eager vs Lazy Access

## The Problem

Views need to serve two use cases:

1. **Eager** — `dictview["key"]` returns a Python value. Works naturally with the entire Python ecosystem (`json.dumps`, `itertools`, `sorted`, `pprint`, etc.).

2. **Lazy** — `dictview["key"]` returns a child View. Enables composition without materializing data — a slice of a dict is just `islice` over child Views, no need for specialized `DictSliceView`.

## Design Decision: `.lazy` Accessor

Default is **eager**. Lazy access is available via a `.lazy` property that returns a lightweight proxy implementing the same collection interface but yielding Views instead of values.

```python
# Eager (default) — Python values out
dictview["key"]              # → value
list(dictview.values())      # → [val1, val2, ...]
for key in dictview:         # → keys

# Lazy — Views out
dictview.lazy["key"]         # → child View
list(dictview.lazy.values()) # → [View1, View2, ...]
for key in dictview.lazy:    # → keys (same for mappings)
```

### Why this approach

**Considered alternatives:**

| Approach | Problem |
|----------|---------|
| Default lazy, explicit eager | Breaks Python ecosystem. `list(view)` gives Views, not data. Every stdlib tool needs `.extract()` calls. |
| Instance config (`lazy=True`) | Same type, different behavior. Can't reason statically. Function receives a view — is it lazy or eager? |
| Two classes per type (`LazyDictView`) | Class explosion. Every utility must handle both. |

**Why `.lazy` wins:**

- Python ecosystem works by default — `islice(dictview, 5)`, `sorted(listview)`, `json.dumps(dict(d.items()))` all just work
- Lazy is equally Pythonic — the proxy implements the same protocols, so `islice(dictview.lazy.values(), 5)` works identically
- Single access point, discoverable, minimal API surface
- Type-safe — the proxy has different return types than the eager view
- Lightweight — wraps the same underlying storage, only changes how results surface

## Key Property: `.lazy` Is Not a Mode

`.lazy` is an **accessor**, not a configuration. It doesn't infect child views:

```python
for user_view in islice(dictview.lazy.values(), 3):
    # user_view is a regular View — defaults to eager
    user_view["name"]           # → "Alice" (eager, returns value)
    user_view.lazy["name"]      # → View (lazy, returns child View)
```

Each navigation step, you choose eager or lazy independently. No hidden state, no spooky action at a distance.

## What Needs a Lazy Variant

Only **read operations that return/yield child data**:

| Operation | Eager | Lazy |
|-----------|-------|------|
| `view["key"]` | value | View |
| `view.values()` | values | Views |
| `view.items()` | (key, value) | (key, View) |
| `iter(listview)` | values | Views |
| `listview[2:5]` | values | Views |

Operations that don't surface child data are **unchanged**:

| Operation | Same either way |
|-----------|----------------|
| `len(view)` | count |
| `"key" in view` | bool |
| `del view["key"]` | mutation |
| `view["key"] = val` | mutation |
| `view.clear()` | mutation |

## Composition Without Specialized Views

The primary motivation: `.lazy` eliminates the need for specialized view types.

```python
# Before: need DictSliceView, DictISliceView, etc.
slice_view = DictISliceView(dictview, 0, 5)

# After: just use Python's tools on lazy views
first_5 = list(islice(dictview.lazy.values(), 5))  # → [View, View, View, View, View]

# Navigate into any of them
for child in first_5:
    print(child.extract())  # materialize only what you need
```

## Implementation Notes

### The `.lazy` proxy

- Property on every View, returns a `LazyProxy` wrapping the same view
- `LazyProxy` implements the same collection base interface (MappingBase, SequenceBase, etc.)
- Instead of calling `extract()` on child containers, returns the child View directly
- For primitives (leaf values), returns the value as-is (no View to wrap)

### `.lazy[key]` vs `open_child`

`.lazy[key]` is ergonomic sugar for `view.open_child(key, auto_detected_view_type)`. The proxy auto-selects the appropriate view type based on the child's container structure.

### Type signatures

```python
class DictView:
    def __getitem__(self, key: str) -> object:           # eager
        ...

    @property
    def lazy(self) -> LazyMappingProxy[str, View]:       # lazy accessor
        ...

class LazyMappingProxy[K, V]:
    def __getitem__(self, key: K) -> View:               # returns View
        ...
    def values(self) -> Iterator[View]:                  # yields Views
        ...
    def items(self) -> Iterator[tuple[K, View]]:         # yields (key, View)
        ...
```
