# Eager vs Lazy Access

## The Principle

Every collection view that supports nested containers has **two symmetric facets** — eager and lazy. Neither is primary. Both share the same storage, differ only in how reads surface results.

```text
DictViewBase                  shared: mutations, keys, lifecycle
├── EagerDictView             reads return Python values
│   └── .lazy → LazyDictView
└── LazyDictView              reads return child Views
    └── .eager → EagerDictView
```

Cross-navigation is a first-class operation. Switching facets creates a lightweight view over the same container — no data copied, no state carried over.

## How It Works

```python
users = EagerDictView.open_root(tx)
users["alice"] = {"name": "Alice", "role": "admin", "score": 95}
users["bob"] = {"name": "Bob", "role": "member", "score": 82}

# --- Eager: Python values out ---
users["alice"]              # → {"name": "Alice", "role": "admin", "score": 95}
list(users.values())        # → [{"name": ...}, {"name": ...}]

# --- Lazy: child Views out ---
users.lazy["alice"]         # → EagerDictView (a live view, not a dict)
list(users.lazy.values())   # → [EagerDictView, EagerDictView]

# --- Cross-navigate freely ---
users.lazy.eager is not users   # different instance, same storage
users.lazy.eager["alice"]       # → {"name": "Alice", ...}
```

### Primitives pass through

Lazy reads return Views for **container** children and **values** for primitive children. There's nothing to wrap — a primitive has no children to navigate into.

```python
users["score"] = 42
users.lazy["score"]         # → 42 (not a View)
```

## Not a Mode

`.lazy` is a **facet**, not a configuration. Child views returned from lazy access are eager by default:

```python
for user_view in islice(users.lazy.values(), 3):
    # user_view is an EagerDictView — defaults to eager
    user_view["name"]           # → "Alice" (value)
    user_view.lazy["name"]      # → "Alice" (also value — it's a primitive)
    user_view.extract()         # → {"name": "Alice", "role": "admin", "score": 95}
```

Each navigation step, you choose eager or lazy independently. No hidden state.

## What Differs Between Facets

Only **read operations that surface child data**:

| Operation | Eager | Lazy |
|-----------|-------|------|
| `view["key"]` | value | View (container) / value (primitive) |
| `view.values()` | values | Views / values |
| `view.items()` | (key, value) | (key, View / value) |
| `iter(listview)` | values | Views / values |
| `view.extract()` | dict / list | eager only |

Everything else lives on the shared **base** — same for both facets:

| Operation | Where |
|-----------|-------|
| `len(view)` | base |
| `"key" in view` | base |
| `view["key"] = val` | base |
| `del view["key"]` | base |
| `view.clear()` | base |
| `view.update(...)` | base |
| `view.store(...)` | base |
| `view.keys()` | base |

## Composition Without Specialized Views

The primary motivation: lazy facets eliminate specialized slice/window view types. Python's stdlib tools compose naturally with lazy views.

```python
from itertools import islice

# Slice — just islice on lazy values
first_3 = list(islice(users.lazy.values(), 3))

# Filter without materializing
admins = [v for v in users.lazy.values() if v["role"] == "admin"]

# Count without extracting
admin_count = sum(1 for v in users.lazy.values() if v["role"] == "admin")

# Selective extraction — materialize only what you need
for view in first_3:
    print(view.extract())
```

## Which Views Have Facets

Only views that support **nested containers** need the eager/lazy split:

| View | Facets | Why |
|------|--------|-----|
| `EagerDictView` / `LazyDictView` | yes | children can be containers |
| `EagerListView` / `LazyListView` | yes | children can be containers |
| `EagerIndexedDictView` / `LazyIndexedDictView` | yes | children can be containers |
| `SetView` | no | stores primitives only |
| `FrozenSetView` | no | stores primitives only |
| `FlatDictView` | no | primitives only by design |
| `FlatListView` | no | primitives only by design |
| `LightDictView` | no | primitives only, no nesting |
| `TupleView` | no | immutable, primitives only |
| `ByteArrayView` | no | raw bytes |

## Architecture

### Three classes per view concept

```text
DictViewBase                    shared mutations, keys, contains, len, lifecycle
├── EagerDictView               eager reads, extract(), functional ops
└── LazyDictView                lazy reads, facet cross-navigation
```

Both facets inherit from the same base. The base carries all mixin capabilities (observable, navigation, nested get/set, etc.). The facets only override the read methods.

### Standardization bases

`view/bases.py` provides reusable bases for building custom views with eager/lazy support:

- `ChildNestedGetBase` — eager reads: `_get_child_value()` extracts containers
- `LazyChildReadBase` — lazy reads: `_get_child_view_or_value()` returns Views for containers

Community views can compose these bases to get the same eager/lazy pattern.

### Facet switching is cheap

Views are stateless (`@attrs.frozen`). Switching facets creates a new instance pointing to the same container and registry — no data copied, no overhead:

```python
# These are equivalent — both point to the same storage
lazy = EagerDictView(container, registry).lazy
lazy = LazyDictView(container=container, registry=registry)
```
