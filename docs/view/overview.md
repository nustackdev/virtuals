# Layer 3: View

## What It Is

View adds data structure semantics on top of Container's hierarchy. It makes containers behave like Python data structures (dicts, lists, sets) while handling all storage details automatically.

## Key Concepts

### Views as Data Structures

Container sees containers. Views see data structures:

```python
# Container Layer 2: generic containers
container.create_container(("users",))
container.set_child("alice", value)

# View Layer 3: Python data structures
users["alice"] = {"name": "Alice"}
profile = users["alice"]
```

Each view implements familiar Python protocols:

- **DictView**: dict-like (`__getitem__`, `keys()`, `values()`)
- **ListView**: list-like (`append()`, `pop()`, `[0]`)
- **SetView**: set-like (`add()`, `remove()`, `in`)
- **QueueView**: queue-like (`enqueue()`, `dequeue()`)

Views define their own interface. No mandatory protocols.

### View Identity

Each view has two attributes:

```python
class DictView(View):
    structure = ContainerStructure(1)     # Storage identifier
    protocol = ContainerProtocol.MUTABLE  # Behavior hints
```

**ContainerStructure** (int):

- Unique ID for this view type
- Used to recreate correct view when reading
- Examples: `1` = dict, `2` = list, `3` = set

**ContainerProtocol** (bitflags):

- Optional hints for debugging/visualization
- NOT enforced - just metadata
- Examples: `MUTABLE`, `SIZED`, `INDEXED`

### Registry: Type Mapping

Registry maps between Python types and Views:

```python
# Python type → View (for writing)
registry.register(dict → DictView)
registry.register(list → ListView)
registry.register(set → SetView)

# Structure ID → View (for reading)
registry.register(1 → DictView)
registry.register(2 → ListView)
registry.register(3 → SetView)
```

This enables automatic nested handling:

```python
# Writing
users["alice"] = {"name": "Alice", "tags": ["python"]}
# Registry: dict → DictView, list → ListView

# Reading
data = users["alice"]  # Returns: {"name": "Alice", "tags": ["python"]}
# Registry: structure=1 → DictView, structure=2 → ListView
```

### Auto-Population and Extraction

Views handle nested structures automatically.

**Population** (Python → Storage):

```python
users["alice"] = {
    "name": "Alice",
    "profile": {
        "age": 30,
        "tags": ["python", "rust"]
    }
}
```

Stores as:

```text
("users", "alice") → <CONTAINER: structure=1>
("users", "alice", "name") → "Alice"
("users", "alice", "profile") → <CONTAINER: structure=1>
("users", "alice", "profile", "age") → 30
("users", "alice", "profile", "tags") → <CONTAINER: structure=2>
("users", "alice", "profile", "tags", 0) → "python"
("users", "alice", "profile", "tags", 1) → "rust"
```

**Extraction** (Storage → Python):

```python
data = users.extract()
# Returns: {
#     "alice": {
#         "name": "Alice",
#         "profile": {
#             "age": 30,
#             "tags": ["python", "rust"]
#         }
#     }
# }
```

Views implement population/extraction protocols:

```python
class View:
    def store(self, value, /, *, replace=False):
        """Populate container from Python value."""
        ...

    def extract(self):
        """Extract container to Python value."""
        ...
```

## Operations

### DictView Operations

```python
# Create/access dict view
users # DictView

# Get/set like dict
users["alice"] = {"name": "Alice", "age": 30}
alice = users["alice"]

# Dict methods
keys = users.keys()              # ["alice", "bob"]
values = users.values()          # [{"name": "Alice"}, ...]
items = users.items()            # [("alice", {...}), ...]
has_alice = "alice" in users     # True

# Mutation
users.update({"charlie": {"name": "Charlie"}})
users.pop("alice")
users.clear()

# Extract entire dict
data = users.extract()  # {"alice": {...}, "bob": {...}}
```

### ListView Operations

```python
# Create/access list view
tasks # ListView

# Get/set like list
tasks.append("Task 1")
tasks.append("Task 2")
first = tasks[0]
tasks[1] = "Updated task"

# List methods
tasks.insert(1, "New task")
tasks.remove("Task 1")
last = tasks.pop()

# Iteration
for task in tasks:
    print(task)

# Extract entire list
all_tasks = tasks.extract()  # ["Task 1", "Task 2"]
```

### SetView Operations

```python
# Create/access set view
tags  # SetView

# Set operations
tags.add("python")
tags.add("rust")
has_python = "python" in tags  # True

# Set methods
tags.remove("python")
tags.discard("java")  # Silent if missing

# Set algebra (plain sets back)
union = tags | other
intersection = tags & other

# Extract entire set
all_tags = tags.extract()  # {"python", "rust"}
```

### QueueView Operations

```python
# Create/access queue view
queue  # QueueView

# Queue operations (FIFO)
queue.enqueue({"type": "login"})
event = queue.dequeue()  # First in, first out
next_event = queue.peek()  # Look without removing

is_empty = queue.is_empty()
size = len(queue)

# Extract as list
events = queue.extract()
```

## Write Rules

The same on every view and every backend. Pinned by `tests/views/test_dict_views_hardening.py`.

- `EMPTY` is never stored. Any write of `EMPTY`, at the top or nested inside a dict, list or lazy iterable, raises `ValueError` before anything is written. `__setitem__`, `store`, `update`, `append`, `prepend`, `insert`, `insert_before`, `insert_after`, `extend`, `add`, `set_primitive` and `set_child_container_as` all check first; so do the container's `put_child_primitive`, `put_child_primitive_unsafe` and `put_metadata`. The private `_unsafe_primitive_write` and the raw storage `put` refuse `EMPTY` itself (a copy or an unpickled one included) but do not walk a compound value for a nested one. A rejected write on an absent view leaves it absent. To remove a key, delete it.
- `None` is a value. `view["k"] = None` stores it and `"k" in view` is true.
- A key keeps its kind. `d["p"] = 1` then `d["p"] = {"z": 1}` raises `ContainerTypeError`, and so does the reverse. Nothing changes. Delete the key, then set it. Compound over compound replaces the whole value.
- Writes create. Setting into an absent view creates it. A delete of a missing key creates nothing: DictView, FlatDictView and LightDictView are silent, the index views and Kh57View raise `KeyError`. Nor does any write that raises or changes nothing: an out-of-range `del` or set on a list or bytearray, `remove` or `discard` of a non-member, `extend([])`, `update({})`, `clear()`.
- A set member is an opaque hashable value. A tuple or frozenset is stored as one row and reads back as is; a list, dict or set raises `TypeError` before any write, as a Python set would, and so does a member that does not pickle. A frozenset member is keyed by its sorted contents, so equal sets match from any process.
- `update` applies its keys in order. A kind change on one key raises there and leaves the keys before it written and the rest unwritten; abort the transaction, or delete the key first.
- Every view is the `collections.abc` type it stands for, with all its methods, on both facets: `pop`, `popitem`, `setdefault`, `update`, `reverse`, `+=`, `|=` and the rest work as on the built-in. A lazy `pop` or `popitem` hands back the value, not a view of the removed child. What a layout cannot do raises: `reversed()` on a Kh57View (no order to reverse), a write at a position on a RingView, a slice write on a list view. `tests/views/test_abc_conformance.py` pins all of it against the built-ins.
- `get(key)` of an absent key returns `None`, as `dict.get` does.
- Existence is one read: `view.container.exists()`. Opening a chain of children touches no storage.
- `len()` on DictView, FlatDictView, LightDictView, LogIndexedDictView and FractionalIndexedDictView counts live rows (O(children)); Kh57View, IndexedDictView, ListView and SetView read a stored count. `bool()` is one storage op on every view. Paging with `keys(after=, limit=)` and `keys_reverse(before=, limit=)` on the dict, log and fractional views returns exactly `limit` keys when they exist. `RingView.len()` is two seeks: its sequence numbers are dense.
- `IndexedDictView` delete and `ListView` insert or delete in the middle are O(n). Use `FractionalIndexedDictView` for ordered collections.

## How It Uses Layer 2

View operations map to Container operations:

### DictView: `users["alice"] = {"name": "Alice"}`

```python
# View Layer 3
users["alice"] = {"name": "Alice"}

# ↓ uses Container Layer 2
container.create_container(("users", "alice"), structure=1)
container.set_child_primitive(("users", "alice"), "name", "Alice")

# ↓ uses Storage Layer 1
storage.put(("users", "alice"), <CONTAINER: structure=1>)
storage.put(("users", "alice", "name"), "Alice")
```

### ListView: `tasks.append("Task 1")`

```python
# View Layer 3
tasks.append("Task 1")

# ↓ uses Container Layer 2
count = container.count_children(("tasks",))
container.set_child_primitive(("tasks",), count, "Task 1")

# ↓ uses Storage Layer 1
storage.put(("tasks", 0), "Task 1")
```

## Registry System

### Registering Views

```python
from virtuals.view import ViewRegistry
from virtuals.views import EagerDictView, EagerFractionalIndexedDictView, EagerListView

registry = ViewRegistry()

# A writer: its structure ID reads back to it, and it writes its Python type.
registry.register(EagerDictView)  # structure 1, dict
registry.register(EagerListView)  # structure 2, list

# A reader: its structure ID reads back to it, but it never writes a type
# (an ordered dict is a dict too, and dict is already EagerDictView's).
registry.register_reader(EagerFractionalIndexedDictView)
```

A `Navigator` builds its registry this way: the views it is given (or the
standard ones) as writers, then every other built-in view as a reader. So
a container stored with any built-in view (an ordered dict, a log dict, a
ring) reads back by its structure, eg when its parent is extracted,
whichever views write the Python types.

### Registry Lookup

**Reading:** Structure ID → View

```python
view_class = registry.get_view_for_structure(1)  # Returns: DictView
```

**Writing:** Python Type → View

```python
view_class = registry.get_view_for_type(dict)  # Returns: EagerDictView
```

## Context Management

### Context Manager (Recommended)

```python
# Write operations
with DictView().open() as users:
    users["alice"] = {"name": "Alice"}
    # Auto-commits on success, auto-rollback on exception

# Read-only snapshot
with DictView().open(readonly=True) as users:
    users["alice"]
```

### Direct Access [TBD]

```python
# Read without transaction
users = tree.at("users").view(DictView)
count = len(users)

# Write with manual transaction
with tree.transaction() as tx:
    users = tree.at("users").view(DictView, ctx=tx)
    users["alice"] = {"name": "Alice"}
```

## Navigation

### Auto-View Detection

```python
# Store nested structures - types auto-detected
users["alice"] = {
    "name": "Alice",
    "tags": ["python"],
    "permissions": {"read", "write"}
}

# Read back - types preserved
alice = users["alice"]  # dict
tags = alice["tags"]     # list
perms = alice["permissions"]  # set
```

## Custom Views

```python
from virtuals.view import View, ContainerStructure, ContainerProtocol

class DocumentView(View):
    """Custom view for documents."""

    structure = ContainerStructure(100)
    protocol = ContainerProtocol.MUTABLE

    def get_title(self):
        return self.container.get_primitive_child("title")

    def set_title(self, title):
        self.container.set_primitive_child("title", title)

    def extract(self):
        return {"title": self.get_title(), ...}

    def store(self, value, /, *, replace=False):
        self.set_title(value["title"])

# Register
registry.register_view(DocumentView,)
```

## What View Does NOT Do

### No Storage Operations

Doesn't handle:

- KV storage access
- Encoding/decoding
- Transaction management

These are Storage (Layer 1) responsibilities.

### No Hierarchy Enforcement

Doesn't handle:

- Parent validation
- Container creation rules
- Type consistency

These are Container (Layer 2) responsibilities.

### No Application Logic

Doesn't handle:

- Business rules
- Domain validation
- Command/query patterns

These are application-level responsibilities.
