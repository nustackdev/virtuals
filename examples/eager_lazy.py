"""Eager vs lazy access - navigate and slice without materializing data."""

from itertools import islice

from virtuals._views import EagerListView
from virtuals.codecs import NoOpCodec
from virtuals.navigator import Navigator
from virtuals.storages.mem import InMemoryStorage


storage = InMemoryStorage(codec=NoOpCodec())
storage.open()

nav = Navigator(storage)

with storage.transaction() as tx:
    root = nav.root(tx)

    # Populate some data
    root["alice"] = {"name": "Alice", "role": "admin", "score": 95}
    root["bob"] = {"name": "Bob", "role": "member", "score": 82}
    root["charlie"] = {"name": "Charlie", "role": "member", "score": 91}
    root["diana"] = {"name": "Diana", "role": "admin", "score": 88}
    root["eve"] = {"name": "Eve", "role": "member", "score": 76}

    # =========================================================================
    # EAGER - the default, returns Python values
    # =========================================================================

    # Just like a regular dict
    print(root["alice"])
    # -> {'name': 'Alice', 'role': 'admin', 'score': 95}

    # Standard iteration yields materialized dicts
    for uid, profile in root.items():
        print(f"  {uid}: {profile['name']} ({profile['role']})")

    # =========================================================================
    # LAZY - returns child Views instead of extracting
    # =========================================================================

    lazy = root.lazy

    # Lazy access returns a View, not a dict
    alice_view = lazy["alice"]
    print(type(alice_view).__name__)
    # -> EagerDictView  (child views are eager by default)

    # The View is live - you can read, navigate, or extract from it
    print(alice_view["name"])  # -> Alice  (eager read on the child)
    print(alice_view.extract())  # -> {'name': 'Alice', 'role': 'admin', 'score': 95}

    # =========================================================================
    # COMPOSITION - lazy + Python stdlib = no specialized views needed
    # =========================================================================

    # Get the first 3 users as Views (no data materialized yet)
    first_3 = list(islice(lazy.values(), 3))
    print(f"Got {len(first_3)} views, no data extracted yet")

    # Now selectively extract only what we need
    for view in first_3:
        print(f"  {view['name']}: score={view['score']}")

    # Pair keys with lazy views, take a slice
    top_2_items = list(islice(lazy.items(), 2))
    for uid, view in top_2_items:
        print(f"  {uid} -> {view.extract()}")

    # Filter with a generator - only extract matching users
    admins = [view.extract() for view in lazy.values() if view["role"] == "admin"]
    print(f"Admins: {admins}")

    # Count without extracting
    admin_count = sum(1 for v in lazy.values() if v["role"] == "admin")
    print(f"Admin count: {admin_count}")

    # =========================================================================
    # LAZY IS NOT A MODE - each step chooses independently
    # =========================================================================

    # Navigate lazily to get a child view
    alice_view = lazy["alice"]

    # The child view is eager by default
    print(alice_view["name"])  # -> "Alice" (value)

    # Explicitly go lazy on the child too
    print(alice_view.lazy["name"])  # -> "Alice" (still a value - it's a primitive)

    # =========================================================================
    # CROSS-NAVIGATION - switch between eager and lazy freely
    # =========================================================================

    # Start eager, go lazy, come back
    eager_again = lazy.eager
    print(eager_again["alice"])  # -> {'name': 'Alice', ...}  (extracted dict again)

    # Mutations work through either facet - same storage underneath
    lazy["frank"] = {"name": "Frank", "role": "member", "score": 70}
    print("frank" in root)  # -> True (visible from eager)
    print(root["frank"]["name"])  # -> Frank

    # =========================================================================
    # LISTS - same pattern
    # =========================================================================

    scores_list = root.open_child("scores", EagerListView)
    scores_list.store(
        [
            {"user": "alice", "points": 100},
            {"user": "bob", "points": 85},
            {"user": "charlie", "points": 92},
            {"user": "diana", "points": 88},
        ]
    )

    # Eager - plain Python values
    print(scores_list[0])  # -> {'user': 'alice', 'points': 100}

    # Lazy - Views
    lazy_scores = scores_list.lazy
    first_score = lazy_scores[0]
    print(f"{first_score['user']}: {first_score['points']} pts")

    # islice on lazy list
    top_2 = list(islice(lazy_scores, 2))
    for view in top_2:
        print(f"  {view['user']}: {view['points']}")

storage.close()
