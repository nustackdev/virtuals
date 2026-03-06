"""Basic Virtuals usage — dict and list over in-memory storage."""

from virtuals._views import DictView, ListView
from virtuals.codecs import NoOpCodec
from virtuals.storages.mem import InMemoryStorage


# Set up storage
storage = InMemoryStorage(codec=NoOpCodec())
storage.open()

with storage.transaction() as tx:
    # Root is a dict — everything lives under it
    root = DictView.open_root(tx)

    # Store users as nested dicts
    root["alice"] = {"name": "Alice", "age": 30}
    root["bob"] = {"name": "Bob", "age": 25}

    # Reads work like a regular dict
    print(root["alice"])  # {'name': 'Alice', 'age': 30}
    print(len(root))  # 2
    print(list(root.keys()))  # ['alice', 'bob']

    # Iteration
    for user_id, profile in root.items():
        print(f"{user_id}: {profile}")

    # Delete
    del root["bob"]
    print("bob" in root)  # False

    # Open a child list under the root dict
    scores = root.open_child("scores", ListView)
    scores.store([100, 200, 300])
    print(scores[0])  # 100
    print(list(scores))  # [100, 200, 300]
    scores.append(400)
    print(len(scores))  # 4

    # Extract — materialize the whole thing
    print(root.extract())

storage.close()
