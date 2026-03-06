"""Multiple collection types sharing one storage."""

from virtuals._views import DictView, ListView, SetView
from virtuals.codecs import NoOpCodec
from virtuals.storages.mem import InMemoryStorage


storage = InMemoryStorage(codec=NoOpCodec())
storage.open()

with storage.transaction() as tx:
    root = DictView.open_root(tx)

    # Users — dict of dicts
    users = root.open_child("users", DictView)
    users["alice"] = {"role": "admin", "active": True}
    users["bob"] = {"role": "member", "active": True}
    users["charlie"] = {"role": "member", "active": False}

    # Events — ordered list
    events = root.open_child("events", ListView)
    events.append("alice logged in")
    events.append("bob signed up")
    events.append("charlie deactivated")

    # Tags — unique set
    tags = root.open_child("tags", SetView)
    tags.add("python")
    tags.add("rust")
    tags.add("python")  # no-op, already exists
    print(f"tags: {tags.extract()}")  # {'python', 'rust'}

    # Everything lives in flat KV pairs under the hood,
    # but you work with native Python interfaces.
    print(f"users: {users.extract()}")
    print(f"events: {events.extract()}")
    print(f"active users: {[k for k, v in users.items() if v['active']]}")

storage.close()
