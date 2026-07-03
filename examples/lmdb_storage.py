"""LMDB storage backend - persistent dict + list over an LMDB env.

Requires the `lmdb` extra:  pip install virtuals-py[lmdb]

Runs two sessions against the same LMDB path to prove data survives
close/reopen. The second session reads what the first one wrote.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

from virtuals._views import EagerListView
from virtuals.codecs import BinaryCodec
from virtuals.navigator import Navigator
from virtuals.storages.lmdb import LMDBStorage


def session_write(path: Path) -> None:
    """Populate the LMDB env with a small user + scores tree."""
    storage = LMDBStorage(path=path, codec=BinaryCodec(), map_size=64 * 1024 * 1024)
    nav = Navigator(storage)

    with nav.storage as st, st.transaction() as tx:
        root = nav.root(tx)

        # Nested dicts under a root dict
        root["alice"] = {"name": "Alice", "age": 30}
        root["bob"] = {"name": "Bob", "age": 25}
        root["carol"] = {"name": "Carol", "age": 42}

        # A child list under the root
        scores = root.open_child("scores", EagerListView)
        scores.store([100, 200, 300])
        scores.append(400)

        print("write session")
        print("  users:", sorted(root.keys()))
        print("  alice:", root["alice"])
        print("  scores:", list(scores))


def session_read(path: Path) -> None:
    """Reopen the same LMDB env and read what session_write persisted."""
    storage = LMDBStorage(path=path, codec=BinaryCodec(), map_size=64 * 1024 * 1024)
    nav = Navigator(storage)

    with nav.storage as st, st.snapshot() as snap:
        root = nav.root(snap)

        print("read session (reopened env)")
        print("  users:", sorted(root.keys()))
        print("  bob:", root["bob"])

        scores = root.open_child("scores", EagerListView)
        print("  scores:", list(scores))
        print("  len(scores):", len(scores))


def session_update(path: Path) -> None:
    """Overwrite + delete inside a fresh transaction and check persistence."""
    storage = LMDBStorage(path=path, codec=BinaryCodec(), map_size=64 * 1024 * 1024)
    nav = Navigator(storage)

    with nav.storage as st:
        with st.transaction() as tx:
            root = nav.root(tx)
            root["alice"] = {"name": "Alice", "age": 31}
            del root["carol"]

        with st.snapshot() as snap:
            root = nav.root(snap)
            print("update session")
            print("  users:", sorted(root.keys()))
            print("  alice:", root["alice"])


def main() -> None:
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "example.lmdb"
        session_write(path)
        session_read(path)
        session_update(path)
        session_read(path)


if __name__ == "__main__":
    main()
