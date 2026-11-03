"""Deleting a subtree drops its metadata too, on every backend.

A site deleted through its parent (``del d[k]``, ``clear``, ``store``) must
not leave ``/m`` rows behind: a view of another type opened later at the
same site would inherit them as its own bookkeeping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from virtuals._views import EagerDictView, SetView
from virtuals.codecs import BinaryCodec, TextCodec
from virtuals.container import Container
from virtuals.navigator import Navigator


if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from virtuals.tkv.storage import StorageProtocol


@pytest.fixture(params=["mem", "textdb", "sqlite", "lmdb", "rocksdb"])
def storage(request: pytest.FixtureRequest, tmp_path: Path) -> Generator[StorageProtocol]:
    if request.param == "mem":
        from virtuals.storages.mem import InMemoryStorage

        s = InMemoryStorage(codec=BinaryCodec())
    elif request.param == "textdb":
        from virtuals.storages.textdb import TextStorage

        s = TextStorage(path=tmp_path / "t", codec=TextCodec())
    elif request.param == "sqlite":
        from virtuals.storages.sqlite import SQLiteStorage

        s = SQLiteStorage(path=tmp_path / "kv.sqlite", codec=BinaryCodec())
    elif request.param == "lmdb":
        from virtuals.storages.lmdb import LMDBStorage

        s = LMDBStorage(path=tmp_path / "l", codec=BinaryCodec(), map_size=2**26)
    else:
        from virtuals.storages.rocksdb import RocksDBStorage

        s = RocksDBStorage(path=tmp_path / "db", codec=BinaryCodec())
    s.open()
    try:
        yield s
    finally:
        s.close()


def test_delete_drops_stale_counter_before_reuse_as_set(storage):
    """Old DictView with ``__len__`` 2, deleted, reopened as SetView: len is 1."""
    nav = Navigator(storage)
    with storage.transaction() as tx:
        d = nav.root(tx).open_child("d", EagerDictView)
        d.store({"a": 1, "b": {"x": 1}})
        d.container.put_metadata("__len__", 2)
        d.open_child("b", EagerDictView).container.put_metadata("__len__", 1)

    with storage.transaction() as tx:
        del nav.root(tx)["d"]

    with storage.transaction() as tx:
        s = nav.root(tx).open_child("d", SetView)
        assert not s.container.exists_metadata("__len__")
        s.add(5)
        assert len(s) == 1
        b = Container(ctx=tx, site=(*s.container.site, "b"))
        assert not b.exists_metadata("__len__")


def test_delete_leaves_sibling_metadata_alone(storage):
    nav = Navigator(storage)
    with storage.transaction() as tx:
        root = nav.root(tx)
        root.open_child("d", EagerDictView).store({"a": 1})
        root.open_child("d", EagerDictView).container.put_metadata("__len__", 1)
        root.open_child("dd", EagerDictView).store({"a": 1})
        root.open_child("dd", EagerDictView).container.put_metadata("__len__", 9)
        root.container.put_metadata("note", "keep")

    with storage.transaction() as tx:
        del nav.root(tx)["d"]

    with storage.snapshot() as snap:
        root = nav.root(snap)
        assert root.open_child("dd", EagerDictView).container.get_metadata("__len__") == 9
        assert root.container.get_metadata("note") == "keep"
        assert not root.open_child("d", EagerDictView).container.exists_metadata("__len__")


def test_clear_and_store_drop_child_metadata(storage):
    nav = Navigator(storage)
    with storage.transaction() as tx:
        d = nav.root(tx).open_child("d", EagerDictView)
        d.store({"a": {"x": 1}})
        d.open_child("a", EagerDictView).container.put_metadata("__len__", 7)

    with storage.transaction() as tx:
        nav.root(tx).open_child("d", EagerDictView).store({"a": {"y": 2}})

    with storage.snapshot() as snap:
        a = nav.root(snap).open_child("d", EagerDictView).open_child("a", EagerDictView)
        assert not a.container.exists_metadata("__len__")
        assert a.extract() == {"y": 2}
