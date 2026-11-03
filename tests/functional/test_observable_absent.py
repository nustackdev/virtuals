"""Subscriptions on an absent container work (contract section 6).

A container ref always reads as a live view, so a watch taken before the
container exists must fire once something is written under it, and must
not write anything of its own.
"""

from __future__ import annotations

import time

import pytest

from virtuals._views import (
    EagerDictView,
    EagerFractionalIndexedDictView,
    EagerListView,
    EagerLogIndexedDictView,
)
from virtuals.codecs import NoOpCodec
from virtuals.navigator import Navigator
from virtuals.observers.mem import InMemoryObserver
from virtuals.publishers.mem import InMemoryPublisher
from virtuals.storages.mem import InMemoryStorage
from virtuals.tkv.filter import PrefixFilter
from virtuals.tkv.storage import StorageScanOptions
from virtuals.tkv.transport import InMemoryTransport


def _wait_until(pred, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


@pytest.fixture
def bus():
    transport = InMemoryTransport()
    publisher = InMemoryPublisher(transport)
    observer = InMemoryObserver(transport)
    storage = InMemoryStorage(codec=NoOpCodec(), publisher=publisher)
    publisher.connect()
    observer.connect()
    storage.open()
    try:
        yield storage, publisher, observer
    finally:
        storage.close()
        observer.disconnect()
        publisher.disconnect()


def _rows(storage):
    with storage.snapshot() as snap:
        prefix = PrefixFilter(prefix=("/",))
        opts = StorageScanOptions(start=("/",), break_filter=prefix, filter=prefix)
        return [k for k, _ in snap.scan(opts).items()]


@pytest.mark.parametrize(
    "view_cls",
    (EagerDictView, EagerLogIndexedDictView, EagerFractionalIndexedDictView, EagerListView),
)
def test_subscribe_on_absent_view_fires_on_first_write_and_writes_nothing(bus, view_cls):
    storage, publisher, observer = bus
    nav = Navigator(storage)
    with storage.snapshot() as snap:
        view = nav.root(snap).open_child("doc", view_cls)
        assert not view.container.exists()
        whole = view.on_change()
        # ListView.on_child_change(i) normalizes i against len(), so on an
        # absent list it raises IndexError; the children watch has no index.
        child = (
            view.on_children_change() if view_cls is EagerListView else view.on_child_change("k")
        )
    assert _rows(storage) == []
    whole_wakes: list = []
    child_wakes: list = []
    observer.subscribe(whole).bind(whole_wakes.append)
    observer.subscribe(child).bind(child_wakes.append)
    assert _rows(storage) == []

    with storage.transaction() as tx:
        view = nav.root(tx).open_child("doc", view_cls)
        if view_cls is EagerListView:
            view.append({"a": 1})
        else:
            view["k"] = {"a": 1}
    publisher.flush()
    observer.flush()
    assert _wait_until(lambda: len(whole_wakes) >= 1), whole_wakes
    assert _wait_until(lambda: len(child_wakes) >= 1), child_wakes
    with storage.snapshot() as snap:
        assert nav.root(snap).open_child("doc", view_cls).container.exists()


def test_subscribe_on_absent_nested_chain_fires_when_the_chain_is_created(bus):
    storage, publisher, observer = bus
    nav = Navigator(storage)
    with storage.snapshot() as snap:
        leaf = (
            nav.root(snap)
            .open_child("a", EagerDictView)
            .open_child("b", EagerDictView)
            .open_child("c", EagerDictView)
        )
        options = leaf.on_change()
    assert _rows(storage) == []
    wakes: list = []
    observer.subscribe(options).bind(wakes.append)
    with storage.transaction() as tx:
        nav.root(tx).open_child("a", EagerDictView)["b"] = {"c": {"x": 1}}
    publisher.flush()
    observer.flush()
    assert _wait_until(lambda: len(wakes) >= 1), wakes
    seen = len(wakes)
    with storage.transaction() as tx:
        nav.root(tx).open_child("a", EagerDictView)["other"] = 1
    publisher.flush()
    observer.flush()
    assert len(wakes) == seen
