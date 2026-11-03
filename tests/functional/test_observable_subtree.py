"""Watches cover what they name (contract section 5).

A child watch on an untyped address fires on every write under it, so a
list or dict stored there does not need a typed watch to be observed.
"""

from __future__ import annotations

import time

from virtuals._views import EagerDictView, EagerListView
from virtuals.codecs import NoOpCodec
from virtuals.navigator import Navigator
from virtuals.observers.mem import InMemoryObserver
from virtuals.publishers.mem import InMemoryPublisher
from virtuals.storages.mem import InMemoryStorage
from virtuals.tkv.transport import InMemoryTransport


def _wait_until(pred, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


def test_untyped_child_watch_fires_on_every_append_to_a_list_stored_there():
    transport = InMemoryTransport()
    publisher = InMemoryPublisher(transport)
    observer = InMemoryObserver(transport)
    storage = InMemoryStorage(codec=NoOpCodec(), publisher=publisher)
    publisher.connect()
    observer.connect()
    storage.open()
    nav = Navigator(storage)
    leaf_wakes: list = []
    sibling_wakes: list = []
    try:
        with storage.transaction() as tx:
            doc = nav.root(tx).open_child("doc", EagerDictView)
            doc["items"] = [1]
            doc["other"] = 0
            leaf_options = doc.on_child_change("items")
            sibling_options = doc.on_child_change("other")
        publisher.flush()  # the setup writes are delivered before anyone watches
        observer.flush()
        observer.subscribe(leaf_options).bind(leaf_wakes.append)
        observer.subscribe(sibling_options).bind(sibling_wakes.append)

        for i in range(3):
            with storage.transaction() as tx:
                items = (
                    nav.root(tx).open_child("doc", EagerDictView).open_child("items", EagerListView)
                )
                items.append(10 + i)
            publisher.flush()
            observer.flush()
        assert _wait_until(lambda: len(leaf_wakes) >= 3), leaf_wakes
        # the typed list watch sees the same writes: the untyped one is not a superset
        with storage.transaction() as tx:
            items = nav.root(tx).open_child("doc", EagerDictView).open_child("items", EagerListView)
            typed_options = items.on_change()
        typed: list = []
        observer.subscribe(typed_options).bind(typed.append)
        with storage.transaction() as tx:
            nav.root(tx).open_child("doc", EagerDictView).open_child("items", EagerListView).append(
                99
            )
        publisher.flush()
        observer.flush()
        assert _wait_until(lambda: len(typed) >= 1)
        assert set(typed) <= set(leaf_wakes)
        assert sibling_wakes == []
        # a primitive child still fires on its own row
        with storage.transaction() as tx:
            nav.root(tx).open_child("doc", EagerDictView)["other"] = 1
        publisher.flush()
        observer.flush()
        assert _wait_until(lambda: len(sibling_wakes) == 1)
    finally:
        storage.close()
        observer.disconnect()
        publisher.disconnect()
