"""Tests for virtuals.tkv.transport.InMemoryTransport."""

from __future__ import annotations

import threading
import time

from virtuals.tkv.transport import InMemoryTransport


class TestRegisterUnregister:
    def test_register_appends_listener(self):
        t = InMemoryTransport()
        cb = lambda _batch: None  # noqa: E731
        t.register(cb)
        assert cb in t._listeners

    def test_unregister_removes_listener(self):
        t = InMemoryTransport()
        cb = lambda _batch: None  # noqa: E731
        t.register(cb)
        t.unregister(cb)
        assert cb not in t._listeners

    def test_unregister_absent_is_noop(self):
        t = InMemoryTransport()
        cb = lambda _batch: None  # noqa: E731
        # Must not raise
        t.unregister(cb)

    def test_multiple_listeners_all_receive(self):
        t = InMemoryTransport()
        seen_a: list = []
        seen_b: list = []
        t.register(lambda batch: seen_a.extend(batch))
        t.register(lambda batch: seen_b.extend(batch))
        t.publish([("k", 1), ("k", 2)])
        assert seen_a == [("k", 1), ("k", 2)]
        assert seen_b == [("k", 1), ("k", 2)]


class TestPublish:
    def test_publish_empty_is_noop(self):
        t = InMemoryTransport()
        called: list = []
        t.register(lambda batch: called.append(batch))
        t.publish([])
        assert called == []

    def test_register_after_publish_does_not_receive_past(self):
        t = InMemoryTransport()
        seen: list = []
        t.publish([("a",)])
        t.register(lambda batch: seen.extend(batch))
        # First batch is already gone; a later publish is picked up.
        t.publish([("b",)])
        assert seen == [("b",)]

    def test_publish_forwards_iterable_as_list(self):
        t = InMemoryTransport()
        seen: list = []
        t.register(lambda batch: seen.extend(batch))
        # Pass a generator to prove publish materialises before fanout.
        t.publish(iter([("x",), ("y",)]))
        assert seen == [("x",), ("y",)]


class TestConcurrency:
    def test_unregister_during_publish_is_safe(self):
        """Concurrent unregister mid-publish must not crash the publisher."""
        t = InMemoryTransport()
        seen: list = []

        def slow_listener(batch):
            time.sleep(0.05)
            seen.extend(batch)

        def other(_batch):
            pass

        t.register(slow_listener)
        t.register(other)

        errors: list = []

        def publisher():
            try:
                t.publish([("k", i) for i in range(3)])
            except Exception as e:  # pragma: no cover
                errors.append(e)

        pub = threading.Thread(target=publisher)
        pub.start()
        # Let publisher enter the fanout, then remove the other listener.
        time.sleep(0.01)
        t.unregister(other)
        pub.join(timeout=2.0)

        assert not errors
        assert seen == [("k", 0), ("k", 1), ("k", 2)]
