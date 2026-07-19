"""Tests for the split InMemoryPublisher."""

from __future__ import annotations

import pytest

from virtuals._backends.publishers.mem import InMemoryPublisher
from virtuals.tkv.publisher import PublisherConnectionError
from virtuals.tkv.transport import InMemoryTransport


class RecordingTransport(InMemoryTransport):
    def __init__(self):
        super().__init__()
        self.batches: list[list] = []
        # single listener that records the batches it sees
        self.register(lambda batch: self.batches.append(list(batch)))


def test_notify_enqueues_and_publishes_batch():
    t = RecordingTransport()
    with InMemoryPublisher(t) as pub:
        pub.notify(("a", 1))
        pub.notify(("a", 2))
        pub.flush(timeout=1.0)
    assert t.batches, "expected at least one batch on the transport"
    # Batches are drained best-effort; flatten and assert content is present.
    flat = [k for b in t.batches for k in b]
    assert ("a", 1) in flat
    assert ("a", 2) in flat


def test_notify_accepts_iterable_of_keys():
    t = RecordingTransport()
    with InMemoryPublisher(t) as pub:
        pub.notify([("x", 1), ("x", 2), ("x", 3)])
        pub.flush(timeout=1.0)
    flat = [k for b in t.batches for k in b]
    assert set(flat) == {("x", 1), ("x", 2), ("x", 3)}


def test_notify_before_connect_raises():
    t = RecordingTransport()
    pub = InMemoryPublisher(t)
    with pytest.raises(PublisherConnectionError):
        pub.notify(("a",))


def test_disconnect_stops_worker_thread():
    t = RecordingTransport()
    pub = InMemoryPublisher(t)
    pub.connect()
    thread = pub._thread
    assert thread is not None and thread.is_alive()
    pub.disconnect()
    assert not thread.is_alive()


def test_reconnect_produces_fresh_queue():
    t = RecordingTransport()
    pub = InMemoryPublisher(t)
    pub.connect()
    pub.notify(("k", 1))
    pub.flush(timeout=1.0)
    pub.disconnect()
    # Second lifecycle: must not carry over queue state.
    pub.connect()
    pub.notify(("k", 2))
    pub.flush(timeout=1.0)
    pub.disconnect()
    flat = [k for b in t.batches for k in b]
    assert ("k", 1) in flat
    assert ("k", 2) in flat


def test_flush_drains_before_returning():
    t = RecordingTransport()
    with InMemoryPublisher(t) as pub:
        pub.notify([("q", i) for i in range(100)])
        pub.flush(timeout=2.0)
    flat = [k for b in t.batches for k in b]
    assert len(flat) == 100
