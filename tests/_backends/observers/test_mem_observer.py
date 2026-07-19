"""Tests for the split InMemoryObserver on a shared InMemoryTransport."""

from __future__ import annotations

from virtuals._backends.observers.mem import InMemoryObserver
from virtuals._backends.publishers.mem import InMemoryPublisher
from virtuals.tkv.filter import PrefixFilter
from virtuals.tkv.observer import SubscriptionOptions
from virtuals.tkv.transport import InMemoryTransport


def test_subscribe_then_publish_fires_callback():
    t = InMemoryTransport()
    with InMemoryPublisher(t) as pub, InMemoryObserver(t) as obs:
        hits: list = []
        sub = obs.subscribe(SubscriptionOptions(filter=PrefixFilter(prefix=("u",))))
        sub.bind(lambda k: hits.append(k))

        pub.notify(("u", "alice"))
        pub.notify(("other", "x"))
        pub.flush(timeout=1.0)
        obs.flush(timeout=1.0)

        assert hits == [("u", "alice")]


def test_multi_publisher_one_observer():
    """Six publishers writing on one shared transport all reach the observer."""
    t = InMemoryTransport()
    publishers = [InMemoryPublisher(t) for _ in range(6)]
    for p in publishers:
        p.connect()

    with InMemoryObserver(t) as obs:
        hits: list = []
        sub = obs.subscribe(SubscriptionOptions(filter=PrefixFilter(prefix=("data",))))
        sub.bind(lambda k: hits.append(k))

        for i, p in enumerate(publishers):
            p.notify(("data", i))
        for p in publishers:
            p.flush(timeout=1.0)
        obs.flush(timeout=1.0)

    for p in publishers:
        p.disconnect()

    assert sorted(hits) == [("data", i) for i in range(6)]


def test_close_subscription_stops_callbacks():
    t = InMemoryTransport()
    with InMemoryPublisher(t) as pub, InMemoryObserver(t) as obs:
        hits: list = []
        sub = obs.subscribe(SubscriptionOptions(filter=PrefixFilter(prefix=("u",))))
        sub.bind(lambda k: hits.append(k))

        pub.notify(("u", "a"))
        pub.flush(timeout=1.0)
        obs.flush(timeout=1.0)
        assert hits == [("u", "a")]

        sub.close()
        pub.notify(("u", "b"))
        pub.flush(timeout=1.0)
        obs.flush(timeout=1.0)
        assert hits == [("u", "a")]


def test_last_sub_close_leaves_observer_listener_registered():
    """Unsubscribing removes the callback, but the observer's transport listener
    stays registered until disconnect."""
    t = InMemoryTransport()
    with InMemoryPublisher(t) as pub, InMemoryObserver(t) as obs:
        sub = obs.subscribe(SubscriptionOptions(filter=PrefixFilter(prefix=("u",))))
        sub.bind(lambda k: None)
        # One listener registered on the transport by the observer.
        assert len(t._listeners) == 1
        sub.close()
        # Observer still registered on the transport -- lifecycle is separate.
        assert len(t._listeners) == 1

        # Prove we can re-subscribe and callbacks resume.
        hits: list = []
        sub2 = obs.subscribe(SubscriptionOptions(filter=PrefixFilter(prefix=("u",))))
        sub2.bind(lambda k: hits.append(k))
        pub.notify(("u", "x"))
        pub.flush(timeout=1.0)
        obs.flush(timeout=1.0)
        assert hits == [("u", "x")]

    # After disconnect the transport listener is gone.
    assert t._listeners == []
