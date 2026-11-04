"""Tests for the split InMemoryObserver on a shared InMemoryTransport."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from virtuals._backends.observers.mem import InMemoryObserver
from virtuals._backends.publishers.mem import InMemoryPublisher
from virtuals.tkv.filter import Filter, PrefixFilter
from virtuals.tkv.observer import SubscriptionOptions
from virtuals.tkv.publisher import PublisherConnectionError, PublisherOperationError
from virtuals.tkv.transport import InMemoryTransport


if TYPE_CHECKING:
    from virtuals.tkv.types import Key


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


def test_resync_wakes_each_sub_on_the_hash_once_after_queued_keys():
    """A keyless resync wake reaches every sub on that filter hash, behind earlier keys."""
    t = InMemoryTransport()
    with InMemoryObserver(t) as obs:
        a: list = []
        a2: list = []
        b: list = []
        sa = obs.subscribe(SubscriptionOptions(filter=PrefixFilter(prefix=("a",))))
        sa.bind(a.append)
        obs.subscribe(SubscriptionOptions(filter=PrefixFilter(prefix=("a",)))).bind(a2.append)
        obs.subscribe(SubscriptionOptions(filter=PrefixFilter(prefix=("b",)))).bind(b.append)

        obs._dispatch_incoming([("a", 1)], filter_hash=sa.filter_hash)
        obs._dispatch_resync(sa.filter_hash)
        obs.flush(timeout=1.0)

        assert a == [("a", 1), None]
        assert a2 == [("a", 1), None]
        assert b == []


# =============================================================================
# CUSTOM FILTERS WITHOUT to_dict()
# =============================================================================


@dataclass(frozen=True, slots=True)
class _RegexFilter(Filter):
    """The filtering page's custom filter: no ``to_dict()``, so no ``filter_hash``."""

    pattern: str

    def matches(self, key: Key) -> bool:
        return bool(re.match(self.pattern, "/".join(map(str, key))))

    def __hash__(self) -> int:
        return hash(("regex", self.pattern))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _RegexFilter) and other.pattern == self.pattern


def test_custom_filter_without_to_dict_costs_nobody_their_resync_wake():
    """One subscription that cannot be hashed never blocks the wake of the others.

    The custom subscription gets a keyless wake of its own, keyed by the
    subscription (process-local, never sent across a transport). A publisher
    that dropped keys sends its wake once and does not retry for ever.
    """
    t = InMemoryTransport()
    with InMemoryObserver(t) as a, InMemoryObserver(t) as b:
        custom: list = []
        prefix_a: list = []
        prefix_b: list = []
        a.subscribe(SubscriptionOptions(filter=_RegexFilter("^c/"))).bind(custom.append)
        a.subscribe(SubscriptionOptions(filter=PrefixFilter(prefix=("p",)))).bind(prefix_a.append)
        b.subscribe(SubscriptionOptions(filter=PrefixFilter(prefix=("p",)))).bind(prefix_b.append)

        t.resync()  # what a publisher sends after dropping keys
        a.flush(timeout=1.0)
        b.flush(timeout=1.0)
        assert prefix_a == [None]
        assert prefix_b == [None]
        assert custom == [None]

        # the publisher's own path: a drop, then the wake goes out once
        pub = InMemoryPublisher(t)
        with pytest.raises(PublisherConnectionError):
            pub.notify(("p", 1))  # not connected: a counted drop
        pub.connect()
        try:
            with pytest.raises(PublisherOperationError):
                pub.flush(timeout=1.0)
            deadline = time.monotonic() + 3.0
            while pub._resync_pending and time.monotonic() < deadline:
                time.sleep(0.01)
            assert not pub._resync_pending
            pub.notify(("c", 1))
            pub.notify(("p", 2))
            pub.flush(timeout=1.0)
        finally:
            pub.disconnect()
        a.flush(timeout=1.0)
        b.flush(timeout=1.0)
        assert prefix_a == [None, None, ("p", 2)]
        assert prefix_b == [None, None, ("p", 2)]
        assert custom == [None, None, ("c", 1)]
