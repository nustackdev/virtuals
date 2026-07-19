"""Cross-compat integration: split RedisPublisher + split RedisObserver.

Confirms the new pair reproduces the same end-to-end wire behaviour as the
old fused pair.
"""

from __future__ import annotations

import time

import pytest


pytestmark = pytest.mark.redis


def _wait_until(pred, timeout=2.0, interval=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return pred()


def test_new_pair_end_to_end(redis_url, unique_channel_prefix, redis_cleanup):
    """Split Publisher + split Observer over Redis: subscribe, publish, callback fires."""
    from virtuals._backends.observers.redis_pubsub import RedisObserver
    from virtuals._backends.publishers.redis_pubsub import RedisPublisher
    from virtuals.tkv.filter import PrefixFilter
    from virtuals.tkv.observer import SubscriptionOptions

    redis_cleanup(unique_channel_prefix)

    obs = RedisObserver(redis_url=redis_url, channel_prefix=unique_channel_prefix)
    obs.connect()

    pub = RedisPublisher(
        redis_url=redis_url,
        channel_prefix=unique_channel_prefix,
        refresh_debounce_seconds=0.01,
    )
    pub.connect()

    try:
        hits: list = []
        sub = obs.subscribe(
            SubscriptionOptions(filter=PrefixFilter(prefix=("users",)))
        )
        sub.bind(lambda k: hits.append(k))

        # Publisher must pick up observer's registration via the control channel.
        assert _wait_until(lambda: len(pub._remote_registry) >= 1, timeout=2.0)

        pub.notify(("users", "alice"))
        pub.notify(("posts", "42"))  # should not match
        pub.flush(timeout=1.0)

        assert _wait_until(lambda: hits == [("users", "alice")], timeout=2.0), hits
    finally:
        pub.disconnect()
        obs.disconnect()


def test_new_pair_zero_interest_no_publishes(
    redis_url, unique_channel_prefix, redis_cleanup
):
    """No observers anywhere -> new publisher never PUBLISHes."""
    import redis as _redis

    from virtuals._backends.publishers.redis_pubsub import RedisPublisher

    redis_cleanup(unique_channel_prefix)

    probe = _redis.from_url(redis_url)
    pub = RedisPublisher(redis_url=redis_url, channel_prefix=unique_channel_prefix)
    pub.connect()
    try:
        for i in range(30):
            pub.notify(("data", i))
        pub.flush(timeout=1.0)
        time.sleep(0.2)

        chans = probe.pubsub_channels(pattern=f"{unique_channel_prefix}:notif:*")
        assert chans == [], f"expected zero notif channels, got {chans}"
    finally:
        pub.disconnect()
        try:
            probe.close()
        except Exception:  # noqa: S110
            pass


def test_new_pair_local_delivery_via_transport(
    redis_url, unique_channel_prefix, redis_cleanup
):
    """Same-process Publisher + Observer: local write reaches local sub via Redis.

    In the split design there is no local shortcut path; the write travels
    publisher -> Redis -> observer.
    """
    from virtuals._backends.observers.redis_pubsub import RedisObserver
    from virtuals._backends.publishers.redis_pubsub import RedisPublisher
    from virtuals.tkv.filter import PrefixFilter
    from virtuals.tkv.observer import SubscriptionOptions

    redis_cleanup(unique_channel_prefix)

    obs = RedisObserver(redis_url=redis_url, channel_prefix=unique_channel_prefix)
    obs.connect()
    pub = RedisPublisher(
        redis_url=redis_url,
        channel_prefix=unique_channel_prefix,
        refresh_debounce_seconds=0.01,
    )
    pub.connect()

    try:
        hits: list = []
        sub = obs.subscribe(SubscriptionOptions(filter=PrefixFilter(prefix=("x",))))
        sub.bind(lambda k: hits.append(k))

        assert _wait_until(lambda: len(pub._remote_registry) >= 1, timeout=2.0)

        pub.notify(("x", "y"))
        pub.flush(timeout=1.0)

        # Delivered exactly once via the transport.
        assert _wait_until(lambda: hits == [("x", "y")], timeout=2.0), hits
    finally:
        pub.disconnect()
        obs.disconnect()
