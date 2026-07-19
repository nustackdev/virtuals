"""Tests for the split RedisObserver over a real Redis server."""

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


def _make_observer(redis_url, channel_prefix, **kwargs):
    from virtuals._backends.observers.redis_pubsub import RedisObserver

    obs = RedisObserver(
        redis_url=redis_url, channel_prefix=channel_prefix, **kwargs
    )
    obs.connect()
    return obs


def test_subscribe_hsets_registry(redis_url, unique_channel_prefix, redis_cleanup):
    """First subscribe on a filter -> HSET into registry HASH."""
    import redis as _redis

    from virtuals.tkv.filter import PrefixFilter, filter_hash
    from virtuals.tkv.observer import SubscriptionOptions

    redis_cleanup(unique_channel_prefix)

    probe = _redis.from_url(redis_url)
    registry_key = f"{unique_channel_prefix}:subs:registry"
    f = PrefixFilter(prefix=("users",))
    h = filter_hash(f)

    obs = _make_observer(redis_url, unique_channel_prefix)
    try:
        sub = obs.subscribe(SubscriptionOptions(filter=f))
        sub.bind(lambda k: None)

        assert _wait_until(
            lambda: probe.hexists(registry_key, h), timeout=2.0
        ), "HASH did not get HSET on subscribe"
    finally:
        obs.disconnect()
        try:
            probe.close()
        except Exception:  # noqa: S110
            pass


def test_external_publish_fires_callback(
    redis_url, unique_channel_prefix, redis_cleanup
):
    """External raw client publishes a matching-payload msg -> observer callback fires."""
    import msgpack
    import redis as _redis

    from virtuals.tkv.filter import PrefixFilter, filter_hash
    from virtuals.tkv.observer import SubscriptionOptions

    redis_cleanup(unique_channel_prefix)

    probe = _redis.from_url(redis_url)

    obs = _make_observer(redis_url, unique_channel_prefix)
    try:
        hits: list = []
        f = PrefixFilter(prefix=("data",))
        sub = obs.subscribe(SubscriptionOptions(filter=f))
        sub.bind(lambda k: hits.append(k))

        # Wait until observer has SUBSCRIBEd on the notif channel.
        notif_channel = f"{unique_channel_prefix}:notif:{filter_hash(f)}"
        assert _wait_until(
            lambda: probe.pubsub_numsub(notif_channel)[0][1] >= 1, timeout=2.0
        )

        payload = msgpack.packb(
            {"iid": 999, "keys": [["data", "abc"]]},
            use_bin_type=True,
        )
        probe.publish(notif_channel, payload)

        assert _wait_until(lambda: hits == [("data", "abc")], timeout=2.0), hits
    finally:
        obs.disconnect()
        try:
            probe.close()
        except Exception:  # noqa: S110
            pass


def test_close_subscription_and_cleanup_hdels_registry(
    redis_url, unique_channel_prefix, redis_cleanup
):
    """After sub close + cleanup sweep, orphan HASH entry is HDELed."""
    import redis as _redis

    from virtuals.tkv.filter import PrefixFilter, filter_hash
    from virtuals.tkv.observer import SubscriptionOptions

    redis_cleanup(unique_channel_prefix)

    probe = _redis.from_url(redis_url)
    registry_key = f"{unique_channel_prefix}:subs:registry"
    f = PrefixFilter(prefix=("orphan",))
    h = filter_hash(f)

    obs = _make_observer(
        redis_url, unique_channel_prefix, cleanup_interval_seconds=0.2
    )
    try:
        sub = obs.subscribe(SubscriptionOptions(filter=f))
        sub.bind(lambda k: None)

        assert _wait_until(lambda: probe.hexists(registry_key, h), timeout=2.0)
        sub.close()

        assert _wait_until(
            lambda: not probe.hexists(registry_key, h), timeout=3.0
        ), f"orphan hash not swept: {probe.hgetall(registry_key)}"
    finally:
        obs.disconnect()
        try:
            probe.close()
        except Exception:  # noqa: S110
            pass


def test_cross_instance_publisher_reaches_observer(
    redis_url, unique_channel_prefix, redis_cleanup
):
    """New RedisPublisher (fresh conn) writes -> new RedisObserver receives."""
    from virtuals._backends.publishers.redis_pubsub import RedisPublisher
    from virtuals.tkv.filter import PrefixFilter
    from virtuals.tkv.observer import SubscriptionOptions

    redis_cleanup(unique_channel_prefix)

    obs = _make_observer(redis_url, unique_channel_prefix)
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

        # Wait until publisher's remote cache has picked up the observer's hash.
        assert _wait_until(lambda: len(pub._remote_registry) >= 1, timeout=2.0)

        pub.notify(("x", "y"))
        pub.flush(timeout=1.0)

        assert _wait_until(lambda: hits == [("x", "y")], timeout=2.0), hits
    finally:
        pub.disconnect()
        obs.disconnect()
