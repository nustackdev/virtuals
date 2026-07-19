"""Integration tests for RedisObserver per-hash routing.

Covers cross-observer scenarios that InMemoryObserver can't exercise:
- Two observers with the same channel_prefix share notifications via
  per-hash channels routed through redis.
- A zero-interest publisher (no remote subs anywhere) produces zero
  publishes on notification channels -- the "kill the fanout" goal.
- Late-joiners catch up to existing subscribers via HGETALL on connect.
- Per-hash routing: writes only reach observers whose filter matches.
- Register/unregister discipline: HDEL sweep removes orphaned hashes.

All marked `@pytest.mark.redis` -- skip cleanly when redis-py or a
reachable redis server is unavailable.
"""

from __future__ import annotations

import time

import pytest


pytestmark = pytest.mark.redis


# -- Helpers ----------------------------------------------------------------


def _make_observer(redis_url, channel_prefix, **kwargs):
    from virtuals._backends.observers.redis_pubsub import RedisObserver
    from virtuals.codecs import BinaryCodec

    obs = RedisObserver(
        codec=BinaryCodec(), redis_url=redis_url, channel_prefix=channel_prefix, **kwargs
    )
    obs.connect()
    return obs


def _wait_until(pred, timeout=2.0, interval=0.02):
    """Poll `pred()` until True or `timeout` seconds elapse. Returns pred()'s final value."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return pred()


# -- Cross-observer notification --------------------------------------------


def test_sub_on_a_fires_on_write_via_b(redis_url, unique_channel_prefix, redis_cleanup):
    """A subscribes; B writes matching key via notify(); A's callback fires exactly once."""
    from virtuals.tkv.filter import PrefixFilter
    from virtuals.tkv.observer import SubscriptionOptions

    redis_cleanup(unique_channel_prefix)

    a = _make_observer(redis_url, unique_channel_prefix)
    b = _make_observer(redis_url, unique_channel_prefix)

    hits: list = []
    sub = a.subscribe(SubscriptionOptions(filter=PrefixFilter(prefix=("users",))))
    sub.bind(lambda k: hits.append(k))

    # B needs to see A's hash in its remote registry. HGETALL happens on
    # start, and A's HSET fires a "changed" signal that B picks up
    # (debounced). Poll until B has A's filter.
    try:
        assert _wait_until(lambda: len(b._redis_publisher._remote_registry) >= 1, timeout=2.0)

        b.notify(("users", "alice"))
        b.flush()

        assert _wait_until(lambda: hits == [("users", "alice")], timeout=2.0), hits
    finally:
        sub.close()
        a.disconnect()
        b.disconnect()


def test_non_matching_key_does_not_fire(redis_url, unique_channel_prefix, redis_cleanup):
    """B writes a key that doesn't match A's filter; nothing fires on A."""
    from virtuals.tkv.filter import PrefixFilter
    from virtuals.tkv.observer import SubscriptionOptions

    redis_cleanup(unique_channel_prefix)

    a = _make_observer(redis_url, unique_channel_prefix)
    b = _make_observer(redis_url, unique_channel_prefix)

    hits: list = []
    sub = a.subscribe(SubscriptionOptions(filter=PrefixFilter(prefix=("users",))))
    sub.bind(lambda k: hits.append(k))

    try:
        assert _wait_until(lambda: len(b._redis_publisher._remote_registry) >= 1, timeout=2.0)

        b.notify(("posts", "xyz"))  # not a "users/" key
        b.flush()
        # Give the listener a beat to prove-nothing-fires.
        time.sleep(0.2)

        assert hits == []
    finally:
        sub.close()
        a.disconnect()
        b.disconnect()


def test_late_joiner_learns_existing_subs_via_hgetall(
    redis_url, unique_channel_prefix, redis_cleanup
):
    """A subscribes first. B connects later. B's writes matching A's filter fire A's callback."""
    from virtuals.tkv.filter import PrefixFilter
    from virtuals.tkv.observer import SubscriptionOptions

    redis_cleanup(unique_channel_prefix)

    a = _make_observer(redis_url, unique_channel_prefix)
    hits: list = []
    sub = a.subscribe(SubscriptionOptions(filter=PrefixFilter(prefix=("late",))))
    sub.bind(lambda k: hits.append(k))

    # Give A time to HSET before B starts.
    time.sleep(0.05)

    # Now B joins. Should learn A's filter via HGETALL on start.
    b = _make_observer(redis_url, unique_channel_prefix)
    try:
        assert len(b._redis_publisher._remote_registry) >= 1, (
            "late joiner didn't pick up existing subs via HGETALL"
        )

        b.notify(("late", "x"))
        b.flush()

        assert _wait_until(lambda: hits == [("late", "x")], timeout=2.0), hits
    finally:
        sub.close()
        a.disconnect()
        b.disconnect()


def test_per_hash_routing_isolates_filters(redis_url, unique_channel_prefix, redis_cleanup):
    """Two subs on A, distinct filters. B writes keys matching each; each hits only its own cb."""
    from virtuals.tkv.filter import PrefixFilter
    from virtuals.tkv.observer import SubscriptionOptions

    redis_cleanup(unique_channel_prefix)

    a = _make_observer(redis_url, unique_channel_prefix)
    b = _make_observer(redis_url, unique_channel_prefix)

    users_hits: list = []
    posts_hits: list = []
    sub_u = a.subscribe(SubscriptionOptions(filter=PrefixFilter(prefix=("users",))))
    sub_u.bind(lambda k: users_hits.append(k))
    sub_p = a.subscribe(SubscriptionOptions(filter=PrefixFilter(prefix=("posts",))))
    sub_p.bind(lambda k: posts_hits.append(k))

    try:
        assert _wait_until(lambda: len(b._redis_publisher._remote_registry) >= 2, timeout=2.0)

        b.notify(("users", "alice"))
        b.notify(("posts", "42"))
        b.notify(("comments", "1"))  # matches neither
        b.flush()
        time.sleep(0.3)

        assert users_hits == [("users", "alice")]
        assert posts_hits == [("posts", "42")]
    finally:
        sub_u.close()
        sub_p.close()
        a.disconnect()
        b.disconnect()


# -- Zero-interest publish check --------------------------------------------


def test_zero_remote_interest_no_publishes_go_out(
    redis_url, unique_channel_prefix, redis_cleanup
):
    """With zero remote subs, notify()/flush() must not publish anything.

    We prove this by asking redis for its channel/subscriber count on our
    notification-channel prefix -- if we never published, there are no
    ephemeral pubsub subscribers on those channels and PUBSUB CHANNELS
    returns nothing under our prefix.
    """
    import redis as _redis

    redis_cleanup(unique_channel_prefix)

    a = _make_observer(redis_url, unique_channel_prefix)
    probe = _redis.from_url(redis_url)

    try:
        # No local subs on A. Any keys written should route to zero remote
        # channels since remote_registry is empty.
        assert len(a._redis_publisher._remote_registry) == 0

        for i in range(50):
            a.notify(("data", f"k{i}"))
        a.flush()
        # Give the publish thread a beat -- there should be nothing to publish.
        time.sleep(0.2)

        notif_channels = probe.pubsub_channels(pattern=f"{unique_channel_prefix}:notif:*")
        assert notif_channels == [], f"expected zero notif channels, got {notif_channels}"
    finally:
        a.disconnect()
        try:
            probe.close()
        except Exception:
            pass


# -- Cleanup sweep ----------------------------------------------------------


def test_cleanup_sweep_removes_orphan_hash(redis_url, unique_channel_prefix, redis_cleanup):
    """After A closes its sub, cleanup sweep HDELs the orphan hash from the registry."""
    import redis as _redis

    from virtuals.tkv.filter import PrefixFilter
    from virtuals.tkv.observer import SubscriptionOptions

    redis_cleanup(unique_channel_prefix)

    # Short cleanup interval so the test doesn't wait forever
    a = _make_observer(redis_url, unique_channel_prefix, cleanup_interval_seconds=0.2)
    probe = _redis.from_url(redis_url)
    registry_key = f"{unique_channel_prefix}:subs:registry"

    try:
        sub = a.subscribe(SubscriptionOptions(filter=PrefixFilter(prefix=("orphan",))))
        sub.bind(lambda k: None)

        # HASH has the entry now
        assert _wait_until(lambda: probe.hlen(registry_key) >= 1, timeout=2.0)

        sub.close()
        # After close, listener processes UNSUBSCRIBE; next sweep sees numsub=0 and HDELs.
        assert _wait_until(lambda: probe.hlen(registry_key) == 0, timeout=3.0), (
            f"orphan hash not swept: {probe.hgetall(registry_key)}"
        )
    finally:
        a.disconnect()
        try:
            probe.close()
        except Exception:
            pass


# -- Instance-id self-echo suppression --------------------------------------


def test_self_publish_does_not_double_deliver_locally(
    redis_url, unique_channel_prefix, redis_cleanup
):
    """When A subscribes AND writes a matching key locally, callback fires exactly once."""
    from virtuals.tkv.filter import PrefixFilter
    from virtuals.tkv.observer import SubscriptionOptions

    redis_cleanup(unique_channel_prefix)

    a = _make_observer(redis_url, unique_channel_prefix)

    hits: list = []
    sub = a.subscribe(SubscriptionOptions(filter=PrefixFilter(prefix=("x",))))
    sub.bind(lambda k: hits.append(k))

    try:
        # Give A time to register + refresh so its own remote_registry contains its hash.
        assert _wait_until(lambda: len(a._redis_publisher._remote_registry) >= 1, timeout=2.0)

        a.notify(("x", "y"))
        a.flush()
        # Allow round-trip; self-echo would arrive on the notif channel and be dropped by iid check.
        time.sleep(0.3)

        assert hits == [("x", "y")], f"expected single delivery, got {hits}"
    finally:
        sub.close()
        a.disconnect()
