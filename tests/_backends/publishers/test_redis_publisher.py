"""Tests for the split RedisPublisher over a real Redis server."""

from __future__ import annotations

import json
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


def _make_publisher(redis_url, channel_prefix, **kwargs):
    from virtuals._backends.publishers.redis_pubsub import RedisPublisher

    pub = RedisPublisher(
        redis_url=redis_url, channel_prefix=channel_prefix, **kwargs
    )
    pub.connect()
    return pub


def test_publish_writes_msgpack_payload_on_notif_channel(
    redis_url, unique_channel_prefix, redis_cleanup
):
    """Register a filter directly in the HASH, then publish a matching key.

    Prove the publisher writes the msgpack payload on the per-hash channel.
    """
    import msgpack
    import redis as _redis

    from virtuals.tkv.filter import PrefixFilter, filter_hash

    redis_cleanup(unique_channel_prefix)

    probe = _redis.from_url(redis_url)
    # Seed one filter in the registry HASH so publisher's cache picks it up.
    f = PrefixFilter(prefix=("users",))
    h = filter_hash(f)
    registry_key = f"{unique_channel_prefix}:subs:registry"
    probe.hset(registry_key, h, json.dumps(f.to_dict(), separators=(",", ":")))

    # External subscriber to observe the notif channel.
    sub_conn = _redis.from_url(redis_url)
    ps = sub_conn.pubsub()
    notif_channel = f"{unique_channel_prefix}:notif:{h}"
    ps.subscribe(notif_channel)
    # Drain the subscribe-confirmation message.
    for _ in range(5):
        m = ps.get_message(timeout=0.1)
        if m is not None and m.get("type") == "subscribe":
            break

    pub = _make_publisher(redis_url, unique_channel_prefix)

    try:
        # Ensure publisher's cache picked up the filter.
        assert _wait_until(lambda: len(pub._remote_registry) >= 1, timeout=2.0)

        pub.notify(("users", "alice"))
        pub.flush(timeout=1.0)

        received: dict | None = None
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and received is None:
            m = ps.get_message(timeout=0.1)
            if m is None:
                continue
            if m.get("type") not in ("message", "pmessage"):
                continue
            if m["channel"].decode("utf-8") != notif_channel:
                continue
            received = msgpack.unpackb(m["data"], raw=False)

        assert received is not None, "no publish received"
        assert received["iid"] == pub._instance_id
        assert received["keys"] == [["users", "alice"]]
    finally:
        pub.disconnect()
        try:
            ps.unsubscribe()
            ps.close()
            sub_conn.close()
            probe.close()
        except Exception:  # noqa: S110
            pass


def test_no_publish_when_no_cluster_interest(
    redis_url, unique_channel_prefix, redis_cleanup
):
    """Empty remote registry -> notify() produces zero publishes."""
    import redis as _redis

    redis_cleanup(unique_channel_prefix)

    probe = _redis.from_url(redis_url)
    pub = _make_publisher(redis_url, unique_channel_prefix)

    try:
        assert pub._remote_registry == {}

        for i in range(50):
            pub.notify(("data", i))
        pub.flush(timeout=1.0)
        time.sleep(0.2)

        notif_channels = probe.pubsub_channels(
            pattern=f"{unique_channel_prefix}:notif:*"
        )
        assert notif_channels == [], f"expected zero notif channels, got {notif_channels}"
    finally:
        pub.disconnect()
        try:
            probe.close()
        except Exception:  # noqa: S110
            pass


def test_filter_hash_bucketing_isolates_publishes(
    redis_url, unique_channel_prefix, redis_cleanup
):
    """Two registered filters -> publisher routes each key only to its bucket."""
    import msgpack
    import redis as _redis

    from virtuals.tkv.filter import PrefixFilter, filter_hash

    redis_cleanup(unique_channel_prefix)

    probe = _redis.from_url(redis_url)
    f_users = PrefixFilter(prefix=("users",))
    f_posts = PrefixFilter(prefix=("posts",))
    h_users = filter_hash(f_users)
    h_posts = filter_hash(f_posts)
    registry_key = f"{unique_channel_prefix}:subs:registry"
    probe.hset(registry_key, h_users, json.dumps(f_users.to_dict(), separators=(",", ":")))
    probe.hset(registry_key, h_posts, json.dumps(f_posts.to_dict(), separators=(",", ":")))

    sub_conn = _redis.from_url(redis_url)
    ps = sub_conn.pubsub()
    ch_users = f"{unique_channel_prefix}:notif:{h_users}"
    ch_posts = f"{unique_channel_prefix}:notif:{h_posts}"
    ps.subscribe(ch_users, ch_posts)
    for _ in range(6):
        m = ps.get_message(timeout=0.1)
        if m is None:
            continue

    pub = _make_publisher(redis_url, unique_channel_prefix)
    try:
        assert _wait_until(lambda: len(pub._remote_registry) >= 2, timeout=2.0)

        pub.notify(("users", "alice"))
        pub.notify(("posts", "42"))
        pub.notify(("comments", "x"))  # matches neither
        pub.flush(timeout=1.0)

        seen_users: list = []
        seen_posts: list = []
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            m = ps.get_message(timeout=0.1)
            if m is None:
                continue
            if m.get("type") not in ("message", "pmessage"):
                continue
            channel = m["channel"].decode("utf-8")
            payload = msgpack.unpackb(m["data"], raw=False)
            if channel == ch_users:
                seen_users.extend(payload["keys"])
            elif channel == ch_posts:
                seen_posts.extend(payload["keys"])
            if seen_users and seen_posts:
                break

        assert seen_users == [["users", "alice"]]
        assert seen_posts == [["posts", "42"]]
    finally:
        pub.disconnect()
        try:
            ps.unsubscribe()
            ps.close()
            sub_conn.close()
            probe.close()
        except Exception:  # noqa: S110
            pass


def test_control_channel_signal_triggers_refresh(
    redis_url, unique_channel_prefix, redis_cleanup
):
    """External HSET + control-channel publish -> publisher refreshes its cache."""
    import redis as _redis

    from virtuals.tkv.filter import PrefixFilter, filter_hash

    redis_cleanup(unique_channel_prefix)

    probe = _redis.from_url(redis_url)
    pub = _make_publisher(
        redis_url, unique_channel_prefix, refresh_debounce_seconds=0.01
    )
    try:
        assert pub._remote_registry == {}

        # Now inject a new filter and signal the control channel.
        f = PrefixFilter(prefix=("late",))
        h = filter_hash(f)
        probe.hset(
            f"{unique_channel_prefix}:subs:registry",
            h,
            json.dumps(f.to_dict(), separators=(",", ":")),
        )
        probe.publish(f"{unique_channel_prefix}:subs:changed", b"")

        assert _wait_until(
            lambda: h in pub._remote_registry, timeout=2.0
        ), "publisher did not refresh cache after control signal"
    finally:
        pub.disconnect()
        try:
            probe.close()
        except Exception:  # noqa: S110
            pass
