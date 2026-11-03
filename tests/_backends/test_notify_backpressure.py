"""Observer dispatch never drops silently: a full queue blocks the transport.

Publisher-side backpressure (notify blocks, flush raises with dropped_keys)
is covered in publishers/test_redis_publisher.py.
"""

from __future__ import annotations

import threading
import time

import pytest

from virtuals._backends.observers._base import ObserverBase
from virtuals.tkv.filter import PrefixFilter
from virtuals.tkv.observer import SubscriptionOptions


def _blocking_observer(maxsize):
    obs = ObserverBase(dispatch_queue_maxsize=maxsize)
    obs.connect()
    gate = threading.Event()
    got: list = []

    def cb(k):
        gate.wait(5)
        got.append(k)

    obs.subscribe(SubscriptionOptions(filter=PrefixFilter(prefix=("k",)))).bind(cb)
    return obs, gate, got


def test_full_dispatch_queue_blocks_then_delivers_all(caplog):
    obs, gate, got = _blocking_observer(2)
    try:
        keys = [("k", i) for i in range(50)]
        t = threading.Thread(target=obs._dispatch_incoming, args=(keys,), daemon=True)
        t.start()
        time.sleep(0.2)
        assert t.is_alive(), "producer should block on the full queue"
        gate.set()
        t.join(5)
        assert not t.is_alive()
        obs.flush(timeout=2.0)
        assert got == keys
        assert "dropping" not in caplog.text
    finally:
        gate.set()
        obs.disconnect()


def test_blocked_producer_released_by_disconnect():
    obs, gate, _ = _blocking_observer(1)
    t = threading.Thread(
        target=obs._dispatch_incoming, args=([("k", i) for i in range(20)],), daemon=True
    )
    t.start()
    time.sleep(0.1)
    assert t.is_alive()
    gate.set()  # let the worker see the stop marker / stop event
    obs.disconnect()
    t.join(2)
    assert not t.is_alive()


@pytest.mark.redis
def test_redis_slow_subscriber_loses_nothing(redis_url, unique_channel_prefix, redis_cleanup):
    pytest.importorskip("redis")
    pytest.importorskip("msgpack")
    from virtuals._backends.observers.redis_pubsub import RedisObserver
    from virtuals._backends.publishers.redis_pubsub import RedisPublisher

    redis_cleanup(unique_channel_prefix)
    obs = RedisObserver(
        redis_url=redis_url, channel_prefix=unique_channel_prefix, dispatch_queue_maxsize=4
    )
    obs.connect()
    pub = RedisPublisher(redis_url=redis_url, channel_prefix=unique_channel_prefix)
    pub.connect()
    gate = threading.Event()
    got: list = []
    try:

        def cb(k):
            gate.wait(5)
            got.append(k)

        obs.subscribe(SubscriptionOptions(filter=PrefixFilter(prefix=("k",)))).bind(cb)
        n = 2000
        for i in range(n):
            pub.notify(("k", i))
        pub.flush(timeout=5.0)
        time.sleep(0.2)
        assert len(got) == 0
        gate.set()
        deadline = time.monotonic() + 5
        while len(got) < n and time.monotonic() < deadline:
            time.sleep(0.02)
        assert sorted(got) == [("k", i) for i in range(n)]
    finally:
        gate.set()
        pub.disconnect()
        obs.disconnect()
