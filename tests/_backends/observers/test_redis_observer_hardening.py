"""RedisObserver under backlog, stalls, failed connects and drops.

Each test here pins one finding of the notify audit: a full dispatch queue
must not block subscribe() (N1), a stalled server must not hold subscribe()
past its deadline (N2), a failed connect leaks nothing (N7), closing
subscriptions after disconnect queues nothing (F12), a dropped observer is
collected (F8).
"""

from __future__ import annotations

import gc
import os
import signal
import threading
import time
import weakref

import pytest


pytestmark = pytest.mark.redis

redis = pytest.importorskip("redis")
pytest.importorskip("msgpack")

from virtuals.tkv.filter import PrefixFilter  # noqa: E402
from virtuals.tkv.observer import (  # noqa: E402
    ObserverConnectionError,
    ObserverSubscriptionError,
    SubscriptionOptions,
)


def _wait_until(pred, timeout=5.0, interval=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return pred()


def _threads() -> set[str]:
    return {t.name for t in threading.enumerate()}


@pytest.fixture
def own_valkey(tmp_path):
    """A private server per test, for tests that stop or restart it."""
    pytest.importorskip("valkeylite")
    from tests.support.valkey import private_valkey

    srv = private_valkey(tmp_path)
    yield srv
    srv.close()


@pytest.fixture
def prefix(unique_channel_prefix, redis_cleanup):
    redis_cleanup(unique_channel_prefix)
    return unique_channel_prefix


def _pair(url, prefix, **obs_kw):
    from virtuals._backends.observers.redis_pubsub import RedisObserver
    from virtuals._backends.publishers.redis_pubsub import RedisPublisher

    obs = RedisObserver(redis_url=url, channel_prefix=prefix, **obs_kw)
    obs.connect()
    pub = RedisPublisher(redis_url=url, channel_prefix=prefix)
    pub.connect()
    return obs, pub


# =============================================================================
# N1: a full dispatch queue does not block subscribe()
# =============================================================================


def test_subscribe_from_another_thread_under_a_full_dispatch_queue(redis_url, prefix):
    obs, pub = _pair(redis_url, prefix, dispatch_queue_maxsize=10, subscribe_timeout=1.0)
    release = threading.Event()
    slow_got: list = []
    other_got: list = []
    try:

        def slow(k):
            release.wait(10)
            slow_got.append(k)

        obs.subscribe(SubscriptionOptions(filter=PrefixFilter(prefix=("s",)))).bind(slow)
        keys = [("s", i) for i in range(200)]
        pub.notify(keys)
        pub.flush(2)
        assert _wait_until(lambda: obs._dispatch_queue.full())

        t0 = time.monotonic()
        sub = obs.subscribe(SubscriptionOptions(filter=PrefixFilter(prefix=("other",))))
        assert time.monotonic() - t0 < 0.5, "subscribe waited for callback queue room"
        sub.bind(other_got.append)
        # live at once: routed while the backlog is still parked
        pub.notify(("other", 1))
        pub.flush(2)
        time.sleep(0.2)
        assert other_got == [], "delivery order broke: the backlog must go first"

        release.set()
        assert _wait_until(lambda: len(slow_got) == 200 and other_got == [("other", 1)], 10)
        assert slow_got == keys
    finally:
        release.set()
        pub.disconnect()
        obs.disconnect()


def test_subscribe_from_inside_a_callback_under_a_full_dispatch_queue(redis_url, prefix):
    obs, pub = _pair(redis_url, prefix, dispatch_queue_maxsize=5, subscribe_timeout=1.0)
    nested: list = []
    errors: list = []
    gate = threading.Event()
    try:

        def cb(k):
            if k == ("s", 0):
                gate.wait(10)  # let the queue fill behind this callback
                try:
                    nested.append(obs.subscribe(SubscriptionOptions(filter=PrefixFilter(("n",)))))
                except ObserverSubscriptionError as e:
                    errors.append(e)

        obs.subscribe(SubscriptionOptions(filter=PrefixFilter(prefix=("s",)))).bind(cb)
        pub.notify([("s", i) for i in range(100)])
        pub.flush(2)
        assert _wait_until(lambda: obs._dispatch_queue.full())
        gate.set()
        assert _wait_until(lambda: nested or errors, 5)
        assert not errors, errors[0] if errors else None
    finally:
        gate.set()
        pub.disconnect()
        obs.disconnect()


def test_overflow_is_bounded_and_nothing_is_lost(redis_url, prefix):
    """Past queue + overflow the listener stops reading; valkey buffers; all arrive."""
    obs, pub = _pair(redis_url, prefix, dispatch_queue_maxsize=8)
    release = threading.Event()
    got: list = []
    try:

        def slow(k):
            release.wait(10)
            got.append(k)

        obs.subscribe(SubscriptionOptions(filter=PrefixFilter(prefix=("s",)))).bind(slow)
        n = 3000
        pub.notify([("s", i) for i in range(n)])
        pub.flush(5)
        time.sleep(0.3)
        assert len(obs._overflow) <= 8 + 250  # cap, plus at most one message past it
        release.set()
        assert _wait_until(lambda: len(got) == n, 15)
        assert got == [("s", i) for i in range(n)]
    finally:
        release.set()
        pub.disconnect()
        obs.disconnect()


# =============================================================================
# N2: a stalled server bounds subscribe() and nu1_reg
# =============================================================================


def test_subscribe_during_a_stalled_registration_is_bounded(own_valkey, monkeypatch):
    from virtuals._backends.observers.redis_pubsub import RedisObserver

    obs = RedisObserver(
        redis_url=own_valkey.url, channel_prefix="st", subscribe_timeout=0.5, socket_timeout=0.3
    )
    obs.connect()
    pid = own_valkey._server._process.pid
    real = obs._register_hash

    def stalling(h, entries):
        os.kill(pid, signal.SIGSTOP)  # the ack is in; nu1_reg hangs
        return real(h, entries)

    monkeypatch.setattr(obs, "_register_hash", stalling)
    try:
        t0 = time.monotonic()
        with pytest.raises(ObserverSubscriptionError):
            obs.subscribe(SubscriptionOptions(filter=PrefixFilter(prefix=("a",))))
        # one attempt plus redis-py's single retry, each bounded by the socket timeout
        assert time.monotonic() - t0 < 1.5
        assert obs._local_hash_refcount == {}
    finally:
        os.kill(pid, signal.SIGCONT)
        obs.disconnect()


# =============================================================================
# N7: a failed connect leaves no server client behind
# =============================================================================


def test_failed_connect_closes_what_it_opened(redis_url, prefix):
    from virtuals._backends.observers.redis_pubsub import RedisObserver

    probe = redis.from_url(redis_url)
    try:
        before = len(probe.client_list())
        obs = RedisObserver(redis_url=redis_url, channel_prefix=prefix)

        def boom(conn):
            raise RuntimeError("boom")

        obs._lua.ensure = boom
        with pytest.raises(ObserverConnectionError, match="boom"):
            obs.connect()
        assert obs._redis is None and obs._redis_sub is None and obs._pubsub is None
        assert _wait_until(lambda: len(probe.client_list()) == before, 2)
        obs.disconnect()  # no-op
        assert len(probe.client_list()) == before
    finally:
        probe.close()


# =============================================================================
# F12: subscription close after disconnect queues nothing
# =============================================================================


def test_subscription_close_after_disconnect_queues_nothing(redis_url, prefix):
    from virtuals._backends.observers.redis_pubsub import RedisObserver

    obs = RedisObserver(redis_url=redis_url, channel_prefix=prefix)
    obs.connect()
    subs = [obs.subscribe(SubscriptionOptions(filter=PrefixFilter(("b", i)))) for i in range(5)]
    obs.disconnect()
    for s in subs:
        s.close()
    assert obs._control_cmd_queue is None


# =============================================================================
# F8 and lifecycle: cycles and drops leave no thread or client
# =============================================================================


def test_connect_cycles_leave_no_thread_or_client(redis_url, prefix):
    probe = redis.from_url(redis_url)
    try:
        base_threads = _threads()
        base_clients = len(probe.client_list())
        for i in range(3):
            obs, pub = _pair(redis_url, f"{prefix}:{i}")
            got: list = []
            obs.subscribe(SubscriptionOptions(filter=PrefixFilter(prefix=("l",)))).bind(got.append)
            pub.notify(("l", i))
            pub.flush(2)
            assert _wait_until(lambda got=got, i=i: got == [("l", i)])
            pub.disconnect()
            obs.disconnect()
            assert _threads() == base_threads
            assert _wait_until(lambda: len(probe.client_list()) == base_clients, 2)
    finally:
        probe.close()


def test_dropped_observer_is_collected_and_releases_everything(redis_url, prefix):
    from virtuals._backends.observers.redis_pubsub import RedisObserver
    from virtuals._backends.publishers.redis_pubsub import RedisPublisher

    probe = redis.from_url(redis_url)
    pub = RedisPublisher(redis_url=redis_url, channel_prefix=prefix)
    pub.connect()
    try:
        base_threads = _threads()
        base_clients = len(probe.client_list())
        obs = RedisObserver(redis_url=redis_url, channel_prefix=prefix)
        obs.connect()
        got: list = []
        obs.subscribe(SubscriptionOptions(filter=PrefixFilter(prefix=("l",)))).bind(got.append)
        pub.notify(("l", 1))
        pub.flush(2)
        assert _wait_until(lambda: got == [("l", 1)])
        ref = weakref.ref(obs)
        del obs

        def collected():
            gc.collect()  # the registry <-> subscription <-> observer cycle needs a pass
            return ref() is None

        assert _wait_until(collected, 3), "a thread or connection kept the observer alive"
        assert _wait_until(lambda: _threads() == base_threads, 5), _threads() - base_threads
        assert _wait_until(lambda: len(probe.client_list()) == base_clients, 2)
    finally:
        pub.disconnect()
        probe.close()


# =============================================================================
# review M1: subscribe behind a large backlog (hundreds of messages)
# =============================================================================


def test_subscribe_behind_hundreds_of_backlog_messages_returns_fast(redis_url, prefix):
    """The confirmation sits behind every message valkey buffered; with a
    SUBSCRIBE pending the listener reads back to back instead of pacing on
    queue room, so subscribe() is not `messages_ahead x 20ms`."""
    obs, pub = _pair(redis_url, prefix, dispatch_queue_maxsize=10, subscribe_timeout=2.0)
    release = threading.Event()
    n_slow = [0]
    other_got: list = []
    try:

        def slow(k):
            release.wait(30)
            n_slow[0] += 1

        obs.subscribe(SubscriptionOptions(filter=PrefixFilter(prefix=("s",)))).bind(slow)
        n = 50_000  # 200 messages of 250 keys
        pub.notify([("s", i) for i in range(n)])
        pub.flush(20)
        time.sleep(0.3)  # valkey has buffered the backlog for this client

        t0 = time.monotonic()
        sub = obs.subscribe(SubscriptionOptions(filter=PrefixFilter(prefix=("other",))))
        assert time.monotonic() - t0 < 1.0, "subscribe paced by the backlog"
        sub.bind(other_got.append)
        pub.notify(("other", 1))
        pub.flush(2)
    finally:
        release.set()
        try:
            assert _wait_until(lambda: n_slow[0] == n and other_got == [("other", 1)], 30)
        finally:
            pub.disconnect()
            obs.disconnect()


# =============================================================================
# review m1: a listener stuck past the join keeps its reference, connect refuses
# =============================================================================


def test_listener_outliving_disconnect_blocks_the_next_connect(redis_url, prefix, caplog):
    import logging

    from virtuals._backends.observers.redis_pubsub import RedisObserver

    obs = RedisObserver(redis_url=redis_url, channel_prefix=prefix)
    obs.connect()
    block = threading.Event()
    entered = threading.Event()

    def stuck(stop):
        entered.set()
        block.wait(10)
        return False

    obs._listener_step = stuck
    assert entered.wait(3)
    old = obs._listener_thread
    caplog.set_level(logging.WARNING, logger="virtuals._backends.observers.redis_pubsub")
    try:
        obs.disconnect()
        assert old.is_alive()
        assert obs._listener_thread is old
        assert any(old.name in r.message for r in caplog.records)
        with pytest.raises(ObserverConnectionError, match="still running"):
            obs.connect()
    finally:
        block.set()
        old.join(3)
    assert not old.is_alive()
    obs.connect()
    obs.disconnect()


# =============================================================================
# r2 N3: W dropping publishers cost one keyless wake per subscription, not W
# =============================================================================


def test_resync_wakes_from_several_publishers_collapse_into_one(redis_url, prefix):
    """Three publishers that all dropped during one stall each send their wake;
    a subscription whose worker is busy gets one keyless wake, then keyed delivery."""
    from virtuals._backends.publishers.redis_pubsub import RedisPublisher

    obs, pub = _pair(redis_url, prefix)
    others = [RedisPublisher(redis_url=redis_url, channel_prefix=prefix) for _ in range(2)]
    for p in others:
        p.connect()
    hold = threading.Event()
    got: list = []
    try:

        def slow(key):
            hold.wait(10)
            got.append(key)

        handled = [0]
        real_handle = obs._handle_notif

        def counting_handle(msg, h):
            real_handle(msg, h)
            handled[0] += 1

        obs._handle_notif = counting_handle
        obs.subscribe(SubscriptionOptions(filter=PrefixFilter(("a",)))).bind(slow)
        pub.notify([("a", 0)])
        pub.flush(5)
        assert _wait_until(lambda: obs._dispatch_queue.qsize() == 0)  # worker inside slow()
        for p in (pub, *others):
            p._send_resync()  # what each publisher does once it can publish again
        # all three wakes have reached the listener before the worker is released
        assert _wait_until(lambda: handled[0] == 4), handled
        hold.set()
        obs.flush(5)
        assert _wait_until(lambda: len(got) >= 2)
        time.sleep(0.2)
        assert got == [("a", 0), None], got
        pub.notify([("a", 1)])
        pub.flush(5)
        assert _wait_until(lambda: got[-1] == ("a", 1)), got
        assert got.count(None) == 1
    finally:
        hold.set()
        for p in others:
            p.disconnect()
        pub.disconnect()
        obs.disconnect()


# =============================================================================
# review B1: a drop wake that found no subscription does not block the next one
# =============================================================================


def test_drop_wake_to_an_empty_registry_does_not_block_the_next_one(redis_url, prefix):
    """The only subscription closes before a parked drop wake is delivered; a new
    subscription on the same filter still gets the next publisher's drop wake."""
    obs, pub = _pair(redis_url, prefix)
    hold = threading.Event()
    handled = [0]
    real_handle = obs._handle_notif

    def counting_handle(msg, h):
        real_handle(msg, h)
        handled[0] += 1

    obs._handle_notif = counting_handle
    try:
        blk = obs.subscribe(SubscriptionOptions(filter=PrefixFilter(("a",))))
        blk.bind(lambda k: hold.wait(10))
        pub.notify([("a", 0)])
        pub.flush(5)
        assert _wait_until(lambda: obs._dispatch_queue.qsize() == 0)  # worker inside blk
        pub._send_resync()
        assert _wait_until(lambda: handled[0] == 2)  # the wake is parked for hash(a)
        blk.close()
        hold.set()
        obs.flush(5)
        assert _wait_until(lambda: not obs._resync_queued), obs._resync_queued
        got: list = []
        obs.subscribe(SubscriptionOptions(filter=PrefixFilter(("a",)))).bind(got.append)
        pub._send_resync()
        assert _wait_until(lambda: got == [None]), got
    finally:
        hold.set()
        pub.disconnect()
        obs.disconnect()
