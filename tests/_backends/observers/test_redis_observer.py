"""RedisObserver over a real server: confirmed subscribe, nu1_reg, readiness."""

from __future__ import annotations

import threading
import time

import pytest


pytestmark = pytest.mark.redis

redis = pytest.importorskip("redis")
msgpack = pytest.importorskip("msgpack")


def _wait_until(pred, timeout=2.0, interval=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return pred()


def _make_observer(redis_url, channel_prefix, **kwargs):
    from virtuals._backends.observers.redis_pubsub import RedisObserver

    obs = RedisObserver(redis_url=redis_url, channel_prefix=channel_prefix, **kwargs)
    obs.connect()
    return obs


def _make_publisher(redis_url, channel_prefix):
    from virtuals._backends.publishers.redis_pubsub import RedisPublisher

    pub = RedisPublisher(redis_url=redis_url, channel_prefix=channel_prefix)
    pub.connect()
    return pub


@pytest.fixture
def prefix(unique_channel_prefix, redis_cleanup):
    redis_cleanup(unique_channel_prefix)
    return unique_channel_prefix


@pytest.fixture
def probe(redis_url):
    c = redis.from_url(redis_url)
    yield c
    c.close()


def test_subscribe_registers_once_confirmed(redis_url, prefix, probe):
    """On return from subscribe() the channel is subscribed and the hash indexed."""
    from virtuals.tkv.filter import PrefixFilter, filter_hash
    from virtuals.tkv.observer import SubscriptionOptions

    f = PrefixFilter(prefix=("users",))
    h = filter_hash(f)
    obs = _make_observer(redis_url, prefix)
    try:
        obs.subscribe(SubscriptionOptions(filter=f))
        # no waiting: both hold the moment subscribe() returns
        assert probe.hexists(f"{prefix}:nx1:refs", h)
        assert probe.pubsub_numsub(f"{prefix}:notif:{h}")[0][1] == 1
        assert not probe.exists(f"{prefix}:subs:registry")
    finally:
        obs.disconnect()


def test_external_publish_fires_callback(redis_url, prefix, probe):
    """External raw client publishes a matching-payload msg -> observer callback fires."""
    from virtuals.tkv.filter import PrefixFilter, filter_hash
    from virtuals.tkv.observer import SubscriptionOptions

    obs = _make_observer(redis_url, prefix)
    try:
        hits: list = []
        f = PrefixFilter(prefix=("data",))
        sub = obs.subscribe(SubscriptionOptions(filter=f))
        sub.bind(lambda k: hits.append(k))

        payload = msgpack.packb({"iid": 999, "keys": [["data", "abc"]]}, use_bin_type=True)
        assert probe.publish(f"{prefix}:notif:{filter_hash(f)}", payload) == 1
        assert _wait_until(lambda: hits == [("data", "abc")], timeout=2.0), hits
    finally:
        obs.disconnect()


def test_close_unsubscribes_and_route_removes_lazily(redis_url, prefix, probe):
    """Last close -> UNSUBSCRIBE; the next routed write drops the index entry."""
    from virtuals.tkv.filter import PrefixFilter, filter_hash
    from virtuals.tkv.observer import SubscriptionOptions

    f = PrefixFilter(prefix=("orphan",))
    h = filter_hash(f)
    obs = _make_observer(redis_url, prefix)
    pub = _make_publisher(redis_url, prefix)
    try:
        sub = obs.subscribe(SubscriptionOptions(filter=f))
        assert probe.hexists(f"{prefix}:nx1:refs", h)
        sub.close()
        assert _wait_until(lambda: probe.pubsub_numsub(f"{prefix}:notif:{h}")[0][1] == 0)

        pub.notify(("orphan", 1))
        pub.flush(timeout=1.0)
        assert not probe.hexists(f"{prefix}:nx1:refs", h)
        assert list(probe.scan_iter(match=f"{prefix}:nx1:*")) == []
    finally:
        pub.disconnect()
        obs.disconnect()


def test_cross_instance_publisher_reaches_observer(redis_url, prefix):
    """New RedisPublisher (fresh conn) writes right after subscribe -> observer receives."""
    from virtuals.tkv.filter import PrefixFilter
    from virtuals.tkv.observer import SubscriptionOptions

    obs = _make_observer(redis_url, prefix)
    pub = _make_publisher(redis_url, prefix)
    try:
        hits: list = []
        sub = obs.subscribe(SubscriptionOptions(filter=PrefixFilter(prefix=("x",))))
        sub.bind(lambda k: hits.append(k))

        pub.notify(("x", "y"))
        pub.flush(timeout=1.0)
        assert _wait_until(lambda: hits == [("x", "y")], timeout=2.0), hits
    finally:
        pub.disconnect()
        obs.disconnect()


def test_pass_none_is_local_only(redis_url, prefix, probe):
    """A filter that matches nothing never SUBSCRIBEs or registers."""
    from virtuals.tkv.filter import PassNone
    from virtuals.tkv.observer import SubscriptionOptions

    obs = _make_observer(redis_url, prefix)
    try:
        sub = obs.subscribe(SubscriptionOptions(filter=PassNone()))
        assert probe.pubsub_channels(pattern=f"{prefix}:*") == []
        assert list(probe.scan_iter(match=f"{prefix}:nx1:*")) == []
        sub.close()
    finally:
        obs.disconnect()


def test_concurrent_same_filter_subscribe(redis_url, prefix):
    """8 threads subscribe one filter at once: one SUBSCRIBE, all live on return."""
    from virtuals.tkv.filter import PrefixFilter
    from virtuals.tkv.observer import SubscriptionOptions

    obs = _make_observer(redis_url, prefix)
    pub = _make_publisher(redis_url, prefix)
    n = 8
    try:
        for trial in range(10):
            f = PrefixFilter(prefix=("c", trial))
            barrier = threading.Barrier(n)
            hits = [threading.Event() for _ in range(n)]
            subs: list = [None] * n
            errors: list = []

            def worker(i, f=f, barrier=barrier, hits=hits, subs=subs, trial=trial, errors=errors):
                barrier.wait()
                try:
                    s = obs.subscribe(SubscriptionOptions(filter=f))
                    s.bind(lambda k, i=i: hits[i].set())
                    subs[i] = s
                    # write the moment this subscribe returned
                    pub.notify(("c", trial, i))
                except Exception as e:  # pragma: no cover - surfaced below
                    errors.append(e)

            ts = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
            for t in ts:
                t.start()
            for t in ts:
                t.join(5)
            assert not errors
            assert all(h.wait(2.0) for h in hits), [h.is_set() for h in hits]
            assert obs._local_hash_refcount == {subs[0].filter_hash: n}
            for s in subs:
                s.close()
            assert obs._local_hash_refcount == {}
    finally:
        pub.disconnect()
        obs.disconnect()


@pytest.fixture
def own_valkey(tmp_path):
    """A private server per test, for tests that pause or restart it."""
    support = pytest.importorskip("tests.support.valkey")
    pytest.importorskip("valkeylite")
    srv = support.private_valkey(tmp_path)
    yield srv
    srv.close()


def test_ack_timeout_raises_and_rolls_back(own_valkey):
    """A stalled server: subscribe raises (first and concurrent waiter), then recovers."""
    from virtuals.tkv.filter import PrefixFilter
    from virtuals.tkv.observer import ObserverSubscriptionError, SubscriptionOptions

    url = own_valkey.url
    obs = _make_observer(url, "t", subscribe_timeout=0.2)
    pub = _make_publisher(url, "t")
    admin = redis.from_url(url)
    try:
        f = PrefixFilter(prefix=("slow",))
        admin.execute_command("CLIENT", "PAUSE", 700, "ALL")
        errors: list = []

        def run():
            try:
                obs.subscribe(SubscriptionOptions(filter=f))
            except ObserverSubscriptionError as e:
                errors.append(str(e))

        t0 = time.monotonic()
        ts = [threading.Thread(target=run) for _ in range(2)]
        for t in ts:  # the second is a waiter on the first's readiness
            t.start()
            time.sleep(0.02)
        for t in ts:
            t.join(2)
        assert time.monotonic() - t0 < 0.6
        assert len(errors) == 2
        assert any("not confirmed" in e for e in errors)
        assert len(obs._registry) == 0
        assert obs._local_hash_refcount == {}

        # Pause over: the stale SUBSCRIBE is confirmed and undone; a fresh
        # subscribe goes live and delivers at 0ms.
        time.sleep(0.6)
        hit = threading.Event()
        sub = obs.subscribe(SubscriptionOptions(filter=f))
        sub.bind(lambda k: hit.set())
        pub.notify(("slow", 1))
        assert hit.wait(2.0)
    finally:
        admin.close()
        pub.disconnect()
        obs.disconnect()


def test_failed_registration_fails_waiters(redis_url, prefix, monkeypatch):
    """nu1_reg error in the first registrant raises in every concurrent waiter."""
    from virtuals.tkv.filter import PrefixFilter
    from virtuals.tkv.observer import ObserverSubscriptionError, SubscriptionOptions

    obs = _make_observer(redis_url, prefix)
    try:
        started = threading.Event()
        release = threading.Event()

        def boom(h, entries):
            started.set()
            release.wait(2)
            raise redis.ResponseError("boom")

        monkeypatch.setattr(obs, "_register_hash", boom)
        f = PrefixFilter(prefix=("bad",))
        errors: list = []

        def run():
            try:
                obs.subscribe(SubscriptionOptions(filter=f))
            except ObserverSubscriptionError as e:
                errors.append(e)

        first = threading.Thread(target=run)
        first.start()
        assert started.wait(2)
        waiters = [threading.Thread(target=run) for _ in range(3)]
        for w in waiters:
            w.start()
        time.sleep(0.05)
        release.set()
        for t in [first, *waiters]:
            t.join(3)
        assert len(errors) == 4
        assert len(obs._registry) == 0
        assert obs._local_hash_refcount == {}
        assert obs._local == {}

        monkeypatch.undo()
        obs.subscribe(SubscriptionOptions(filter=f))  # fresh first registrant works
    finally:
        obs.disconnect()
