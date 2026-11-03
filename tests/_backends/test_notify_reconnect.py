"""Redis notify pair survives a valkey restart that wipes everything.

The server runs without persistence, so a restart drops the routing index and
the function library. The observer's listener reconnects, redis-py
resubscribes, and each resubscribe confirmation re-runs nu1_reg (which
reloads the library). The publisher reloads it on its own next call.
"""

from __future__ import annotations

import threading
import time

import pytest


pytestmark = pytest.mark.redis

redis = pytest.importorskip("redis")
pytest.importorskip("msgpack")
pytest.importorskip("valkeylite")


@pytest.fixture
def server(tmp_path):
    from tests.support.valkey import private_valkey

    srv = private_valkey(tmp_path)
    yield srv
    srv.close()


def _deliver_until(pub, key, ev, timeout=10.0):
    """Write `key` until `ev` fires; returns the number of writes needed."""
    deadline = time.monotonic() + timeout
    n = 0
    while time.monotonic() < deadline:
        n += 1
        pub.notify(key)
        try:
            pub.flush(timeout=2.0)
        except Exception:  # noqa: S110 - server may still be coming back
            pass
        if ev.wait(0.2):
            return n
    raise AssertionError(f"{key} not delivered within {timeout}s")


def _kill(server, how):
    if how == "term":
        server.stop()
        return
    proc = server._server._process  # SIGKILL: no shutdown, no goodbye to clients
    proc.kill()
    proc.wait(5)
    server._server._process = None


@pytest.mark.parametrize(("how", "restarts"), [("term", 1), ("kill", 1), ("kill", 2)])
def test_subscriptions_keep_delivering_after_restart(server, how, restarts):
    from virtuals._backends.observers.redis_pubsub import RedisObserver
    from virtuals._backends.publishers.redis_pubsub import RedisPublisher
    from virtuals.tkv.filter import Or, PrefixFilter, SuffixFilter, WildcardFilter
    from virtuals.tkv.observer import SubscriptionOptions

    filters = [
        PrefixFilter(prefix=("a",)),
        WildcardFilter(pattern=("w", "*", 3)),
        Or(filters=(SuffixFilter(suffix=("z",)), PrefixFilter(prefix=("o",)))),
    ]
    keys = [("a", 1), ("w", "x", 3), ("q", "z")]

    obs = RedisObserver(redis_url=server.url, channel_prefix="r")
    obs.connect()
    pub = RedisPublisher(redis_url=server.url, channel_prefix="r", publish_retry_seconds=1.0)
    pub.connect()
    probe = redis.from_url(server.url)
    try:
        evs: list[threading.Event] = []
        for f in filters:
            ev = threading.Event()
            # real keys only: a keyless resync wake after the restart must not
            # stand in for the delivery this test checks
            obs.subscribe(SubscriptionOptions(filter=f)).bind(
                lambda k, ev=ev: k is not None and ev.set()
            )
            evs.append(ev)
        for k, ev in zip(keys, evs, strict=True):
            assert _deliver_until(pub, k, ev) == 1

        for _ in range(restarts):
            _kill(server, how)
            time.sleep(0.2)
            server.start()
            for ev in evs:
                ev.clear()
            for k, ev in zip(keys, evs, strict=True):
                _deliver_until(pub, k, ev)
            # every local hash is back in the index, library reloaded
            assert probe.hlen("r:nx1:refs") == len(filters)
            assert probe.function_list(library="nu_notify_v1")

        # once re-registered, a fresh subscribe is still 0ms live
        hit = threading.Event()
        obs.subscribe(SubscriptionOptions(filter=PrefixFilter(prefix=("new",)))).bind(
            lambda k: hit.set()
        )
        pub.notify(("new", 1))
        assert hit.wait(2.0)
    finally:
        probe.close()
        pub.disconnect()
        obs.disconnect()


def test_reregistration_follows_resubscribe_confirmation(server, monkeypatch):
    """nu1_reg after reconnect runs only once the server confirmed the resubscribe."""
    from virtuals._backends.observers.redis_pubsub import RedisObserver
    from virtuals.tkv.filter import PrefixFilter, filter_hash
    from virtuals.tkv.observer import SubscriptionOptions

    obs = RedisObserver(redis_url=server.url, channel_prefix="r")
    obs.connect()
    probe = redis.from_url(server.url)
    f = PrefixFilter(prefix=("a",))
    h = filter_hash(f)
    seen: list[int] = []
    real = obs._register_hash

    def spy(hh, entries):
        # the channel is subscribed on the server when registration runs
        n = probe.pubsub_numsub(f"r:notif:{hh}")[0][1]
        out = real(hh, entries)
        seen.append(n)  # after the call, so the test reads a finished nu1_reg
        return out

    try:
        obs.subscribe(SubscriptionOptions(filter=f))
        monkeypatch.setattr(obs, "_register_hash", spy)
        _kill(server, "kill")
        server.start()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not seen:
            time.sleep(0.05)
        assert seen and all(n == 1 for n in seen), seen
        assert probe.hexists("r:nx1:refs", h)
    finally:
        probe.close()
        obs.disconnect()


def _wait_until(pred, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    return pred()


def test_same_filter_subscribe_waits_for_reregistration(server, caplog):
    """Readiness resets on a drop: a same-filter subscribe is live on return.

    While the server is down it raises instead of reporting ready. After the
    restart it returns only once the hash is re-registered, so a write right
    after it is delivered.
    """
    import logging

    from virtuals._backends.observers.redis_pubsub import RedisObserver
    from virtuals._backends.publishers.redis_pubsub import RedisPublisher
    from virtuals.tkv.filter import PrefixFilter, filter_hash
    from virtuals.tkv.observer import ObserverSubscriptionError, SubscriptionOptions

    f = PrefixFilter(prefix=("u",))
    h = filter_hash(f)
    obs = RedisObserver(redis_url=server.url, channel_prefix="r", subscribe_timeout=0.5)
    obs.connect()
    pub = None
    caplog.set_level(logging.WARNING, logger="virtuals._backends.observers.redis_pubsub")
    try:
        obs.subscribe(SubscriptionOptions(filter=f))
        assert obs._ready[h].event.is_set()

        _kill(server, "kill")
        assert _wait_until(lambda: not obs._ready[h].event.is_set())
        # The gap is named, not silent.
        assert any("not routed until re-registered" in r.message for r in caplog.records)
        t0 = time.monotonic()
        with pytest.raises(ObserverSubscriptionError):
            obs.subscribe(SubscriptionOptions(filter=f))
        assert time.monotonic() - t0 >= 0.4
        assert obs._local_hash_refcount == {h: 1}

        server.start()
        pub = RedisPublisher(redis_url=server.url, channel_prefix="r")
        pub.connect()
        got: list = []
        t0 = time.monotonic()
        while True:  # the restart may not be noticed yet: retry the subscribe
            try:
                s = obs.subscribe(SubscriptionOptions(filter=f))
                break
            except ObserverSubscriptionError:
                assert time.monotonic() - t0 < 10
        s.bind(got.append)
        pub.notify(("u", "k"))
        pub.flush(2)
        # a resync wake (None) for the re-registered hash may reach s too
        assert _wait_until(lambda: [k for k in got if k is not None] == [("u", "k")], 2.0), got
    finally:
        if pub is not None:
            pub.disconnect()
        obs.disconnect()


def test_reregistration_raced_by_reconnect_waits(server, monkeypatch):
    """A first registrant whose nu1_reg raced a drop waits for the re-registration."""
    from virtuals._backends.observers.redis_pubsub import RedisObserver
    from virtuals.tkv.filter import PrefixFilter, filter_hash
    from virtuals.tkv.observer import SubscriptionOptions

    obs = RedisObserver(redis_url=server.url, channel_prefix="r", subscribe_timeout=5.0)
    obs.connect()
    f = PrefixFilter(prefix=("race",))
    h = filter_hash(f)
    real = obs._register_hash
    first = [True]

    def racing(hh, entries):
        out = real(hh, entries)
        if first[0]:
            first[0] = False
            obs._invalidate_ready("lost")  # a drop right after nu1_reg
        return out

    monkeypatch.setattr(obs, "_register_hash", racing)
    try:
        done = threading.Event()

        def run():
            obs.subscribe(SubscriptionOptions(filter=f))
            done.set()

        t = threading.Thread(target=run, daemon=True)
        t.start()
        assert not done.wait(0.3), "returned ready on a stale registration"
        # The resubscribe confirmation after the reconnect queues a re-registration.
        obs._rereg_q.put((h, obs._conn_gen))
        assert done.wait(3)
        assert obs._ready[h].event.is_set()
    finally:
        obs.disconnect()


def test_level_subscriber_converges_after_restart_mid_stream(server):
    """Writes made while the observer is cut off are recovered by the resync wake.

    A writer streams values for a few keys and keeps writing while valkey is
    down (those PUBLISHes go nowhere). The last write lands while it is down.
    A level subscriber re-reads the whole state on every wake, whatever the
    key. After the restart nobody writes again, so only the keyless resync
    wake can bring it to the final state.
    """
    from virtuals._backends.observers.redis_pubsub import RedisObserver
    from virtuals._backends.publishers.redis_pubsub import RedisPublisher
    from virtuals.tkv.filter import PrefixFilter
    from virtuals.tkv.observer import SubscriptionOptions

    truth: dict = {}
    truth_lock = threading.Lock()
    view: dict = {}
    wakes: list = []
    asked = threading.Event()
    stop = threading.Event()

    def reader():  # the level flow: one pass per wake, re-reads everything
        while not stop.is_set():
            if not asked.wait(0.05):
                continue
            asked.clear()
            with truth_lock:
                snap = dict(truth)
            view.clear()
            view.update(snap)

    def on_wake(key):
        wakes.append(key)
        asked.set()

    obs = RedisObserver(redis_url=server.url, channel_prefix="r")
    obs.connect()
    pub = RedisPublisher(redis_url=server.url, channel_prefix="r", publish_retry_seconds=0.2)
    pub.connect()
    t = threading.Thread(target=reader, daemon=True)
    t.start()

    def write(i):
        k = ("lv", i % 5)
        with truth_lock:
            truth[k] = i
        pub.notify(k)

    try:
        obs.subscribe(SubscriptionOptions(filter=PrefixFilter(prefix=("lv",)))).bind(on_wake)
        for i in range(200):
            write(i)
        pub.flush(timeout=5)
        assert _wait_until(lambda: view == truth, 5.0)

        _kill(server, "kill")
        for i in range(200, 400):  # written while the observer is cut off
            write(i)
            time.sleep(0.002)
        try:
            pub.flush(timeout=3)
        except Exception:  # noqa: S110 - publishes during the outage are dropped
            pass
        time.sleep(0.5)  # outlast the publisher's retry window
        assert view != truth
        n_keyed = sum(1 for k in wakes if k is not None)

        server.start()
        assert _wait_until(lambda: view == truth, 10.0), (view, truth)
        assert None in wakes
        # nothing keyed arrived after the restart: the resync did it
        assert sum(1 for k in wakes if k is not None) == n_keyed
        assert truth[("lv", 399 % 5)] == 399
    finally:
        stop.set()
        t.join(2)
        pub.disconnect()
        obs.disconnect()


def test_event_consumer_gets_keyless_wake_without_crashing(server):
    """An event-style consumer sees exactly one None per subscription and keeps going.

    A receiver that assumes a real key (and raises on None) is logged, not
    fatal: other receivers still get the wake and later keys still arrive.
    """
    from virtuals._backends.observers.redis_pubsub import RedisObserver
    from virtuals._backends.publishers.redis_pubsub import RedisPublisher
    from virtuals.tkv.filter import PrefixFilter
    from virtuals.tkv.observer import SubscriptionOptions

    obs = RedisObserver(redis_url=server.url, channel_prefix="r")
    obs.connect()
    pub = None
    events: list = []
    other: list = []
    strict_seen: list = []

    def strict(key):  # an event body that indexes the key
        strict_seen.append(key[0])

    try:
        s1 = obs.subscribe(SubscriptionOptions(filter=PrefixFilter(prefix=("ev",))))
        s1.bind(strict)
        s1.bind(events.append)
        s2 = obs.subscribe(SubscriptionOptions(filter=PrefixFilter(prefix=("ot",))))
        s2.bind(other.append)

        _kill(server, "kill")
        server.start()
        assert _wait_until(lambda: events == [None] and other == [None], 10.0), (events, other)

        pub = RedisPublisher(redis_url=server.url, channel_prefix="r")
        pub.connect()
        pub.notify([("ev", 1), ("ot", 2)])
        pub.flush(2)
        assert _wait_until(lambda: events == [None, ("ev", 1)] and other == [None, ("ot", 2)])
        assert strict_seen == ["ev"]
        time.sleep(0.2)
        assert events.count(None) == 1 and other.count(None) == 1
    finally:
        if pub is not None:
            pub.disconnect()
        obs.disconnect()
