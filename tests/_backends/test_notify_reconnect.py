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
            obs.subscribe(SubscriptionOptions(filter=f)).bind(lambda k, ev=ev: ev.set())
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
