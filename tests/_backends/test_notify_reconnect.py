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
        assert _wait_until(lambda: got == [("u", "k")], 2.0), got
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


# =============================================================================
# End to end through a storage: connection drop and server restart
# =============================================================================


def _level_reader(storage, view: dict, wakes: list):
    """A level flow bound to a subscription: re-read the whole watched state on any wake."""
    from virtuals.tkv.filter import PrefixFilter
    from virtuals.tkv.storage import StorageScanOptions

    def on_wake(key):
        wakes.append(key)
        with storage.snapshot() as sn:
            rows = dict(
                sn.scan(
                    StorageScanOptions(start=("lv",), break_filter=PrefixFilter(prefix=("lv",)))
                ).items()
            )
        view.clear()
        view.update(rows)

    return on_wake


@pytest.mark.parametrize("how", ["client_kill", "server_kill"])
def test_connection_drop_through_storage(server, tmp_path, how):
    """Writes land in a sqlite storage while the observer is cut off.

    `client_kill` drops only the observer's pubsub connection (the likelier
    production event: idle timeout, output buffer limit). redis-py reconnects
    within milliseconds, so to get a gap of known length the re-registration
    is held and the index is dropped the way a publish reaching no subscriber
    drops it; publishes in the gap then reach nobody. `server_kill` loses
    the index and the library too. Either way: once re-registered, keyed
    routing resumes, filters still hold, the level reader catches up on its
    next wake, and shutdown leaves no thread or server client.
    """
    from virtuals._backends.observers.redis_pubsub import RedisObserver
    from virtuals._backends.publishers.redis_pubsub import RedisPublisher
    from virtuals.codecs import NoOpCodec
    from virtuals.storages.sqlite import SQLiteStorage
    from virtuals.tkv.filter import PrefixFilter, WildcardFilter
    from virtuals.tkv.observer import SubscriptionOptions

    prefix = "e2e"
    probe = redis.from_url(server.url)
    pub = RedisPublisher(redis_url=server.url, channel_prefix=prefix, publish_retry_seconds=0.3)
    pub.connect()
    obs = RedisObserver(redis_url=server.url, channel_prefix=prefix)
    obs.connect()
    storage = SQLiteStorage(path=tmp_path / "db.sqlite", codec=NoOpCodec(), publisher=pub)
    storage.open()
    truth: dict = {}
    view: dict = {}
    lv_wakes: list = []
    w_got: list = []

    def write(rows: dict) -> None:
        with storage.transaction() as t:
            for k, v in rows.items():
                t.put(k, v)
        truth.update({k: v for k, v in rows.items() if k[0] == "lv"})

    try:
        obs.subscribe(SubscriptionOptions(filter=PrefixFilter(prefix=("lv",)))).bind(
            _level_reader(storage, view, lv_wakes)
        )
        obs.subscribe(SubscriptionOptions(filter=WildcardFilter(pattern=("w", "*", "x")))).bind(
            w_got.append
        )
        for i in range(100):
            write({("lv", i): b"%d" % i})
        # a barrier write: once its wake is in, every earlier keyed wake is too
        # (one ordered channel), so nothing from the setup lands in the gap
        write({("lv", "barrier"): b"1"})
        assert _wait_until(lambda: lv_wakes and lv_wakes[-1] == ("lv", "barrier"), 5.0)
        assert view == truth

        # the gap
        hold = threading.Event()
        if how == "client_kill":
            entered = threading.Event()
            real = obs._register_hash

            def held(h, entries):
                entered.set()
                hold.wait(10)
                return real(h, entries)

            obs._register_hash = held
            assert probe.client_kill_filter(_type="pubsub") >= 1
            assert entered.wait(5), "no re-registration after the reconnect"
            for key in probe.scan_iter(match=f"{prefix}:nx1:*"):
                probe.delete(key)
        else:
            _kill(server, "kill")
        for i in range(100, 300):
            write({("lv", i): b"%d" % i})
        for i in range(20):
            write({("w", i, "y"): b"y", ("w", i, "x"): b"x"})
        try:
            pub.flush(timeout=3)
        except Exception:  # noqa: S110 - server_kill: these publishes are dropped
            pass
        time.sleep(0.3)
        assert view != truth
        if how == "client_kill":
            assert probe.hlen(f"{prefix}:nx1:refs") == 0  # nothing re-registered yet
            hold.set()
        else:
            server.start()

        # both filter hashes are routed again
        assert _wait_until(lambda: probe.hlen(f"{prefix}:nx1:refs") == 2, 10.0)

        # routing is live again, filters still hold, the level reader catches up
        write({("lv", 0): b"again", ("w", 1, "x"): b"x2", ("w", 1, "y"): b"y2"})
        pub.flush(2)
        assert _wait_until(lambda: view.get(("lv", 0)) == b"again", 2.0)
        assert view == truth
        assert _wait_until(lambda: w_got and w_got[-1] == ("w", 1, "x"), 2.0), w_got
        assert lv_wakes[-1] == ("lv", 0)
        time.sleep(0.2)
        assert all(k[2] == "x" for k in w_got), w_got
    finally:
        storage.close()
        obs.disconnect()
        pub.disconnect()
    leftover = [
        t.name
        for t in threading.enumerate()
        if t.name.startswith(("RedisObserver", "Observer-", "Publisher-"))
    ]
    assert leftover == []
    clients = probe.client_list()
    assert len(clients) == 1, [c.get("name") or c.get("cmd") for c in clients]
    probe.close()


def test_client_kill_ungated_resumes_delivery(server):
    """r1 as a test: the real path with nothing held. redis-py reconnects at
    once, the re-registration lands, nothing non-matching is ever delivered,
    keyed delivery resumes."""
    from virtuals._backends.observers.redis_pubsub import RedisObserver
    from virtuals._backends.publishers.redis_pubsub import RedisPublisher
    from virtuals.tkv.filter import PrefixFilter, WildcardFilter
    from virtuals.tkv.observer import SubscriptionOptions

    probe = redis.from_url(server.url)
    obs = RedisObserver(redis_url=server.url, channel_prefix="r1")
    obs.connect()
    pub = RedisPublisher(redis_url=server.url, channel_prefix="r1", publish_retry_seconds=1.0)
    pub.connect()
    got_a: list = []
    got_w: list = []
    try:
        obs.subscribe(SubscriptionOptions(filter=PrefixFilter(prefix=("a",)))).bind(got_a.append)
        obs.subscribe(SubscriptionOptions(filter=WildcardFilter(pattern=("w", "*", "x")))).bind(
            got_w.append
        )
        pub.notify([("a", 1), ("w", 1, "x"), ("b", 1), ("w", 1, "y")])
        pub.flush(2)
        assert _wait_until(lambda: got_a == [("a", 1)] and got_w == [("w", 1, "x")])

        assert probe.client_kill_filter(_type="pubsub") >= 1
        for i in range(50):
            pub.notify([("a", "gap", i), ("b", "gap", i), ("w", i, "x"), ("w", i, "y")])
        pub.flush(2)
        deadline = time.monotonic() + 10.0
        while not (got_a[-1] == ("a", 2) and got_w[-1] == ("w", 2, "x")):
            assert time.monotonic() < deadline, (got_a, got_w)
            pub.notify([("a", 2), ("b", 2), ("w", 2, "x"), ("w", 2, "y"), ("c",)])
            pub.flush(2)
            _wait_until(lambda: got_a[-1] == ("a", 2) and got_w[-1] == ("w", 2, "x"), 0.2)
        time.sleep(0.2)
        assert all(k[0] == "a" for k in got_a), got_a
        assert all(len(k) == 3 and k[0] == "w" and k[2] == "x" for k in got_w), got_w
    finally:
        pub.disconnect()
        obs.disconnect()
        probe.close()
