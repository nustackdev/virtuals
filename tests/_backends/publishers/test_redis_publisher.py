"""RedisPublisher over a real server: server-side routing via nu1_route.

Interest is registered straight through `nu1_reg` (the observer migration
is a later step), so these tests cover the publisher on its own.
"""

from __future__ import annotations

import threading
import time

import pytest

from virtuals._backends.notify.keyenc import hint_entries
from virtuals._backends.notify.router import NotifyLua
from virtuals._backends.publishers._base import PublisherBase
from virtuals.tkv.filter import Filter, PassAll, PrefixFilter, canonicalize, filter_hash
from virtuals.tkv.publisher import PublisherOperationError


redis = pytest.importorskip("redis")
msgpack = pytest.importorskip("msgpack")


def _make_publisher(redis_url, channel_prefix, **kwargs):
    from virtuals._backends.publishers.redis_pubsub import RedisPublisher

    pub = RedisPublisher(redis_url=redis_url, channel_prefix=channel_prefix, **kwargs)
    pub.connect()
    return pub


class Sub:
    """Registers filters via nu1_reg, subscribes their channels, reads payloads."""

    def __init__(self, url: str, prefix: str) -> None:
        self.url, self.P = url, prefix
        self.conn = redis.from_url(url)
        self.lua = NotifyLua()
        self.ps = self.conn.pubsub()

    def reg(self, f: Filter) -> str:
        h = filter_hash(f)
        ch = f"{self.P}:notif:{h}"
        self.ps.subscribe(ch)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            m = self.ps.get_message(timeout=0.1)
            if m and m["type"] == "subscribe" and m["channel"].decode() == ch:
                break
        entries = hint_entries(canonicalize(f).routing_hints())
        self.lua.call(self.conn, "nu1_reg", [self.P, h] + [x for e in entries for x in e])
        return h

    def read(self, timeout: float = 0.5) -> list[tuple[str, dict]]:
        """All messages until `timeout` passes with nothing new."""
        out = []
        pre = f"{self.P}:notif:"
        while True:
            m = self.ps.get_message(timeout=timeout)
            if m is None:
                return out
            if m["type"] == "message":
                out.append((m["channel"].decode()[len(pre) :], msgpack.unpackb(m["data"])))

    def close(self) -> None:
        self.ps.close()
        self.conn.close()


@pytest.fixture
def prefix(unique_channel_prefix, redis_cleanup):
    redis_cleanup(unique_channel_prefix)
    return unique_channel_prefix


@pytest.fixture
def sub(redis_url, prefix):
    s = Sub(redis_url, prefix)
    yield s
    s.close()


# =============================================================================
# Routing
# =============================================================================


@pytest.mark.redis
def test_publish_writes_msgpack_payload_on_notif_channel(redis_url, prefix, sub):
    h = sub.reg(PrefixFilter(("users",)))
    pub = _make_publisher(redis_url, prefix)
    try:
        pub.notify(("users", "alice"))
        pub.flush()
        msgs = sub.read()
    finally:
        pub.disconnect()
    assert msgs == [(h, {"iid": pub._instance_id, "keys": [["users", "alice"]]})]


@pytest.mark.redis
def test_subscription_registered_before_write_is_routed(redis_url, prefix, sub):
    """No cache: interest registered after connect is seen by the very next write."""
    pub = _make_publisher(redis_url, prefix)
    try:
        h = sub.reg(PrefixFilter(("late",)))
        pub.notify(("late", 1))
        pub.flush()
        assert sub.read() == [(h, {"iid": pub._instance_id, "keys": [["late", 1]]})]
    finally:
        pub.disconnect()


@pytest.mark.redis
def test_no_publish_when_no_cluster_interest(redis_url, prefix):
    probe = redis.from_url(redis_url)
    ps = probe.pubsub()
    ps.psubscribe(f"{prefix}:notif:*")
    ps.get_message(timeout=0.5)
    pub = _make_publisher(redis_url, prefix)
    try:
        pub.notify([("data", i) for i in range(50)])
        pub.flush()
        assert ps.get_message(timeout=0.3) is None
    finally:
        pub.disconnect()
        ps.close()
        probe.close()


@pytest.mark.redis
def test_filter_hash_bucketing_isolates_publishes(redis_url, prefix, sub):
    h_users = sub.reg(PrefixFilter(("users",)))
    h_posts = sub.reg(PrefixFilter(("posts",)))
    pub = _make_publisher(redis_url, prefix)
    try:
        pub.notify([("users", "alice"), ("posts", "42"), ("comments", "x")])
        pub.flush()
        got: dict[str, list] = {}
        for h, p in sub.read():
            got.setdefault(h, []).extend(p["keys"])
    finally:
        pub.disconnect()
    assert got == {h_users: [["users", "alice"]], h_posts: [["posts", "42"]]}


@pytest.mark.redis
def test_reloads_library_after_function_flush(redis_url, prefix, sub):
    h = sub.reg(PassAll())
    pub = _make_publisher(redis_url, prefix)
    try:
        redis.from_url(redis_url).function_flush()
        pub.notify(("a",))
        pub.flush()
        assert [x[0] for x in sub.read()] == [h]
    finally:
        pub.disconnect()


@pytest.mark.redis
def test_evalsha_fallback_on_redis6():
    from tests.support.valkey import private_redis6

    server = private_redis6()
    if server is None:
        pytest.skip("system redis-server not found")
    s = Sub(server.url, "vtest:r6")
    pub = None
    try:
        h = s.reg(PrefixFilter((2**60,)))
        pub = _make_publisher(server.url, "vtest:r6")
        assert pub._lua.mode == "script"
        pub.notify([(2**60, "x"), (2**60 + 1, "x")])
        pub.flush()
        assert s.read() == [(h, {"iid": pub._instance_id, "keys": [[2**60, "x"]]})]
    finally:
        if pub is not None:
            pub.disconnect()
        s.close()
        server.close()


# =============================================================================
# Chunking
# =============================================================================


class _Recorder(PublisherBase):
    MAX_KEYS_PER_CALL = 1000

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.calls: list[int] = []

    def _publish_batch(self, batch) -> None:
        self.calls.append(len(batch))


def test_base_splits_batches_into_chunks():
    p = _Recorder()
    p._safe_publish([(i,) for i in range(2500)])
    assert p.calls == [1000, 1000, 500]


@pytest.mark.redis
def test_chunks_at_most_max_keys_per_call(redis_url, prefix, sub):
    from virtuals._backends.publishers.redis_pubsub import MAX_KEYS_PER_CALL

    assert MAX_KEYS_PER_CALL == 250
    h = sub.reg(PassAll())
    pub = _make_publisher(redis_url, prefix)
    sizes: list[int] = []
    orig = pub._lua.call

    def spy(conn, name, args):
        sizes.append(sum(1 for a in args[2:] if a[:1] in (b"\x90", b"\x91", b"\x92")))
        return orig(conn, name, args)

    pub._lua.call = spy
    try:
        pub.notify([(i,) for i in range(2500)])
        pub.flush(timeout=5)
        msgs = sub.read()
    finally:
        pub.disconnect()
    assert sum(sizes) == 2500 and max(sizes) <= MAX_KEYS_PER_CALL and len(sizes) >= 10
    assert all(m[0] == h and len(m[1]["keys"]) <= MAX_KEYS_PER_CALL for m in msgs)
    assert [k for _, p in msgs for k in p["keys"]] == [[i] for i in range(2500)]


# =============================================================================
# No silent drops
# =============================================================================


@pytest.mark.redis
def test_unroutable_key_surfaces_on_flush(redis_url, prefix, sub):
    h = sub.reg(PassAll())
    pub = _make_publisher(redis_url, prefix)
    try:
        pub.notify([("ok", 1), ("bad", 1.5), ("big", 2**64 + 1), ("ok", 2)])
        with pytest.raises(PublisherOperationError):
            pub.flush()
        assert pub.dropped_keys == 2
        pub.flush()  # raised once, then cleared
        assert sub.read() == [(h, {"iid": pub._instance_id, "keys": [["ok", 1], ["ok", 2]]})]
    finally:
        pub.disconnect()


@pytest.fixture
def own_valkey(tmp_path):
    from tests.support.valkey import private_valkey

    try:
        server = private_valkey(tmp_path)
    except ImportError:
        pytest.skip("valkeylite not installed")
    yield server
    server.close()


@pytest.mark.redis
def test_backpressure_then_surfaced_drop_when_server_down(own_valkey):
    pub = _make_publisher(own_valkey.url, "vtest:bp", queue_maxsize=10, publish_retry_seconds=0.3)
    try:
        own_valkey.stop()
        done = threading.Event()

        def writer():
            pub.notify([("k", i) for i in range(40)])
            done.set()

        t = threading.Thread(target=writer, daemon=True)
        t.start()
        assert not done.wait(0.2), "notify should block while the worker retries"
        assert done.wait(20), "notify never unblocked"
        deadline = time.monotonic() + 10
        while pub.dropped_keys < 40 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert pub.dropped_keys == 40
        with pytest.raises(PublisherOperationError):
            pub.flush(timeout=1)
    finally:
        pub.disconnect()


@pytest.mark.redis
def test_retry_delivers_when_server_comes_back(own_valkey):
    pub = _make_publisher(own_valkey.url, "vtest:rt", publish_retry_seconds=10)
    s = None
    attempts: list[int] = []
    up = threading.Event()  # retries hold until the subscriber is back too
    orig = pub._lua.call

    def gated(conn, name, args):
        attempts.append(1)
        if len(attempts) > 1:
            up.wait(5)
        return orig(conn, name, args)

    pub._lua.call = gated
    try:
        own_valkey.stop()
        pub.notify(("k", 1))
        time.sleep(0.3)  # first attempt failed, worker is retrying
        own_valkey.start()  # fresh server: no library, no index
        s = Sub(own_valkey.url, "vtest:rt")
        h = s.reg(PassAll())
        up.set()
        pub.flush(timeout=10)
        assert len(attempts) >= 2
        assert pub.dropped_keys == 0
        assert s.read(timeout=1) == [(h, {"iid": pub._instance_id, "keys": [["k", 1]]})]
    finally:
        pub.disconnect()
        if s is not None:
            s.close()


# =============================================================================
# Drain on disconnect, drop counting
# =============================================================================


class _Slow(PublisherBase):
    def __init__(self, delay: float = 0.2, fail: bool = False, **kw) -> None:
        super().__init__(**kw)
        self.delay = delay
        self.fail = fail
        self.sent: list = []

    def _publish_batch(self, batch) -> None:
        time.sleep(self.delay)
        if self.fail:
            raise ConnectionError("down")
        self.sent.extend(batch)


def test_disconnect_publishes_keys_queued_behind_inflight_batch():
    """Keys queued while a batch is in flight are published, not lost, at stop."""
    p = _Slow()
    p.connect()
    p.notify(("a",))
    time.sleep(0.05)  # worker is inside the slow publish of ("a",)
    p.notify([("b",), ("c",)])
    p.disconnect()
    assert sorted(p.sent) == [("a",), ("b",), ("c",)]
    assert p.dropped_keys == 0
    p.flush()  # nothing surfaced


def test_disconnect_records_keys_it_cannot_publish():
    """A failing transport at stop: one attempt per chunk, then counted and surfaced."""
    p = _Slow(delay=0.05, fail=True, publish_retry_seconds=30)
    p.connect()
    p.notify(("a",))
    time.sleep(0.02)
    p.notify([("b",), ("c",)])
    t0 = time.monotonic()
    p.disconnect()
    assert time.monotonic() - t0 < 5  # the retry window is cut short at stop
    assert p.dropped_keys == 3
    with pytest.raises(PublisherOperationError):
        p.flush()


@pytest.mark.redis
def test_unroutable_key_counted_once_across_retries(redis_url, prefix, sub):
    """Encoding runs once per chunk: retries do not re-record unroutable keys."""
    h = sub.reg(PassAll())
    pub = _make_publisher(redis_url, prefix)
    real = pub._lua.call
    calls = [0]

    def flaky(*a):
        calls[0] += 1
        if calls[0] <= 3:
            raise redis.ConnectionError("blip")
        return real(*a)

    pub._lua.call = flaky
    try:
        pub.notify([("ok",), ("bad", 1.5)])
        with pytest.raises(PublisherOperationError):
            pub.flush(timeout=3)
        assert calls[0] == 4
        assert pub.dropped_keys == 1
        assert sub.read() == [(h, {"iid": pub._instance_id, "keys": [["ok"]]})]
    finally:
        pub.disconnect()
