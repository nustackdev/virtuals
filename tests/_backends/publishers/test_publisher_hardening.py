"""Publisher under stalls, drops, failed connects and worker lifecycle.

Each test pins one finding of the notify or lifecycle audit: a stalled
transport never holds a writer in notify() for long (N2), drops are
counted and surfaced, one worker ever (F4), a failed connect
leaks nothing (N7), a dropped publisher is collected (F8), a dead server
bounds the drain at disconnect.
"""

from __future__ import annotations

import gc
import logging
import os
import signal
import threading
import time
import weakref

import pytest

from virtuals._backends.publishers import _base as base_mod
from virtuals._backends.publishers._base import PublisherBase
from virtuals.tkv.filter import PrefixFilter
from virtuals.tkv.observer import SubscriptionOptions
from virtuals.tkv.publisher import PublisherConnectionError, PublisherOperationError


def _wait_until(pred, timeout=5.0, interval=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return pred()


class _Gated(PublisherBase):
    """A transport that blocks each send on `gate`, or fails when `fail`."""

    def __init__(self, fail: bool = False, delay: float = 0.0, **kw) -> None:
        super().__init__(**kw)
        self.gate = threading.Event()
        self.fail = fail
        self.delay = delay
        self.sent: list = []

    def _publish_batch(self, batch) -> None:
        if self.delay:
            time.sleep(self.delay)
        else:
            self.gate.wait(10)
        if self.fail:
            raise ConnectionError("down")
        self.sent.extend(batch)


# =============================================================================
# N2: notify() is bounded
# =============================================================================


def test_notify_on_a_stalled_transport_returns_within_its_timeout_and_surfaces():
    p = _Gated(queue_maxsize=2, notify_timeout=0.3)
    p.connect()
    try:
        t0 = time.monotonic()
        p.notify([("k", i) for i in range(10)])
        elapsed = time.monotonic() - t0
        assert 0.25 < elapsed < 0.8, elapsed
        # 1 or 2 in flight (the worker batches what it finds), 2 queued, the
        # rest dropped and surfaced
        assert p.dropped_keys in (6, 7)
        p.gate.set()
        with pytest.raises(PublisherOperationError, match="dropped"):
            p.flush(timeout=2)
        assert sorted(p.sent) == [("k", i) for i in range(10 - p.dropped_keys)]
    finally:
        p.gate.set()
        p.disconnect()


def test_notify_timeout_is_per_call_not_per_key():
    p = _Gated(queue_maxsize=1, notify_timeout=0.2)
    p.connect()
    try:
        t0 = time.monotonic()
        p.notify([("k", i) for i in range(50)])
        assert time.monotonic() - t0 < 0.6
    finally:
        p.gate.set()
        p.disconnect()


# =============================================================================
# F4: exactly one worker
# =============================================================================


def test_worker_outliving_disconnect_is_never_joined_by_a_second(monkeypatch, caplog):
    monkeypatch.setattr(base_mod, "_STOP_JOIN_SECONDS", 0.3)
    p = _Gated()
    p.connect()
    old = p._thread
    try:
        p.notify(("a",))
        time.sleep(0.05)  # worker is inside the gated send
        caplog.set_level(logging.WARNING, logger="virtuals._backends.publishers._base")
        p.disconnect()
        assert old.is_alive()
        assert any(old.name in r.message for r in caplog.records)
        with pytest.raises(PublisherConnectionError, match="still sending"):
            p.connect()
        p.gate.set()
        old.join(2)
        assert not old.is_alive()
        p.connect()
        assert p._thread is not old
        assert len([t for t in threading.enumerate() if t.name.startswith("Publisher-")]) == 1
        p.notify(("b",))
        p.flush(2)
        assert p.sent == [("a",), ("b",)]
    finally:
        p.gate.set()
        p.disconnect()


# =============================================================================
# disconnect with a dead transport is bounded by one attempt
# =============================================================================


def test_disconnect_drain_stops_at_the_first_failed_chunk():
    class OneKeyChunks(_Gated):
        MAX_KEYS_PER_CALL = 1

    p = OneKeyChunks(fail=True, delay=0.2, publish_retry_seconds=30)
    p.connect()
    p.notify([("k", i) for i in range(20)])
    time.sleep(0.05)
    t0 = time.monotonic()
    p.disconnect()
    assert time.monotonic() - t0 < 1.0  # one attempt, then the rest are dropped unsent
    assert p.dropped_keys == 20
    with pytest.raises(PublisherOperationError):
        p.flush()


# =============================================================================
# N7 and F8 (transport-free)
# =============================================================================


def test_failed_connect_runs_the_disconnect_hook():
    class Flaky(PublisherBase):
        def __init__(self):
            super().__init__()
            self.closed = 0

        def _on_connect(self):
            raise RuntimeError("boom")

        def _on_disconnect(self):
            self.closed += 1

    p = Flaky()
    with pytest.raises(PublisherConnectionError, match="boom"):
        p.connect()
    assert p.closed == 1
    assert p._thread is None
    p.disconnect()
    assert p.closed == 1


def test_dropped_publisher_is_collected_and_its_worker_exits():
    p = _Gated()
    p.gate.set()
    p.connect()
    p.notify(("a",))
    p.flush()
    worker = p._thread
    ref = weakref.ref(p)
    del p
    gc.collect()
    assert ref() is None
    worker.join(3)
    assert not worker.is_alive()


# =============================================================================
# Redis: stalled server, failed connect, drop
# =============================================================================


@pytest.fixture
def own_valkey(tmp_path):
    pytest.importorskip("valkeylite")
    from tests.support.valkey import private_valkey

    srv = private_valkey(tmp_path)
    yield srv
    srv.close()


@pytest.mark.redis
def test_stalled_server_bounds_notify_and_delivery_resumes_after(own_valkey):
    """SIGSTOP valkey: writers return within notify_timeout, drops are counted,
    and once the server is back keyed delivery goes on."""
    pytest.importorskip("redis")
    from virtuals._backends.observers.redis_pubsub import RedisObserver
    from virtuals._backends.publishers.redis_pubsub import RedisPublisher

    pid = own_valkey._server._process.pid
    pub = RedisPublisher(
        redis_url=own_valkey.url,
        channel_prefix="h",
        queue_maxsize=4,
        publish_retry_seconds=0.5,
        notify_timeout=0.2,
        socket_timeout=0.3,
    )
    pub.connect()
    obs = RedisObserver(redis_url=own_valkey.url, channel_prefix="h")
    obs.connect()
    got_a: list = []
    got_b: list = []
    try:
        obs.subscribe(SubscriptionOptions(filter=PrefixFilter(prefix=("a",)))).bind(got_a.append)
        obs.subscribe(SubscriptionOptions(filter=PrefixFilter(prefix=("b",)))).bind(got_b.append)
        pub.notify(("a", 0))
        pub.flush(2)
        assert _wait_until(lambda: got_a == [("a", 0)])

        os.kill(pid, signal.SIGSTOP)
        try:
            pub.notify(("a", 1))  # worker takes it and stalls in the send
            time.sleep(0.1)
            t0 = time.monotonic()
            for i in range(2, 12):
                pub.notify(("a", i))
            assert time.monotonic() - t0 < 1.0, "writers held by a stalled server"
            assert pub.dropped_keys > 0
        finally:
            os.kill(pid, signal.SIGCONT)
        with pytest.raises(PublisherOperationError):
            pub.flush(timeout=5)
        # keyed delivery goes on
        pub.notify(("b", 1))
        pub.flush(2)
        assert _wait_until(lambda: got_b == [("b", 1)]), got_b
    finally:
        try:
            os.kill(pid, signal.SIGCONT)
        except OSError:
            pass
        pub.disconnect()
        obs.disconnect()


@pytest.mark.redis
def test_failed_connect_and_drop_leave_no_client(own_valkey):
    redis = pytest.importorskip("redis")
    from virtuals._backends.publishers.redis_pubsub import RedisPublisher

    probe = redis.from_url(own_valkey.url)
    try:
        before = len(probe.client_list())
        pub = RedisPublisher(redis_url=own_valkey.url, channel_prefix="c")

        def boom(conn):
            raise RuntimeError("boom")

        pub._lua.ensure = boom
        with pytest.raises(PublisherConnectionError, match="boom"):
            pub.connect()
        assert pub._redis_pub is None
        assert _wait_until(lambda: len(probe.client_list()) == before, 2)

        pub = RedisPublisher(redis_url=own_valkey.url, channel_prefix="c")
        pub.connect()
        pub.notify(("c", 1))
        pub.flush(2)
        assert len(probe.client_list()) == before + 1
        worker = pub._thread
        ref = weakref.ref(pub)
        del pub
        gc.collect()
        assert ref() is None
        worker.join(3)
        assert not worker.is_alive()
        assert _wait_until(lambda: len(probe.client_list()) == before, 2)
    finally:
        probe.close()


# =============================================================================
# review m2: a stall costs the first commit the timeout, later ones nothing
# =============================================================================


def test_stalled_queue_drops_later_calls_at_once_until_the_worker_sends():
    p = _Gated(queue_maxsize=1, notify_timeout=0.3)
    p.connect()
    try:
        p.notify(("a",))  # in flight, gated
        time.sleep(0.05)
        p.notify(("b",))  # queued
        t0 = time.monotonic()
        p.notify(("c",))  # first drop: waits the timeout
        first = time.monotonic() - t0
        assert 0.25 < first < 0.6, first
        t0 = time.monotonic()
        for i in range(10):
            p.notify(("d", i))
        assert time.monotonic() - t0 < 0.05, "later writers waited again"
        assert p.dropped_keys == 11
        p.gate.set()  # the worker sends again: writers wait for room again
        assert _wait_until(lambda: sorted(p.sent) == [("a",), ("b",)], 2)
        p.gate.clear()
        p.notify(("e",))
        time.sleep(0.05)
        p.notify(("f",))
        t0 = time.monotonic()
        p.notify(("g",))
        assert 0.25 < time.monotonic() - t0 < 0.6
    finally:
        p.gate.set()
        p.disconnect()


# =============================================================================
# r2 N2: the stalled fast path is per connection
# =============================================================================


def test_stalled_fast_path_does_not_survive_a_reconnect():
    """A publisher that dropped while stalled and is connected again starts with
    a fresh queue: its first full-queue notify() waits the timeout, not zero."""
    p = _Gated(fail=True, queue_maxsize=1, notify_timeout=0.3)
    p.gate.set()
    p.connect()
    try:
        p.notify(("a",))  # in flight, failing and retrying
        time.sleep(0.05)
        p.notify(("b",))  # queued
        p.notify(("c",))  # dropped after the timeout: stalled from here
        assert p._stalled
        p.disconnect()  # the drain fails too: no send ever cleared the flag
        p.connect()
        assert not p._stalled
        p.notify(("d",))
        time.sleep(0.05)
        p.notify(("e",))
        t0 = time.monotonic()
        p.notify(("f",))
        assert 0.25 < time.monotonic() - t0 < 0.6, "dropped at once on a fresh connection"
    finally:
        p.disconnect()


# =============================================================================
# r3 F1: a notify while disconnected is a counted drop, never a silent one
# =============================================================================


def test_notify_while_disconnected_counts_the_drop():
    """notify() between disconnect() and connect() still raises, and the keys
    it could not queue are a drop: counted and raised once by flush()."""
    p = _Gated()
    p.gate.set()
    p.connect()
    p.disconnect()
    with pytest.raises(PublisherConnectionError, match="not connected"):
        p.notify(("a",))
    with pytest.raises(PublisherConnectionError, match="not connected"):
        p.notify([("b",), ("c",)])
    assert p.dropped_keys == 3
    p.connect()
    try:
        with pytest.raises(PublisherOperationError, match="not connected"):
            p.flush(2)
        p.flush(2)  # raised once, then clean
        assert p.sent == []
    finally:
        p.disconnect()


@pytest.mark.redis
def test_commit_while_publisher_disconnected_is_a_counted_drop(own_valkey, tmp_path):
    """The storage keeps committing while its publisher is disconnected (the
    failed notify is logged, the commit stands). The key is a counted drop
    and never reaches the reader."""
    pytest.importorskip("redis")
    from virtuals._backends.observers.redis_pubsub import RedisObserver
    from virtuals._backends.publishers.redis_pubsub import RedisPublisher
    from virtuals._backends.storages.sqlite import SQLiteStorage
    from virtuals.codecs import NoOpCodec

    pub = RedisPublisher(redis_url=own_valkey.url, channel_prefix="d")
    pub.connect()
    storage = SQLiteStorage(tmp_path / "s.sqlite", codec=NoOpCodec(), publisher=pub)
    storage.open()
    obs = RedisObserver(redis_url=own_valkey.url, channel_prefix="d")
    obs.connect()
    got: list = []
    try:
        obs.subscribe(SubscriptionOptions(filter=PrefixFilter(prefix=("k",)))).bind(got.append)
        with storage.begin_transaction() as tx:
            tx.put(("k", 1), b"a")
        pub.flush(2)
        assert _wait_until(lambda: got == [("k", 1)], 5)

        pub.disconnect()
        with storage.begin_transaction() as tx:
            tx.put(("k", 2), b"b")  # committed; the key cannot be sent
        assert pub.dropped_keys == 1
        pub.connect()
        with pytest.raises(PublisherOperationError, match="not connected"):
            pub.flush(2)
        pub.flush(2)
        assert got == [("k", 1)]
        with storage.snapshot() as snap:
            assert snap.get(("k", 2)) == b"b"
    finally:
        obs.disconnect()
        storage.close()
        pub.disconnect()
