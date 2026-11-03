"""Observer worker lifecycle: one worker ever, callback faults, leaks.

Transport-free: these use ObserverBase directly or the in-memory pair.
"""

from __future__ import annotations

import gc
import logging
import threading
import time
import weakref

import pytest

from virtuals._backends.observers._base import ObserverBase
from virtuals._backends.observers.mem import InMemoryObserver
from virtuals._backends.publishers.mem import InMemoryPublisher
from virtuals.tkv.filter import PrefixFilter
from virtuals.tkv.observer import ObserverConnectionError, SubscriptionOptions
from virtuals.tkv.transport import InMemoryTransport


def _wait_until(pred, timeout=3.0, interval=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return pred()


def _pair():
    tr = InMemoryTransport()
    pub = InMemoryPublisher(tr)
    obs = InMemoryObserver(tr)
    pub.connect()
    obs.connect()
    return tr, pub, obs


# =============================================================================
# F4: exactly one dispatch worker
# =============================================================================


def test_worker_outliving_disconnect_never_delivers_from_the_next_queue(caplog):
    """A callback blocking past disconnect's join: the old worker exits on its own,
    connect() refuses until it has, and the new worker is the only deliverer."""
    _, pub, obs = _pair()
    block = threading.Event()
    seen: list[tuple[str, object]] = []
    lock = threading.Lock()
    inside = [0]
    concurrent = [0]

    def cb(key):
        with lock:
            inside[0] += 1
            concurrent[0] = max(concurrent[0], inside[0])
        if key == ("slow",):
            block.wait(10)
        with lock:
            inside[0] -= 1
        seen.append((threading.current_thread().name, key))

    try:
        obs.subscribe(SubscriptionOptions(filter=PrefixFilter(()))).bind(cb)
        pub.notify({("slow",)})
        pub.flush()
        assert _wait_until(lambda: inside[0] == 1)
        old = obs._dispatch_thread
        caplog.set_level(logging.WARNING, logger="virtuals._backends.observers._base")
        t0 = time.monotonic()
        obs.disconnect()
        assert 1.5 < time.monotonic() - t0 < 3.5
        assert old.is_alive()
        assert any(old.name in r.message for r in caplog.records)

        with pytest.raises(ObserverConnectionError, match="still inside a callback"):
            obs.connect()

        block.set()
        old.join(2)
        assert not old.is_alive()
        obs.connect()
        new = obs._dispatch_thread
        assert new is not old
        obs.subscribe(SubscriptionOptions(filter=PrefixFilter(()))).bind(cb)
        for i in range(100):
            pub.notify({("k", i)})
        pub.flush()
        obs.flush(timeout=5)
        workers = {w for w, k in seen if k != ("slow",)}
        assert workers == {new.name}
        assert concurrent[0] == 1
    finally:
        block.set()
        obs.disconnect()
        pub.disconnect()


def test_reconnect_after_clean_disconnect_starts_one_worker():
    _, pub, obs = _pair()
    try:
        first = obs._dispatch_thread
        obs.disconnect()
        assert obs._dispatch_thread is None
        assert not first.is_alive()
        obs.connect()
        workers = [t for t in threading.enumerate() if t.name.startswith("Observer-")]
        assert len(workers) == 1
    finally:
        obs.disconnect()
        pub.disconnect()


# =============================================================================
# N5: a BaseException in a callback does not kill the worker
# =============================================================================


@pytest.mark.parametrize("exc", [SystemExit, KeyboardInterrupt])
def test_base_exception_in_callback_is_logged_and_delivery_continues(exc, caplog):
    _, pub, obs = _pair()
    got: list = []
    try:

        def bad(key):
            raise exc("from a callback")

        obs.subscribe(SubscriptionOptions(filter=PrefixFilter(("bad",)))).bind(bad)
        obs.subscribe(SubscriptionOptions(filter=PrefixFilter(("ok",)))).bind(got.append)
        caplog.set_level(logging.ERROR, logger="virtuals._backends.observers._base")
        pub.notify({("bad", 1)})
        pub.flush()
        obs.flush(timeout=2)
        pub.notify({("ok", 1)})
        pub.flush()
        assert _wait_until(lambda: got == [("ok", 1)])
        assert obs._dispatch_thread.is_alive()
        assert any("callback raised" in r.message for r in caplog.records)
    finally:
        obs.disconnect()
        pub.disconnect()


# =============================================================================
# N9: receivers are iterated as a snapshot
# =============================================================================


def test_unbind_from_inside_a_receiver_does_not_skip_the_next_one():
    _, pub, obs = _pair()
    calls: list[str] = []
    try:
        sub = obs.subscribe(SubscriptionOptions(filter=PrefixFilter(("k",))))

        def second(key):
            calls.append("second")

        def first(key):
            calls.append("first")
            sub.unbind(second)

        sub.bind(first)
        sub.bind(second)
        pub.notify({("k", 1)})
        pub.flush()
        obs.flush(timeout=2)
        assert calls == ["first", "second"]
        pub.notify({("k", 2)})
        pub.flush()
        obs.flush(timeout=2)
        assert calls == ["first", "second", "first"]
    finally:
        obs.disconnect()
        pub.disconnect()


def test_close_from_inside_a_receiver_ends_the_wake():
    _, pub, obs = _pair()
    calls: list[str] = []
    try:
        sub = obs.subscribe(SubscriptionOptions(filter=PrefixFilter(("k",))))
        sub.bind(lambda k: (calls.append("first"), sub.close()))
        sub.bind(lambda k: calls.append("second"))
        pub.notify({("k", 1)})
        pub.flush()
        obs.flush(timeout=2)
        assert calls == ["first"]
        assert sub.is_closed
    finally:
        obs.disconnect()
        pub.disconnect()


# =============================================================================
# F8: an observer dropped without disconnect() is collected, its worker exits
# =============================================================================


def test_dropped_observer_is_collected_and_its_worker_exits():
    tr = InMemoryTransport()
    pub = InMemoryPublisher(tr)
    pub.connect()
    try:
        before = {t.name for t in threading.enumerate()}
        obs = InMemoryObserver(tr)
        obs.connect()
        got: list = []
        obs.subscribe(SubscriptionOptions(filter=PrefixFilter(("k",)))).bind(got.append)
        pub.notify({("k", 1)})
        pub.flush()
        assert _wait_until(lambda: got == [("k", 1)])
        worker = obs._dispatch_thread
        ref = weakref.ref(obs)
        del obs
        gc.collect()
        assert ref() is None, "worker or transport kept the observer alive"
        worker.join(3)
        assert not worker.is_alive()
        assert {t.name for t in threading.enumerate()} <= before | {worker.name}
        # the transport's listener for it is a no-op now
        pub.notify({("k", 2)})
        pub.flush()
    finally:
        pub.disconnect()


def test_dropped_observer_with_an_open_subscription_handle_stays_alive():
    """A live Subscription keeps its observer: dropping only the observer name is not a leak."""
    tr = InMemoryTransport()
    obs = InMemoryObserver(tr)
    obs.connect()
    sub = obs.subscribe(SubscriptionOptions(filter=PrefixFilter(("k",))))
    ref = weakref.ref(obs)
    del obs
    gc.collect()
    assert ref() is not None
    sub.close()
    ref().disconnect()


# =============================================================================
# N7 (base): a failed connect releases what _on_connect opened
# =============================================================================


def test_failed_connect_runs_the_disconnect_hook():
    class Flaky(ObserverBase):
        def __init__(self):
            super().__init__()
            self.opened = 0
            self.closed = 0

        def _on_connect(self):
            self.opened += 1
            raise RuntimeError("boom")

        def _on_disconnect(self):
            self.closed += 1

    obs = Flaky()
    with pytest.raises(ObserverConnectionError, match="boom"):
        obs.connect()
    assert (obs.opened, obs.closed) == (1, 1)
    assert not obs._connected
    assert obs._dispatch_thread is None
    obs.disconnect()  # no-op
    assert obs.closed == 1


# =============================================================================
# review M2: the in-process pair never drops silently
# =============================================================================


def test_in_process_publisher_blocks_by_default_and_never_drops():
    tr = InMemoryTransport()
    pub = InMemoryPublisher(tr, queue_maxsize=2)
    obs = InMemoryObserver(tr, dispatch_queue_maxsize=2)
    pub.connect()
    obs.connect()
    gate = threading.Event()
    got: list = []
    try:

        def slow(k):
            gate.wait(10)
            got.append(k)

        obs.subscribe(SubscriptionOptions(filter=PrefixFilter(("k",)))).bind(slow)
        done = threading.Event()
        keys = [("k", i) for i in range(40)]

        def writer():
            pub.notify(keys)
            done.set()

        t = threading.Thread(target=writer, daemon=True)
        t.start()
        assert not done.wait(1.5), "the in-process publisher dropped instead of blocking"
        gate.set()
        assert done.wait(5)
        pub.flush(5)
        obs.flush(5)
        assert got == keys
        assert pub.dropped_keys == 0
    finally:
        gate.set()
        obs.disconnect()
        pub.disconnect()


def test_in_process_publisher_with_a_timeout_wakes_every_observer_after_a_drop():
    tr = InMemoryTransport()
    pub = InMemoryPublisher(tr, queue_maxsize=2, notify_timeout=0.2)
    obs = InMemoryObserver(tr, dispatch_queue_maxsize=2)
    other = InMemoryObserver(tr)
    pub.connect()
    obs.connect()
    other.connect()
    gate = threading.Event()
    got: list = []
    other_got: list = []
    try:

        def slow(k):
            gate.wait(10)
            got.append(k)

        obs.subscribe(SubscriptionOptions(filter=PrefixFilter(("k",)))).bind(slow)
        other.subscribe(SubscriptionOptions(filter=PrefixFilter(("x",)))).bind(other_got.append)
        t0 = time.monotonic()
        pub.notify([("k", i) for i in range(40)])
        assert time.monotonic() - t0 < 1.0
        assert pub.dropped_keys > 0
        gate.set()
        # once the transport moves again, every observer on it gets a keyless wake
        assert _wait_until(lambda: got.count(None) == 1 and other_got == [None], 5), (
            got,
            other_got,
        )
        with pytest.raises(Exception, match="dropped"):
            pub.flush(2)
    finally:
        gate.set()
        other.disconnect()
        obs.disconnect()
        pub.disconnect()
