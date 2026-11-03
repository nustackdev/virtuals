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


# =============================================================================
# r2 N1: disconnect() ends delivery, does not merely stop waiting for it
# =============================================================================


def test_disconnect_returns_with_no_callback_after_it_and_connect_works_at_once():
    """A long backlog with a slow receiver: disconnect() closes every subscription,
    so no receiver runs after it returns, and the next connect() is not refused."""
    _, pub, obs = _pair()
    started: list[float] = []
    gate = threading.Event()
    gate.set()
    try:
        sub = obs.subscribe(SubscriptionOptions(filter=PrefixFilter(("q",))))

        def cb(key):
            started.append(time.monotonic())
            time.sleep(0.01)

        sub.bind(cb)
        pub.notify([("q", i) for i in range(1500)])
        pub.flush(5)
        assert _wait_until(lambda: len(started) >= 5)
        old = obs._dispatch_thread
        t0 = time.monotonic()
        obs.disconnect()
        returned = time.monotonic()
        assert returned - t0 < 1.0, "disconnect waited out the join instead of ending delivery"
        assert sub.is_closed
        assert old is not None and _wait_until(lambda: not old.is_alive(), 1.0)
        time.sleep(0.2)
        late = [t for t in started if t > returned]
        assert late == [], f"{len(late)} receivers ran after disconnect() returned"
        assert len(started) < 1500
        obs.connect()  # not refused: the old worker is gone
        assert obs._dispatch_thread is not old
    finally:
        gate.set()
        obs.disconnect()
        pub.disconnect()


# =============================================================================
# r2 N5: the worker takes a bounded batch, so the queue shows the backlog
# =============================================================================


def test_worker_takes_at_most_one_batch_off_the_queue_per_round():
    from virtuals._backends.observers._base import DISPATCH_BATCH_MAX

    _, pub, obs = _pair()
    sem = threading.Semaphore(0)
    total = 1000
    try:

        def cb(key):
            sem.acquire(timeout=10)

        obs.subscribe(SubscriptionOptions(filter=PrefixFilter(("k",)))).bind(cb)
        pub.notify([("k", "slow")])
        pub.flush(5)
        assert _wait_until(lambda: obs._dispatch_queue.qsize() == 0)
        pub.notify([("k", i) for i in range(total)])
        pub.flush(5)
        assert obs._dispatch_queue.qsize() == total
        sem.release()  # the worker leaves the slow key and takes its next batch
        assert _wait_until(lambda: obs._dispatch_queue.qsize() < total)
        time.sleep(0.1)
        left = obs._dispatch_queue.qsize()
        assert left >= total - DISPATCH_BATCH_MAX, f"the worker took {total - left} keys at once"
    finally:
        for _ in range(total + 2):
            sem.release()
        obs.disconnect()
        pub.disconnect()


# =============================================================================
# r2 N3: resync wakes coalesce per filter hash until delivered
# =============================================================================


def test_resync_wakes_asked_for_while_one_is_pending_collapse_into_it():
    """Three publishers dropping in one stall ask three times; each subscription
    re-reads once. After that wake is delivered the next request wakes again."""
    tr, pub, obs = _pair()
    hold = threading.Event()
    a: list = []
    b: list = []
    try:

        def slow(key):
            hold.wait(10)
            a.append(key)

        obs.subscribe(SubscriptionOptions(filter=PrefixFilter(("a",)))).bind(slow)
        obs.subscribe(SubscriptionOptions(filter=PrefixFilter(("b",)))).bind(b.append)
        pub.notify([("a", 0)])
        pub.flush(5)
        assert _wait_until(lambda: obs._dispatch_queue.qsize() == 0)  # worker inside slow()
        for _ in range(3):
            tr.resync()  # what each dropping publisher does
        hold.set()
        obs.flush(5)
        time.sleep(0.1)
        assert a == [("a", 0), None] and b == [None], (a, b)
        tr.resync()
        obs.flush(5)
        assert a == [("a", 0), None, None] and b == [None, None], (a, b)
    finally:
        hold.set()
        obs.disconnect()
        pub.disconnect()


# =============================================================================
# r2 N4: flush(timeout) returns at the timeout on a full queue
# =============================================================================


def test_flush_honours_its_timeout_on_a_full_dispatch_queue():
    tr = InMemoryTransport()
    pub = InMemoryPublisher(tr)
    obs = InMemoryObserver(tr, dispatch_queue_maxsize=5)
    pub.connect()
    obs.connect()
    hold = threading.Event()
    try:
        obs.subscribe(SubscriptionOptions(filter=PrefixFilter(("z",)))).bind(
            lambda k: hold.wait(10)
        )
        filler = threading.Thread(
            target=lambda: pub.notify([("z", i) for i in range(40)]), daemon=True
        )
        filler.start()
        assert _wait_until(lambda: obs._dispatch_queue.full())
        t0 = time.monotonic()
        obs.flush(timeout=0.3)
        assert time.monotonic() - t0 < 1.0, "flush waited for room instead of its timeout"
    finally:
        hold.set()
        obs.disconnect()
        pub.disconnect()


# =============================================================================
# review B1: a resync wake that finds no subscription releases its hash
# =============================================================================


def test_resync_delivered_to_an_empty_registry_does_not_block_the_next_one():
    """Every subscription closes before a pending wake is delivered; a later
    subscription on the same filter still gets a keyless wake on the next request."""
    tr, pub, obs = _pair()
    hold = threading.Event()
    try:
        blk = obs.subscribe(SubscriptionOptions(filter=PrefixFilter(("x",))))
        blk.bind(lambda k: hold.wait(10))
        pub.notify([("x", 0)])
        pub.flush(5)
        assert _wait_until(lambda: obs._dispatch_queue.qsize() == 0)  # worker inside blk
        s1 = obs.subscribe(SubscriptionOptions(filter=PrefixFilter(("a",))))
        tr.resync()  # wakes queued for hash(a) and hash(x)
        s1.close()
        blk.close()  # registry empty before the batch is delivered
        hold.set()
        obs.flush(5)
        assert _wait_until(lambda: not obs._resync_queued), obs._resync_queued
        got: list = []
        obs.subscribe(SubscriptionOptions(filter=PrefixFilter(("a",)))).bind(got.append)
        tr.resync()
        obs.flush(5)
        assert _wait_until(lambda: got == [None]), got
    finally:
        hold.set()
        obs.disconnect()
        pub.disconnect()


# =============================================================================
# review m1: disconnect() from inside a callback still runs the hook
# =============================================================================


def test_disconnect_from_inside_a_callback_runs_the_disconnect_hook():
    tr, pub, obs = _pair()
    err: list = []
    try:

        def cb(key):
            try:
                obs.disconnect()
            except BaseException as e:  # the test records whatever escapes
                err.append(repr(e))

        obs.subscribe(SubscriptionOptions(filter=PrefixFilter(("a",)))).bind(cb)
        worker = obs._dispatch_thread
        pub.notify([("a", 1)])
        pub.flush(5)
        assert _wait_until(lambda: not obs._connected)
        assert err == []
        assert tr._listeners == [], "the transport listener was left registered"
        assert _wait_until(lambda: not worker.is_alive())
        obs.connect()
        got: list = []
        obs.subscribe(SubscriptionOptions(filter=PrefixFilter(("a",)))).bind(got.append)
        pub.notify([("a", 2)])
        pub.flush(5)
        assert _wait_until(lambda: got == [("a", 2)])
    finally:
        obs.disconnect()
        pub.disconnect()


# =============================================================================
# review n1: subscribe() racing disconnect() never returns an open dead subscription
# =============================================================================


def test_subscribe_during_disconnect_raises_and_reports_closed(monkeypatch):
    _, pub, obs = _pair()
    seen: list = []
    real = obs._on_subscribe

    def hook(sub):
        real(sub)
        seen.append(sub)
        obs.disconnect()  # the race: disconnect lands after the registry insert

    monkeypatch.setattr(obs, "_on_subscribe", hook)
    try:
        with pytest.raises(ObserverConnectionError, match="disconnected"):
            obs.subscribe(SubscriptionOptions(filter=PrefixFilter(("a",))))
        assert len(seen) == 1 and seen[0].is_closed
        with pytest.raises(ValueError, match="closed"):
            seen[0].bind(lambda k: None)
    finally:
        obs.disconnect()
        pub.disconnect()
