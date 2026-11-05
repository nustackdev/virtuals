"""Contract section 5: notifications, over every storage backend and transport.

Each case runs per storage backend (``BACKENDS``) and per notify transport:
the in-process pair (mem) and the redis pair over a private valkey. The
storage publishes after commit; the transport carries it; the observer
delivers to bound receivers.
"""

from __future__ import annotations

import threading
import time
import uuid

import pytest

from virtuals.tkv.filter import PrefixFilter
from virtuals.tkv.observer import SubscriptionOptions


TRANSPORTS = ("mem", "redis")

# How long a case waits for a wake that must come, and how long it watches
# for one that must not.
WAKE = 5.0
QUIET = 0.3


def _wait_until(pred, timeout=WAKE, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return pred()


class NotifyPair:
    """A connected publisher and observer on one transport."""

    def __init__(self, kind: str, redis_url: str | None) -> None:
        self.kind = kind
        if kind == "mem":
            from virtuals.observers.mem import InMemoryObserver
            from virtuals.publishers.mem import InMemoryPublisher
            from virtuals.tkv.transport import InMemoryTransport

            transport = InMemoryTransport()
            self.publisher = InMemoryPublisher(transport)
            self.observer = InMemoryObserver(transport)
            self.probe = None
        else:
            import redis

            from virtuals.observers.redis_pubsub import RedisObserver
            from virtuals.publishers.redis_pubsub import RedisPublisher

            assert redis_url is not None
            self.prefix = f"s5:{uuid.uuid4().hex[:8]}"
            self.publisher = RedisPublisher(
                redis_url=redis_url, channel_prefix=self.prefix, publish_retry_seconds=0.5
            )
            self.observer = RedisObserver(redis_url=redis_url, channel_prefix=self.prefix)
            self.probe = redis.from_url(redis_url)
        self.publisher.connect()
        self.observer.connect()

    def settle(self, timeout: float = 2.0) -> None:
        """Everything published so far has been handed to receivers."""
        self.publisher.flush(timeout=timeout)
        self.observer.flush(timeout=timeout)

    def cut_off_observer(self) -> None:
        """Cut the observer off while the server stays up, until `reconnect_observer`.

        Kills the pubsub connection, holds the re-registration that follows
        the reconnect, and drops the routing index the way a publish reaching
        no subscriber would. Writes made now are routed to nobody.
        """
        assert self.probe is not None
        obs = self.observer
        real = obs._register_hash
        self._hold = threading.Event()
        entered = threading.Event()

        def held(h, entries):
            entered.set()
            self._hold.wait(10)
            return real(h, entries)

        obs._register_hash = held
        assert self.probe.client_kill_filter(_type="pubsub") >= 1
        assert entered.wait(5), "no re-registration after the reconnect"
        for key in self.probe.scan_iter(match=f"{self.prefix}:nx1:*"):
            self.probe.delete(key)

    def reconnect_observer(self) -> None:
        self._hold.set()

    def close(self) -> None:
        self.observer.disconnect()
        self.publisher.disconnect()
        if self.probe is not None:
            for key in self.probe.scan_iter(match=f"{self.prefix}:*"):
                self.probe.delete(key)
            self.probe.close()


@pytest.fixture(params=TRANSPORTS, ids=TRANSPORTS)
def pair(request):
    url = request.getfixturevalue("redis_url") if request.param == "redis" else None
    p = NotifyPair(request.param, url)
    yield p
    p.close()


@pytest.fixture
def notifying_storage(backend, tmp_path, pair):
    st = backend.make(tmp_path, publisher=pair.publisher)
    st.open()
    yield st
    st.close()


def _watch(pair, f, got: list):
    sub = pair.observer.subscribe(SubscriptionOptions(filter=f))
    sub.bind(got.append)
    return sub


# =============================================================================
# 5.1 notify after commit
# =============================================================================


def test_s5_wake_follows_the_commit(notifying_storage, pair, monkeypatch):
    """A receiver woken by a change reads that commit or a later one.

    The writer is held inside the publisher's notify() until the wake has
    been delivered and the receiver has read from its own snapshot. So a
    notify issued before the commit lands is read as the old value every
    time, on every backend, not only when the worker happens to win a race.
    """
    st = notifying_storage
    seen: list = []
    read = threading.Event()

    def on_wake(key):
        if key != ("n", "k"):
            return
        with st.snapshot() as sn:  # a reader of its own: never the writer's view
            seen.append(sn.get(("n", "k")))
        read.set()

    pair.observer.subscribe(SubscriptionOptions(filter=PrefixFilter(prefix=("n",)))).bind(on_wake)
    real_notify = pair.publisher.notify

    def notify_and_wait_for_the_reader(keys):
        real_notify(keys)
        read.wait(WAKE)

    monkeypatch.setattr(pair.publisher, "notify", notify_and_wait_for_the_reader)
    with st.transaction() as t:
        t.put(("n", "k"), 1)
    assert read.is_set(), "no wake for the commit"
    assert seen == [1], f"the wake read {seen}, not the commit"


def test_s5_nothing_on_rollback(notifying_storage, pair):
    """Abort, an exception in the body and an empty transaction wake nobody."""
    st = notifying_storage
    got: list = []
    _watch(pair, PrefixFilter(prefix=("r",)), got)

    t = st.begin_transaction()
    t.put(("r", 1), 1)
    t.abort()
    with pytest.raises(RuntimeError), st.transaction() as t:
        t.put(("r", 2), 2)
        raise RuntimeError("body failed")
    with st.transaction() as t:
        pass
    pair.settle()
    time.sleep(QUIET)
    assert got == []
    # the channel works: a real commit arrives
    with st.transaction() as t:
        t.put(("r", 3), 3)
    assert _wait_until(lambda: got == [("r", 3)]), got


def test_s5_write_batch_notifies(notifying_storage, pair):
    st = notifying_storage
    got: list = []
    _watch(pair, PrefixFilter(prefix=("b",)), got)
    with st.transaction() as t:
        t.put(("b", "old"), 0)
    assert _wait_until(lambda: got == [("b", "old")])
    with st.begin_write_batch() as b:
        b.put(("b", "new"), 1)
        b.delete(("b", "old"))
    assert _wait_until(lambda: sorted(got[1:]) == [("b", "new"), ("b", "old")]), got


def test_s5_one_wake_per_subscription_per_commit(notifying_storage, pair):
    """A key matching two subscriptions with different filters wakes each once."""
    st = notifying_storage
    a: list = []
    b: list = []
    _watch(pair, PrefixFilter(prefix=("o",)), a)
    _watch(pair, PrefixFilter(prefix=("o", "x")), b)
    with st.transaction() as t:
        t.put(("o", "x", 1), 1)
    assert _wait_until(lambda: a == [("o", "x", 1)] and b == [("o", "x", 1)])
    pair.settle()
    time.sleep(QUIET)
    assert a == [("o", "x", 1)] and b == [("o", "x", 1)]


# =============================================================================
# 5.2 subscribe is ready on return
# =============================================================================


def test_s5_subscribe_ready_on_return(notifying_storage, pair):
    """Once subscribe() returns, the very next commit is delivered. No window."""
    st = notifying_storage
    for i in range(20):
        got: list = []
        sub = _watch(pair, PrefixFilter(prefix=("s", i)), got)
        with st.transaction() as t:
            t.put(("s", i), i)
        assert _wait_until(lambda got=got, i=i: got == [("s", i)]), (i, got)
        sub.close()


# =============================================================================
# 5.3 watches cover what they name
# =============================================================================


def test_s5_watch_covers_the_subtree(notifying_storage, pair):
    """A hand-built prefix watch sees every row under the address and no sibling."""
    st = notifying_storage
    got: list = []
    _watch(pair, PrefixFilter(prefix=("t", "a")), got)
    with st.transaction() as t:
        t.put(("t", "a"), 0)
        t.put(("t", "a", "deep"), 1)
        t.put(("t", "a", "deeper", "still"), 2)
        t.put(("t", "ab"), 3)  # a sibling whose name only starts the same
        t.put(("t",), 4)
    assert _wait_until(
        lambda: sorted(got) == [("t", "a"), ("t", "a", "deep"), ("t", "a", "deeper", "still")]
    ), got
    pair.settle()
    time.sleep(QUIET)
    assert ("t", "ab") not in got and ("t",) not in got


DICT_VIEWS = ("dict", "log_indexed", "fractional")


def _dict_view_class(kind: str):
    from virtuals._views import (
        EagerDictView,
        EagerFractionalIndexedDictView,
        EagerLogIndexedDictView,
    )

    return {
        "dict": EagerDictView,
        "log_indexed": EagerLogIndexedDictView,
        "fractional": EagerFractionalIndexedDictView,
    }[kind]


@pytest.mark.parametrize("kind", DICT_VIEWS)
def test_s5_untyped_child_watch_covers_the_child_subtree(notifying_storage, pair, kind):
    """The watch a view builds for an untyped child address fires on a write
    deep inside the compound value stored there, and not on a sibling."""
    from virtuals._views import EagerDictView
    from virtuals.navigator import Navigator

    st = notifying_storage
    cls = _dict_view_class(kind)
    with st.transaction() as tx:
        doc = Navigator(st).root(tx).open_child("doc", cls)
        doc["a"] = {"deep": 1, "deeper": {"still": 2}}
        doc["ab"] = 3  # a sibling whose name only starts the same
        options = doc.on_child_change("a")
    pair.settle()  # the setup writes are delivered before anyone watches
    got: list = []
    pair.observer.subscribe(options).bind(got.append)

    with st.transaction() as tx:
        a = Navigator(st).root(tx).open_child("doc", cls).open_child("a", EagerDictView)
        a["deep"] = 10
        a.open_child("deeper", EagerDictView)["still"] = 20
    assert _wait_until(lambda: any(k[-2:] == ("deeper", "still") for k in got)), got
    assert any(k[-1] == "deep" for k in got), got
    with st.transaction() as tx:
        Navigator(st).root(tx).open_child("doc", cls)["ab"] = 4
    pair.settle()
    time.sleep(QUIET)
    assert all("ab" not in k for k in got), got


# =============================================================================
# 5.5 delivery resumes after a reconnect
# =============================================================================


def test_s5_delivery_resumes_after_reconnect(notifying_storage, pair):
    """Writes made while the observer is cut off are not delivered; once it is
    re-registered keyed delivery resumes and a level reader catches up on its
    next wake."""
    if pair.kind == "mem":
        pytest.skip("the in-process transport has no connection to lose")
    from virtuals.tkv.storage import StorageScanOptions

    st = notifying_storage
    truth: dict = {}
    view: dict = {}
    wakes: list = []
    lock = threading.Lock()

    def on_wake(key):
        wakes.append(key)
        with st.snapshot() as sn, lock:
            view.clear()
            view.update(
                sn.scan(
                    StorageScanOptions(start=("lv",), break_filter=PrefixFilter(prefix=("lv",)))
                ).items()
            )

    def write(i: int) -> None:
        with st.transaction() as t:
            t.put(("lv", i % 7), i)
        truth[("lv", i % 7)] = i

    pair.observer.subscribe(SubscriptionOptions(filter=PrefixFilter(prefix=("lv",)))).bind(on_wake)
    for i in range(30):
        write(i)
    # a barrier write: once its wake is in, every earlier keyed wake is too
    # (one ordered channel), so nothing from the setup lands in the gap
    with st.transaction() as t:
        t.put(("lv", "barrier"), 1)
    truth[("lv", "barrier")] = 1
    assert _wait_until(lambda: wakes and wakes[-1] == ("lv", "barrier")), wakes
    assert view == truth

    pair.cut_off_observer()
    for i in range(30, 60):
        write(i)
    pair.publisher.flush(2)  # these reached nobody
    time.sleep(QUIET)
    assert view != truth
    pair.reconnect_observer()

    # keyed delivery resumes once the filter is re-registered
    deadline = time.monotonic() + 10.0
    i = 60
    while True:
        write(i)
        key = ("lv", i % 7)
        if _wait_until(lambda key=key: wakes[-1] == key and view == truth, 0.2):
            break
        assert time.monotonic() < deadline, (wakes, view, truth)
        i += 1
