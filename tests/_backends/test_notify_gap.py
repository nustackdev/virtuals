"""Redis notify pair: a write committed 0ms after subscribe() returns is delivered.

- in-process: 200 trials, publisher and observer in this process
- cross-process: 200 trials, publisher in a spawned child that writes the
  moment it hears subscribe() returned
- 7 filter types deliver exactly what Filter.matches says, end to end
- edge keys (negative / huge ints, unicode, int vs str) end to end
"""

from __future__ import annotations

import multiprocessing as mp
import threading
import time

import pytest


pytestmark = pytest.mark.redis

redis = pytest.importorskip("redis")
pytest.importorskip("msgpack")

TRIALS = 200


def _pair(url, prefix):
    from virtuals._backends.observers.redis_pubsub import RedisObserver
    from virtuals._backends.publishers.redis_pubsub import RedisPublisher

    pub = RedisPublisher(redis_url=url, channel_prefix=prefix)
    pub.connect()
    obs = RedisObserver(redis_url=url, channel_prefix=prefix)
    obs.connect()
    return pub, obs


@pytest.fixture
def prefix(unique_channel_prefix, redis_cleanup):
    redis_cleanup(unique_channel_prefix)
    return unique_channel_prefix


def test_zero_gap_in_process(redis_url, prefix):
    from virtuals.tkv.filter import PrefixFilter
    from virtuals.tkv.observer import SubscriptionOptions

    pub, obs = _pair(redis_url, prefix)
    try:
        got = 0
        for t in range(TRIALS):
            ev = threading.Event()
            sub = obs.subscribe(SubscriptionOptions(filter=PrefixFilter(prefix=("g", t))))
            sub.bind(lambda k, ev=ev: ev.set())
            pub.notify(("g", t, "x"))
            got += ev.wait(1.0)
            sub.close()
        assert got == TRIALS
    finally:
        obs.disconnect()
        pub.disconnect()


def _publisher_proc(url, prefix, conn):
    from virtuals._backends.publishers.redis_pubsub import RedisPublisher

    pub = RedisPublisher(redis_url=url, channel_prefix=prefix)
    pub.connect()
    conn.send("up")
    while True:
        t = conn.recv()
        if t is None:
            break
        pub.notify(("g", t, "x"))
    pub.flush(timeout=2.0)
    pub.disconnect()


def test_zero_gap_cross_process(redis_url, prefix):
    from virtuals._backends.observers.redis_pubsub import RedisObserver
    from virtuals.tkv.filter import PrefixFilter
    from virtuals.tkv.observer import SubscriptionOptions

    ctx = mp.get_context("spawn")
    a, b = ctx.Pipe()
    p = ctx.Process(target=_publisher_proc, args=(redis_url, prefix, b), daemon=True)
    p.start()
    obs = RedisObserver(redis_url=redis_url, channel_prefix=prefix)
    obs.connect()
    try:
        assert a.poll(30) and a.recv() == "up"
        got = 0
        for t in range(TRIALS):
            ev = threading.Event()
            sub = obs.subscribe(SubscriptionOptions(filter=PrefixFilter(prefix=("g", t))))
            sub.bind(lambda k, ev=ev: ev.set())
            a.send(t)
            got += ev.wait(1.0)
            sub.close()
        assert got == TRIALS
    finally:
        a.send(None)
        p.join(10)
        if p.is_alive():
            p.kill()
        obs.disconnect()


def _collect(pub, obs, filters, keys):
    from virtuals.tkv.observer import SubscriptionOptions

    got = {n: [] for n in filters}
    subs = []
    for n, f in filters.items():
        s = obs.subscribe(SubscriptionOptions(filter=f))
        s.bind(lambda k, n=n: got[n].append(k))
        subs.append(s)
    for k in keys:
        pub.notify(k)
    pub.flush(timeout=2.0)
    deadline = time.monotonic() + 3.0
    expected = {n: sorted(map(repr, (k for k in keys if f.matches(k)))) for n, f in filters.items()}
    while time.monotonic() < deadline:
        obs.flush()
        if all(sorted(map(repr, got[n])) == expected[n] for n in filters):
            break
        time.sleep(0.02)
    time.sleep(0.1)  # nothing extra trickles in
    obs.flush()
    for s in subs:
        s.close()
    return {n: sorted(map(repr, v)) for n, v in got.items()}, expected


def test_filter_types_exact(redis_url, prefix):
    from virtuals.tkv.filter import (
        And,
        LengthFilter,
        Or,
        PrefixFilter,
        SuffixFilter,
        WildcardFilter,
    )

    filters = {
        "prefix": PrefixFilter(prefix=("u", 1)),
        "suffix": SuffixFilter(suffix=("profile",)),
        "length": LengthFilter(length=4),
        "wild": WildcardFilter(pattern=("u", "*", "profile")),
        "and": And(filters=(PrefixFilter(prefix=("u",)), LengthFilter(length=2))),
        "or": Or(
            filters=(PrefixFilter(prefix=("p",)), WildcardFilter(pattern=("*", "*", "settings")))
        ),
        "orand": Or(
            filters=(
                And(filters=(PrefixFilter(prefix=("q",)), SuffixFilter(suffix=("z",)))),
                LengthFilter(length=5),
            )
        ),
    }
    keys = [
        ("u", 1), ("u", 1, "profile"), ("u", "1", "profile"), ("u", 2, "profile"),
        ("u", 2, "settings"), ("a", "b", "c", "d"), ("p", 9), ("q", "z"), ("q", "y"),
        ("a", "b", "c", "d", "e"), ("x", "profile"),
    ]  # fmt: skip
    pub, obs = _pair(redis_url, prefix)
    try:
        got, expected = _collect(pub, obs, filters, keys)
        assert got == expected
    finally:
        obs.disconnect()
        pub.disconnect()


def test_edge_keys_exact(redis_url, prefix):
    from virtuals.tkv.filter import PrefixFilter, SuffixFilter, WildcardFilter

    big = 2**60
    filters = {
        "neg": PrefixFilter(prefix=("n", -1)),
        "big": PrefixFilter(prefix=("b", big)),
        "big1": WildcardFilter(pattern=("b", big + 1)),
        "uni": SuffixFilter(suffix=("зд", "🙂")),
        "int": PrefixFilter(prefix=("t", 1)),
        "str": PrefixFilter(prefix=("t", "1")),
    }
    keys = [
        ("n", -1), ("n", 1), ("n", -1, "x"), ("b", big), ("b", big + 1), ("b", big, 0),
        ("x", "зд", "🙂"), ("зд", "🙂"), ("x", "зд🙂"), ("t", 1), ("t", "1"), ("t", True),
    ]  # fmt: skip
    pub, obs = _pair(redis_url, prefix)
    try:
        got, expected = _collect(pub, obs, filters, keys)
        assert got == expected
    finally:
        obs.disconnect()
        pub.disconnect()
