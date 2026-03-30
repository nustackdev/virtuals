"""Benchmark: observer overhead measurement (v2 - fire-and-forget).

Measures the cost of observer machinery during writes:
- No observer (observer=None) - baseline
- InMemoryObserver enabled/disabled x subscription count
- RedisObserver enabled/disabled x subscription count

notify() is now fire-and-forget (enqueue to deque). We measure:
1. Write batch cost (what the write path actually pays)
2. Raw notify() cost (enqueue only)
3. End-to-end with flush() (enqueue + bg thread matching + delivery)

Run with: pytest tests/tkv/bench_observer_overhead.py -v --benchmark-only
"""

from __future__ import annotations

import pytest

from virtuals.codecs import NoOpCodec
from virtuals.observers.mem import InMemoryObserver
from virtuals.storages.mem import InMemoryStorage
from virtuals.tkv.filter import PrefixFilter
from virtuals.tkv.observer import SubscriptionOptions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_inmem_observer(num_subscriptions: int = 0) -> InMemoryObserver:
    observer = InMemoryObserver(codec=NoOpCodec())
    observer.connect()
    _add_subs(observer, num_subscriptions)
    return observer


def _make_redis_observer(num_subscriptions: int = 0):
    from virtuals.observers.redis_pubsub import RedisObserver

    observer = RedisObserver(codec=NoOpCodec(), redis_url="redis://localhost:6380")
    observer.connect()
    _add_subs(observer, num_subscriptions)
    return observer


def _add_subs(observer, n: int) -> None:
    for i in range(n):
        sub = observer.subscribe(
            SubscriptionOptions(filter=PrefixFilter(prefix=("data", f"sub{i}")))
        )
        sub.bind(lambda key: None)


def _make_storage(observer=None) -> InMemoryStorage:
    storage = InMemoryStorage(codec=NoOpCodec(), observer=observer)
    storage.open()
    return storage


KEY_COUNTS = [100, 1_000, 10_000]


# ---------------------------------------------------------------------------
# Write batch: no observer vs inmem vs redis
# ---------------------------------------------------------------------------


class TestWriteBatchOverhead:
    """Measure full write-batch cost (put N keys + commit).

    notify() is fire-and-forget now, so this measures enqueue cost only.
    """

    @pytest.mark.parametrize("n_keys", KEY_COUNTS)
    def test_no_observer(self, benchmark, n_keys: int) -> None:
        storage = _make_storage()

        def run():
            with storage.batch_write() as wb:
                for i in range(n_keys):
                    wb.put(("data", f"key{i}"), f"value{i}")

        try:
            benchmark(run)
        finally:
            storage.close()

    @pytest.mark.parametrize("n_keys", KEY_COUNTS)
    def test_inmem_0_subs(self, benchmark, n_keys: int) -> None:
        observer = _make_inmem_observer(0)
        storage = _make_storage(observer)

        def run():
            with storage.batch_write() as wb:
                for i in range(n_keys):
                    wb.put(("data", f"key{i}"), f"value{i}")

        try:
            benchmark(run)
        finally:
            storage.close()
            observer.disconnect()

    @pytest.mark.parametrize("n_keys", KEY_COUNTS)
    def test_inmem_1_sub(self, benchmark, n_keys: int) -> None:
        observer = _make_inmem_observer(1)
        storage = _make_storage(observer)

        def run():
            with storage.batch_write() as wb:
                for i in range(n_keys):
                    wb.put(("data", f"key{i}"), f"value{i}")

        try:
            benchmark(run)
        finally:
            storage.close()
            observer.disconnect()

    @pytest.mark.parametrize("n_keys", KEY_COUNTS)
    def test_redis_0_subs(self, benchmark, n_keys: int) -> None:
        observer = _make_redis_observer(0)
        storage = _make_storage(observer)

        def run():
            with storage.batch_write() as wb:
                for i in range(n_keys):
                    wb.put(("data", f"key{i}"), f"value{i}")

        try:
            benchmark(run)
        finally:
            storage.close()
            observer.disconnect()

    @pytest.mark.parametrize("n_keys", KEY_COUNTS)
    def test_redis_1_sub(self, benchmark, n_keys: int) -> None:
        observer = _make_redis_observer(1)
        storage = _make_storage(observer)

        def run():
            with storage.batch_write() as wb:
                for i in range(n_keys):
                    wb.put(("data", f"key{i}"), f"value{i}")

        try:
            benchmark(run)
        finally:
            storage.close()
            observer.disconnect()


# ---------------------------------------------------------------------------
# Raw notify() - fire-and-forget enqueue cost
# ---------------------------------------------------------------------------


class TestNotifyEnqueue:
    """Measure raw notify() cost (enqueue to deque, no matching)."""

    def test_inmem_single_key(self, benchmark) -> None:
        observer = _make_inmem_observer(0)
        key = ("data", "key0")
        try:
            benchmark(observer.notify, key)
        finally:
            observer.disconnect()

    def test_inmem_batch_100(self, benchmark) -> None:
        observer = _make_inmem_observer(0)
        keys = [("data", f"key{i}") for i in range(100)]
        try:
            benchmark(observer.notify, keys)
        finally:
            observer.disconnect()

    def test_inmem_batch_1000(self, benchmark) -> None:
        observer = _make_inmem_observer(0)
        keys = [("data", f"key{i}") for i in range(1000)]
        try:
            benchmark(observer.notify, keys)
        finally:
            observer.disconnect()

    def test_redis_single_key(self, benchmark) -> None:
        observer = _make_redis_observer(0)
        key = ("data", "key0")
        try:
            benchmark(observer.notify, key)
        finally:
            observer.disconnect()

    def test_redis_batch_100(self, benchmark) -> None:
        observer = _make_redis_observer(0)
        keys = [("data", f"key{i}") for i in range(100)]
        try:
            benchmark(observer.notify, keys)
        finally:
            observer.disconnect()


# ---------------------------------------------------------------------------
# End-to-end: notify + flush (enqueue + bg matching + delivery)
# ---------------------------------------------------------------------------


class TestNotifyFlush:
    """Measure notify + flush: full round-trip including bg thread work."""

    def test_inmem_0_subs_100_keys(self, benchmark) -> None:
        observer = _make_inmem_observer(0)
        keys = [("data", f"key{i}") for i in range(100)]

        def run():
            observer.notify(keys)
            observer.flush()

        try:
            benchmark(run)
        finally:
            observer.disconnect()

    def test_inmem_0_subs_1000_keys(self, benchmark) -> None:
        observer = _make_inmem_observer(0)
        keys = [("data", f"key{i}") for i in range(1000)]

        def run():
            observer.notify(keys)
            observer.flush()

        try:
            benchmark(run)
        finally:
            observer.disconnect()

    def test_inmem_1_sub_100_keys(self, benchmark) -> None:
        observer = _make_inmem_observer(1)
        keys = [("data", f"key{i}") for i in range(100)]

        def run():
            observer.notify(keys)
            observer.flush()

        try:
            benchmark(run)
        finally:
            observer.disconnect()

    def test_inmem_1_sub_1000_keys(self, benchmark) -> None:
        observer = _make_inmem_observer(1)
        keys = [("data", f"key{i}") for i in range(1000)]

        def run():
            observer.notify(keys)
            observer.flush()

        try:
            benchmark(run)
        finally:
            observer.disconnect()

    def test_redis_0_subs_100_keys(self, benchmark) -> None:
        observer = _make_redis_observer(0)
        keys = [("data", f"key{i}") for i in range(100)]

        def run():
            observer.notify(keys)
            observer.flush()

        try:
            benchmark(run)
        finally:
            observer.disconnect()

    def test_redis_0_subs_1000_keys(self, benchmark) -> None:
        observer = _make_redis_observer(0)
        keys = [("data", f"key{i}") for i in range(1000)]

        def run():
            observer.notify(keys)
            observer.flush()

        try:
            benchmark(run)
        finally:
            observer.disconnect()

    def test_redis_1_sub_100_keys(self, benchmark) -> None:
        observer = _make_redis_observer(1)
        keys = [("data", f"key{i}") for i in range(100)]

        def run():
            observer.notify(keys)
            observer.flush()

        try:
            benchmark(run)
        finally:
            observer.disconnect()

    def test_redis_1_sub_1000_keys(self, benchmark) -> None:
        observer = _make_redis_observer(1)
        keys = [("data", f"key{i}") for i in range(1000)]

        def run():
            observer.notify(keys)
            observer.flush()

        try:
            benchmark(run)
        finally:
            observer.disconnect()
