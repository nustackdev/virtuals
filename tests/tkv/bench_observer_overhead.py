"""Benchmark: publisher/observer overhead measurement (v3 - split pubsub).

Measures the cost of publisher machinery during writes:
- No publisher (publisher=None) - baseline
- InMemoryPublisher enabled/disabled x subscription count
- RedisPublisher enabled/disabled x subscription count

publisher.notify() is fire-and-forget (enqueue to deque). We measure:
1. Write batch cost (what the write path actually pays)
2. Raw publisher.notify() cost (enqueue only)
3. End-to-end with publisher.flush() (enqueue + bg thread + transport +
   observer bg thread + delivery)

Run with: pytest tests/tkv/bench_observer_overhead.py -v --benchmark-only
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from virtuals.codecs import NoOpCodec
from virtuals.observers.mem import InMemoryObserver
from virtuals.publishers.mem import InMemoryPublisher
from virtuals.storages.mem import InMemoryStorage
from virtuals.tkv.filter import PrefixFilter
from virtuals.tkv.observer import SubscriptionOptions
from virtuals.tkv.transport import InMemoryTransport


if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _InMemBundle:
    transport: InMemoryTransport
    publisher: InMemoryPublisher
    observer: InMemoryObserver

    def disconnect(self) -> None:
        self.observer.disconnect()
        self.publisher.disconnect()


def _make_inmem_bundle(num_subscriptions: int = 0) -> _InMemBundle:
    transport = InMemoryTransport()
    publisher = InMemoryPublisher(transport)
    observer = InMemoryObserver(transport)
    publisher.connect()
    observer.connect()
    _add_subs(observer, num_subscriptions)
    return _InMemBundle(transport=transport, publisher=publisher, observer=observer)


def _make_redis_bundle(num_subscriptions: int = 0):
    from virtuals._backends.publishers.redis_pubsub import RedisPublisher
    from virtuals.observers.redis_pubsub import RedisObserver

    publisher = RedisPublisher(redis_url="redis://localhost:6380")
    observer = RedisObserver(redis_url="redis://localhost:6380")
    publisher.connect()
    observer.connect()
    _add_subs(observer, num_subscriptions)
    return publisher, observer


def _add_subs(observer, n: int) -> None:
    for i in range(n):
        sub = observer.subscribe(
            SubscriptionOptions(filter=PrefixFilter(prefix=("data", f"sub{i}")))
        )
        sub.bind(lambda key: None)


def _make_storage(publisher=None) -> InMemoryStorage:
    storage = InMemoryStorage(codec=NoOpCodec(), publisher=publisher)
    storage.open()
    return storage


KEY_COUNTS = [100, 1_000, 10_000]


# ---------------------------------------------------------------------------
# Write batch: no publisher vs inmem vs redis
# ---------------------------------------------------------------------------


class TestWriteBatchOverhead:
    """Measure full write-batch cost (put N keys + commit).

    publisher.notify() is fire-and-forget, so this measures enqueue cost only.
    """

    @pytest.mark.parametrize("n_keys", KEY_COUNTS)
    def test_no_publisher(self, benchmark, n_keys: int) -> None:
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
        bundle = _make_inmem_bundle(0)
        storage = _make_storage(bundle.publisher)

        def run():
            with storage.batch_write() as wb:
                for i in range(n_keys):
                    wb.put(("data", f"key{i}"), f"value{i}")

        try:
            benchmark(run)
        finally:
            storage.close()
            bundle.disconnect()

    @pytest.mark.parametrize("n_keys", KEY_COUNTS)
    def test_inmem_1_sub(self, benchmark, n_keys: int) -> None:
        bundle = _make_inmem_bundle(1)
        storage = _make_storage(bundle.publisher)

        def run():
            with storage.batch_write() as wb:
                for i in range(n_keys):
                    wb.put(("data", f"key{i}"), f"value{i}")

        try:
            benchmark(run)
        finally:
            storage.close()
            bundle.disconnect()

    @pytest.mark.parametrize("n_keys", KEY_COUNTS)
    def test_redis_0_subs(self, benchmark, n_keys: int) -> None:
        publisher, observer = _make_redis_bundle(0)
        storage = _make_storage(publisher)

        def run():
            with storage.batch_write() as wb:
                for i in range(n_keys):
                    wb.put(("data", f"key{i}"), f"value{i}")

        try:
            benchmark(run)
        finally:
            storage.close()
            publisher.disconnect()
            observer.disconnect()

    @pytest.mark.parametrize("n_keys", KEY_COUNTS)
    def test_redis_1_sub(self, benchmark, n_keys: int) -> None:
        publisher, observer = _make_redis_bundle(1)
        storage = _make_storage(publisher)

        def run():
            with storage.batch_write() as wb:
                for i in range(n_keys):
                    wb.put(("data", f"key{i}"), f"value{i}")

        try:
            benchmark(run)
        finally:
            storage.close()
            publisher.disconnect()
            observer.disconnect()


# ---------------------------------------------------------------------------
# Raw publisher.notify() - fire-and-forget enqueue cost
# ---------------------------------------------------------------------------


class TestNotifyEnqueue:
    """Measure raw publisher.notify() cost (enqueue to deque, no matching)."""

    def test_inmem_single_key(self, benchmark) -> None:
        bundle = _make_inmem_bundle(0)
        key = ("data", "key0")
        try:
            benchmark(bundle.publisher.notify, key)
        finally:
            bundle.disconnect()

    def test_inmem_batch_100(self, benchmark) -> None:
        bundle = _make_inmem_bundle(0)
        keys = [("data", f"key{i}") for i in range(100)]
        try:
            benchmark(bundle.publisher.notify, keys)
        finally:
            bundle.disconnect()

    def test_inmem_batch_1000(self, benchmark) -> None:
        bundle = _make_inmem_bundle(0)
        keys = [("data", f"key{i}") for i in range(1000)]
        try:
            benchmark(bundle.publisher.notify, keys)
        finally:
            bundle.disconnect()

    def test_redis_single_key(self, benchmark) -> None:
        publisher, observer = _make_redis_bundle(0)
        key = ("data", "key0")
        try:
            benchmark(publisher.notify, key)
        finally:
            publisher.disconnect()
            observer.disconnect()

    def test_redis_batch_100(self, benchmark) -> None:
        publisher, observer = _make_redis_bundle(0)
        keys = [("data", f"key{i}") for i in range(100)]
        try:
            benchmark(publisher.notify, keys)
        finally:
            publisher.disconnect()
            observer.disconnect()


# ---------------------------------------------------------------------------
# End-to-end: publisher.notify + publisher.flush
# ---------------------------------------------------------------------------


class TestNotifyFlush:
    """Measure notify + flush: full round-trip including bg thread work."""

    def test_inmem_0_subs_100_keys(self, benchmark) -> None:
        bundle = _make_inmem_bundle(0)
        keys = [("data", f"key{i}") for i in range(100)]

        def run():
            bundle.publisher.notify(keys)
            bundle.publisher.flush()

        try:
            benchmark(run)
        finally:
            bundle.disconnect()

    def test_inmem_0_subs_1000_keys(self, benchmark) -> None:
        bundle = _make_inmem_bundle(0)
        keys = [("data", f"key{i}") for i in range(1000)]

        def run():
            bundle.publisher.notify(keys)
            bundle.publisher.flush()

        try:
            benchmark(run)
        finally:
            bundle.disconnect()

    def test_inmem_1_sub_100_keys(self, benchmark) -> None:
        bundle = _make_inmem_bundle(1)
        keys = [("data", f"key{i}") for i in range(100)]

        def run():
            bundle.publisher.notify(keys)
            bundle.publisher.flush()

        try:
            benchmark(run)
        finally:
            bundle.disconnect()

    def test_inmem_1_sub_1000_keys(self, benchmark) -> None:
        bundle = _make_inmem_bundle(1)
        keys = [("data", f"key{i}") for i in range(1000)]

        def run():
            bundle.publisher.notify(keys)
            bundle.publisher.flush()

        try:
            benchmark(run)
        finally:
            bundle.disconnect()

    def test_redis_0_subs_100_keys(self, benchmark) -> None:
        publisher, observer = _make_redis_bundle(0)
        keys = [("data", f"key{i}") for i in range(100)]

        def run():
            publisher.notify(keys)
            publisher.flush()

        try:
            benchmark(run)
        finally:
            publisher.disconnect()
            observer.disconnect()

    def test_redis_0_subs_1000_keys(self, benchmark) -> None:
        publisher, observer = _make_redis_bundle(0)
        keys = [("data", f"key{i}") for i in range(1000)]

        def run():
            publisher.notify(keys)
            publisher.flush()

        try:
            benchmark(run)
        finally:
            publisher.disconnect()
            observer.disconnect()

    def test_redis_1_sub_100_keys(self, benchmark) -> None:
        publisher, observer = _make_redis_bundle(1)
        keys = [("data", f"key{i}") for i in range(100)]

        def run():
            publisher.notify(keys)
            publisher.flush()

        try:
            benchmark(run)
        finally:
            publisher.disconnect()
            observer.disconnect()

    def test_redis_1_sub_1000_keys(self, benchmark) -> None:
        publisher, observer = _make_redis_bundle(1)
        keys = [("data", f"key{i}") for i in range(1000)]

        def run():
            publisher.notify(keys)
            publisher.flush()

        try:
            benchmark(run)
        finally:
            publisher.disconnect()
            observer.disconnect()
