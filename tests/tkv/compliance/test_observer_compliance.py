"""Compliance tests for tkv publisher/observer bundles.

Runs the ObserverCompliance suite against every storage backend that
supports publishers (InMemory, RocksDB, LMDB, TextDB). The publisher/
observer implementations under test can be the in-mem pair (default)
or the Redis pair. This proves the notify/flush contract holds
uniformly across storage backends.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from virtuals._backends.observers.mem import InMemoryObserver
from virtuals._backends.publishers.mem import InMemoryPublisher
from virtuals._backends.storages.lmdb import LMDBStorage
from virtuals._backends.storages.mem import InMemoryStorage
from virtuals._backends.storages.rocksdb import RocksDBStorage
from virtuals._backends.storages.textdb import TextStorage
from virtuals.codecs import BinaryCodec, NoOpCodec, TextCodec
from virtuals.testing import ObservableBundle, ObserverCompliance, RegistryCompliance
from virtuals.tkv.transport import InMemoryTransport


if TYPE_CHECKING:
    pass


class TestInMemoryObserverRegistryCompliance(RegistryCompliance):
    """Run registry compliance tests (uses default registry from tkv)."""

    pass


class TestInMemDBObserverCompliance(ObserverCompliance):
    """ObserverProtocol contract against InMemory storage."""

    @pytest.fixture
    def bundle(self):
        codec = NoOpCodec()
        transport = InMemoryTransport()
        publisher = InMemoryPublisher(transport)
        publisher.connect()
        observer = InMemoryObserver(transport)
        observer.connect()
        storage = InMemoryStorage(codec=codec, publisher=publisher)
        storage.open()
        yield ObservableBundle(storage=storage, publisher=publisher, observer=observer)
        storage.close()
        observer.disconnect()
        publisher.disconnect()


class TestRocksDBObserverCompliance(ObserverCompliance):
    """ObserverProtocol contract against RocksDB storage."""

    @pytest.fixture
    def bundle(self, tmp_path: Path):
        codec = BinaryCodec()
        transport = InMemoryTransport()
        publisher = InMemoryPublisher(transport)
        publisher.connect()
        observer = InMemoryObserver(transport)
        observer.connect()
        storage = RocksDBStorage(path=tmp_path / "test.db", codec=codec, publisher=publisher)
        storage.open()
        yield ObservableBundle(storage=storage, publisher=publisher, observer=observer)
        storage.close()
        observer.disconnect()
        publisher.disconnect()


class TestLMDBObserverCompliance(ObserverCompliance):
    """ObserverProtocol contract against LMDB storage."""

    @pytest.fixture
    def bundle(self, tmp_path: Path):
        codec = BinaryCodec()
        transport = InMemoryTransport()
        publisher = InMemoryPublisher(transport)
        publisher.connect()
        observer = InMemoryObserver(transport)
        observer.connect()
        storage = LMDBStorage(
            path=tmp_path / "test.lmdb",
            codec=codec,
            map_size=64 * 1024 * 1024,
            publisher=publisher,
        )
        storage.open()
        yield ObservableBundle(storage=storage, publisher=publisher, observer=observer)
        storage.close()
        observer.disconnect()
        publisher.disconnect()


class TestTextDBObserverCompliance(ObserverCompliance):
    """ObserverProtocol contract against TextDB storage."""

    @pytest.fixture
    def bundle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            codec = TextCodec()
            transport = InMemoryTransport()
            publisher = InMemoryPublisher(transport)
            publisher.connect()
            observer = InMemoryObserver(transport)
            observer.connect()
            storage = TextStorage(
                path=Path(tmpdir) / "test_storage",
                codec=codec,
                publisher=publisher,
            )
            storage.open()
            yield ObservableBundle(storage=storage, publisher=publisher, observer=observer)
            storage.close()
            observer.disconnect()
            publisher.disconnect()


@pytest.mark.redis
class TestRedisObserverRocksDBCompliance(ObserverCompliance):
    """ObserverProtocol contract with Redis publisher+observer wired into RocksDB."""

    @pytest.fixture
    def bundle(self, tmp_path: Path, redis_url, unique_channel_prefix, redis_cleanup):
        from virtuals._backends.observers.redis_pubsub import RedisObserver
        from virtuals._backends.publishers.redis_pubsub import RedisPublisher

        redis_cleanup(unique_channel_prefix)
        codec = BinaryCodec()
        observer = RedisObserver(redis_url=redis_url, channel_prefix=unique_channel_prefix)
        observer.connect()
        publisher = RedisPublisher(redis_url=redis_url, channel_prefix=unique_channel_prefix)
        publisher.connect()
        storage = RocksDBStorage(path=tmp_path / "test.db", codec=codec, publisher=publisher)
        storage.open()
        yield ObservableBundle(storage=storage, publisher=publisher, observer=observer)
        storage.close()
        publisher.disconnect()
        observer.disconnect()


@pytest.mark.redis
class TestRedisObserverLMDBCompliance(ObserverCompliance):
    """ObserverProtocol contract with Redis publisher+observer wired into LMDB."""

    @pytest.fixture
    def bundle(self, tmp_path: Path, redis_url, unique_channel_prefix, redis_cleanup):
        from virtuals._backends.observers.redis_pubsub import RedisObserver
        from virtuals._backends.publishers.redis_pubsub import RedisPublisher

        redis_cleanup(unique_channel_prefix)
        codec = BinaryCodec()
        observer = RedisObserver(redis_url=redis_url, channel_prefix=unique_channel_prefix)
        observer.connect()
        publisher = RedisPublisher(redis_url=redis_url, channel_prefix=unique_channel_prefix)
        publisher.connect()
        storage = LMDBStorage(
            path=tmp_path / "test.lmdb",
            codec=codec,
            map_size=64 * 1024 * 1024,
            publisher=publisher,
        )
        storage.open()
        yield ObservableBundle(storage=storage, publisher=publisher, observer=observer)
        storage.close()
        publisher.disconnect()
        observer.disconnect()
