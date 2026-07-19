"""Compliance tests for tkv observer implementations.

Runs the ObserverCompliance suite against every storage backend that
supports observers (InMemory, RocksDB, LMDB, TextDB). The observer
implementation under test is always `InMemoryObserver` -- what varies
is the storage backend it's wired into. This proves the notify/flush
contract holds uniformly across storage backends.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from virtuals._backends.observers.mem import InMemoryObserver
from virtuals._backends.storages.lmdb import LMDBStorage
from virtuals._backends.storages.mem import InMemoryStorage
from virtuals._backends.storages.rocksdb import RocksDBStorage
from virtuals._backends.storages.textdb import TextStorage
from virtuals.codecs import BinaryCodec, NoOpCodec, TextCodec
from virtuals.testing import ObserverCompliance, RegistryCompliance


if TYPE_CHECKING:
    pass


class TestInMemoryObserverRegistryCompliance(RegistryCompliance):
    """Run registry compliance tests (uses default registry from tkv)."""

    pass


class TestInMemDBObserverCompliance(ObserverCompliance):
    """ObserverProtocol contract against InMemory storage."""

    @pytest.fixture
    def observable_storage(self):
        codec = NoOpCodec()
        observer = InMemoryObserver(codec=codec)
        observer.connect()
        storage = InMemoryStorage(codec=codec, observer=observer)
        storage.open()
        yield storage
        storage.close()
        observer.disconnect()


class TestRocksDBObserverCompliance(ObserverCompliance):
    """ObserverProtocol contract against RocksDB storage."""

    @pytest.fixture
    def observable_storage(self, tmp_path: Path):
        codec = BinaryCodec()
        observer = InMemoryObserver(codec=codec)
        observer.connect()
        storage = RocksDBStorage(path=tmp_path / "test.db", codec=codec, observer=observer)
        storage.open()
        yield storage
        storage.close()
        observer.disconnect()


class TestLMDBObserverCompliance(ObserverCompliance):
    """ObserverProtocol contract against LMDB storage."""

    @pytest.fixture
    def observable_storage(self, tmp_path: Path):
        codec = BinaryCodec()
        observer = InMemoryObserver(codec=codec)
        observer.connect()
        storage = LMDBStorage(
            path=tmp_path / "test.lmdb",
            codec=codec,
            map_size=64 * 1024 * 1024,
            observer=observer,
        )
        storage.open()
        yield storage
        storage.close()
        observer.disconnect()


class TestTextDBObserverCompliance(ObserverCompliance):
    """ObserverProtocol contract against TextDB storage."""

    @pytest.fixture
    def observable_storage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            codec = TextCodec()
            observer = InMemoryObserver(codec=codec)
            observer.connect()
            storage = TextStorage(
                path=Path(tmpdir) / "test_storage",
                codec=codec,
                observer=observer,
            )
            storage.open()
            yield storage
            storage.close()
            observer.disconnect()


@pytest.mark.redis
class TestRedisObserverRocksDBCompliance(ObserverCompliance):
    """ObserverProtocol contract with RedisObserver wired into RocksDB."""

    @pytest.fixture
    def observable_storage(self, tmp_path: Path, redis_url, unique_channel_prefix, redis_cleanup):
        from virtuals._backends.observers.redis_pubsub import RedisObserver

        redis_cleanup(unique_channel_prefix)
        codec = BinaryCodec()
        observer = RedisObserver(
            codec=codec, redis_url=redis_url, channel_prefix=unique_channel_prefix
        )
        observer.connect()
        storage = RocksDBStorage(path=tmp_path / "test.db", codec=codec, observer=observer)
        storage.open()
        yield storage
        storage.close()
        observer.disconnect()


@pytest.mark.redis
class TestRedisObserverLMDBCompliance(ObserverCompliance):
    """ObserverProtocol contract with RedisObserver wired into LMDB."""

    @pytest.fixture
    def observable_storage(self, tmp_path: Path, redis_url, unique_channel_prefix, redis_cleanup):
        from virtuals._backends.observers.redis_pubsub import RedisObserver

        redis_cleanup(unique_channel_prefix)
        codec = BinaryCodec()
        observer = RedisObserver(
            codec=codec, redis_url=redis_url, channel_prefix=unique_channel_prefix
        )
        observer.connect()
        storage = LMDBStorage(
            path=tmp_path / "test.lmdb",
            codec=codec,
            map_size=64 * 1024 * 1024,
            observer=observer,
        )
        storage.open()
        yield storage
        storage.close()
        observer.disconnect()
