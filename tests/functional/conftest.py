"""Functional test configuration and shared fixtures."""

from collections.abc import Generator

import pytest

from virtuals.codecs import NoOpCodec
from virtuals.navigator import Navigator
from virtuals.observers.mem import InMemoryObserver
from virtuals.publishers.mem import InMemoryPublisher
from virtuals.storages.mem import InMemoryStorage
from virtuals.tkv.storage import SnapshotProtocol, TransactionProtocol
from virtuals.tkv.transport import InMemoryTransport


@pytest.fixture
def storage() -> Generator[InMemoryStorage, None, None]:
    """Memory storage instance for functional tests.

    Provides a clean storage instance for each test with automatic cleanup.
    """
    transport = InMemoryTransport()
    publisher = InMemoryPublisher(transport)
    observer = InMemoryObserver(transport)
    storage = InMemoryStorage(
        codec=NoOpCodec(),
        publisher=publisher,
    )
    publisher.connect()
    observer.connect()
    storage.open()
    try:
        yield storage
    finally:
        storage.close()
        observer.disconnect()
        publisher.disconnect()


@pytest.fixture
def tx(storage: InMemoryStorage) -> Generator[TransactionProtocol, None, None]:
    """Read-write transaction context.

    Auto-commits on successful completion, rolls back on exception.
    """
    with storage.transaction() as transaction:
        yield transaction


@pytest.fixture
def snapshot(storage: InMemoryStorage) -> Generator[SnapshotProtocol, None, None]:
    """Read-only snapshot context.

    Useful for testing isolation and concurrent read scenarios.
    """
    with storage.snapshot() as snap:
        yield snap


@pytest.fixture
def nav(storage: InMemoryStorage) -> Navigator:
    """Navigator instance for testing."""
    return Navigator(storage)
