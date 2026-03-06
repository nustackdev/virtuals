"""Compliance tests for tkv observer implementations.

This file runs the observer compliance tests against tkv storages
with observer support to verify they correctly implement the
ObserverProtocol interface.
"""

from __future__ import annotations

import pytest
from tkv._observers.mem import InMemoryObserver
from tkv._storages.mem import InMemoryStorage
from tkv.codecs import NoOpCodec
from tkv.testing import ObserverCompliance, RegistryCompliance


class TestInMemoryObserverRegistryCompliance(RegistryCompliance):
    """Run registry compliance tests (uses default registry from tkv)."""

    pass


class TestInMemDBObserverCompliance(ObserverCompliance):
    """Compliance tests for InMemDB with observer support."""

    @pytest.fixture
    def observable_storage(self):
        """Provide InMemory storage with observer for compliance testing."""
        codec = NoOpCodec()
        observer = InMemoryObserver(codec=codec)
        observer.connect()
        storage = InMemoryStorage(codec=codec, observer=observer)
        storage.open()
        yield storage
        storage.close()
        observer.disconnect()
