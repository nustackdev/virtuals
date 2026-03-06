"""Compliance tests for InMemDB storage implementation.

This test file inherits from tkv's StorageProtocolCompliance suite
to verify that InMemoryStorage correctly implements the StorageProtocol interface.
"""

from __future__ import annotations

import pytest
from tkv._storages.mem import InMemoryStorage
from tkv.codecs import NoOpCodec
from tkv.testing import StorageProtocolCompliance


class TestInMemDBCompliance(StorageProtocolCompliance):
    """Compliance test suite for InMemDB storage."""

    @pytest.fixture
    def storage(self):
        """Provide InMemory storage for compliance testing."""
        codec = NoOpCodec()
        storage = InMemoryStorage(codec=codec)
        storage.open()
        yield storage
        storage.close()
