"""Compliance tests for RocksDB storage implementation.

This test file inherits from tkv's StorageProtocolCompliance suite
to verify that RocksDBStorage correctly implements the StorageProtocol interface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from tkv._storages.rocksdb import RocksDBStorage
from tkv.codecs import BinaryCodec
from tkv.testing import StorageProtocolCompliance


if TYPE_CHECKING:
    from pathlib import Path


class TestRocksDBCompliance(StorageProtocolCompliance):
    """Compliance test suite for RocksDB storage."""

    @pytest.fixture
    def storage(self, tmp_path: Path):
        """Provide RocksDB storage for compliance testing."""
        codec = BinaryCodec()
        db_path = tmp_path / "test.db"
        storage = RocksDBStorage(path=db_path, codec=codec)
        storage.open()
        yield storage
        storage.close()
