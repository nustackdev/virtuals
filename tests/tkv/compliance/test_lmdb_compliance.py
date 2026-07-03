"""Compliance tests for LMDB storage implementation.

This test file inherits from tkv's StorageProtocolCompliance suite
to verify that LMDBStorage correctly implements the StorageProtocol interface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from virtuals._backends.storages.lmdb import LMDBStorage
from virtuals.codecs import BinaryCodec
from virtuals.testing import StorageProtocolCompliance


if TYPE_CHECKING:
    from pathlib import Path


class TestLMDBCompliance(StorageProtocolCompliance):
    """Compliance test suite for LMDB storage."""

    @pytest.fixture
    def storage(self, tmp_path: Path):
        """Provide LMDB storage for compliance testing."""
        codec = BinaryCodec()
        db_path = tmp_path / "test.lmdb"
        storage = LMDBStorage(path=db_path, codec=codec, map_size=64 * 1024 * 1024)
        storage.open()
        yield storage
        storage.close()
