"""Compliance tests for SQLite storage implementation.

This test file inherits from tkv's StorageProtocolCompliance suite
to verify that SQLiteStorage correctly implements the StorageProtocol interface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from virtuals._backends.storages.sqlite import SQLiteStorage
from virtuals.codecs import BinaryCodec
from virtuals.testing import StorageProtocolCompliance


if TYPE_CHECKING:
    from pathlib import Path


class TestSQLiteCompliance(StorageProtocolCompliance):
    """Compliance test suite for SQLite storage."""

    @pytest.fixture
    def storage(self, tmp_path: Path):
        """Provide SQLite storage for compliance testing."""
        codec = BinaryCodec()
        storage = SQLiteStorage(path=tmp_path / "test.sqlite", codec=codec)
        storage.open()
        yield storage
        storage.close()
