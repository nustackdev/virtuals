"""Compliance tests for TextDB storage implementation.

This test file inherits from tkv's StorageProtocolCompliance suite
to verify that TextStorage correctly implements the StorageProtocol interface.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from virtuals._backends.storages.textdb import TextStorage
from virtuals.codecs import TextCodec
from virtuals.testing import StorageProtocolCompliance


class TestTextDBCompliance(StorageProtocolCompliance):
    """Compliance test suite for TextDB storage."""

    @pytest.fixture
    def storage(self):
        """Provide TextDB storage for compliance testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            codec = TextCodec()
            storage = TextStorage(
                path=Path(tmpdir) / "test_storage",
                codec=codec,
            )
            storage.open()
            yield storage
            storage.close()
