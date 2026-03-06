"""Compliance tests for tkv codec implementations.

This file runs the KeyCodecCompliance and ValueCodecCompliance tests against
all tkv codecs to verify they correctly implement the codec protocols.
"""

from __future__ import annotations

import pytest

from virtuals._backends.key_codecs import BinaryKeyCodec, PyBinaryKeyCodec, StringKeyCodec
from virtuals.codecs.json import JSONCodec
from virtuals.codecs.passthrough import PassthroughCodec
from virtuals.codecs.pickle import PickleCodec
from virtuals.testing import KeyCodecCompliance, ValueCodecCompliance


# =============================================================================
# Key Codec Compliance Tests
# =============================================================================


class TestBinaryKeyCodecCompliance(KeyCodecCompliance):
    """Compliance tests for BinaryKeyCodec (Cython-based)."""

    @pytest.fixture
    def codec(self):
        """Provide BinaryKeyCodec for testing."""
        return BinaryKeyCodec()


class TestPyBinaryKeyCodecCompliance(KeyCodecCompliance):
    """Compliance tests for PyBinaryKeyCodec (pure Python)."""

    @pytest.fixture
    def codec(self):
        """Provide PyBinaryKeyCodec for testing."""
        return PyBinaryKeyCodec()


class TestStringKeyCodecCompliance(KeyCodecCompliance):
    """Compliance tests for StringKeyCodec."""

    @pytest.fixture
    def codec(self):
        """Provide StringKeyCodec for testing."""
        return StringKeyCodec()


# =============================================================================
# Value Codec Compliance Tests
# =============================================================================


class TestJSONCodecCompliance(ValueCodecCompliance):
    """Compliance tests for JSONCodec."""

    @pytest.fixture
    def value_codec(self):
        """Provide JSONCodec for testing."""
        return JSONCodec()


class TestPickleCodecCompliance(ValueCodecCompliance):
    """Compliance tests for PickleCodec."""

    @pytest.fixture
    def value_codec(self):
        """Provide PickleCodec for testing."""
        return PickleCodec()


class TestPassthroughCodecCompliance(ValueCodecCompliance):
    """Compliance tests for PassthroughCodec.

    Note: PassthroughCodec doesn't serialize, so it only works with
    values that can be compared directly (no bytes roundtrip through string).
    We skip the bytes test since passthrough doesn't transform values.
    """

    @pytest.fixture
    def value_codec(self):
        """Provide PassthroughCodec for testing."""
        return PassthroughCodec()


try:
    from virtuals.codecs.msgpack import MessagePackCodec

    class TestMessagePackCodecCompliance(ValueCodecCompliance):
        """Compliance tests for MessagePackCodec."""

        @pytest.fixture
        def value_codec(self):
            """Provide MessagePackCodec for testing."""
            return MessagePackCodec()

except ImportError:
    pass  # msgpack not installed, skip tests
