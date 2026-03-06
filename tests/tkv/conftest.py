"""Shared test configuration and fixtures for all test types."""

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests - fast, isolated")
    config.addinivalue_line("markers", "functional: Functional end-to-end tests")
    config.addinivalue_line("markers", "textdb: Text storage tests")
    config.addinivalue_line("markers", "inmemdb: In-memory storage tests")
    config.addinivalue_line("markers", "tupkey: Tuple codec tests")
    config.addinivalue_line("markers", "codec: Codec-specific tests")
    config.addinivalue_line("markers", "slow: Slow-running tests")
