"""Shared test configuration and fixtures for all test types."""

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: Slow-running tests")
