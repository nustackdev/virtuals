"""Shared test configuration and fixtures for all test types."""

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers (the `slow` tier is registered in pyproject.toml)."""
