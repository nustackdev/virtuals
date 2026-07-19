"""Fixtures for _backends tests. Mirrors tests/tkv/conftest.py."""

import os
import uuid

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "redis: Tests requiring redis-py + msgpack installed and a reachable "
        "redis server (TEST_REDIS_URL env, default redis://localhost:6380).",
    )


@pytest.fixture(scope="session")
def redis_url() -> str:
    """URL for the test redis server. Skips the test if unreachable or deps missing."""
    url = os.environ.get("TEST_REDIS_URL", "redis://localhost:6380")
    try:
        import msgpack  # noqa: F401
        import redis  # noqa: F401
    except ImportError as e:
        pytest.skip(f"redis extras not installed: {e}")
    try:
        import redis as _r

        _r.from_url(url).ping()
    except Exception as e:
        pytest.skip(f"redis unreachable at {url}: {e}")
    return url


@pytest.fixture
def unique_channel_prefix() -> str:
    """Per-test unique redis channel prefix so tests can't leak state into each other."""
    return f"vtest:{uuid.uuid4().hex[:8]}"


@pytest.fixture
def redis_cleanup(redis_url: str):
    """Best-effort teardown of any test-created redis keys/channels for a prefix."""
    import redis as _r

    client = _r.from_url(redis_url)
    prefixes_used: list[str] = []

    def register(prefix: str) -> None:
        prefixes_used.append(prefix)

    yield register

    for prefix in prefixes_used:
        try:
            for key in client.scan_iter(match=f"{prefix}:*"):
                client.delete(key)
        except Exception:  # noqa: S110
            pass
    try:
        client.close()
    except Exception:  # noqa: S110
        pass
