"""Shared test configuration and fixtures for all test types."""

import os
import uuid

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
        import redis  # noqa: F401
        import msgpack  # noqa: F401
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
    """Best-effort teardown of any test-created redis keys/channels for a prefix.

    Usage: request `redis_cleanup` and `unique_channel_prefix`, no cleanup call
    needed -- the fixture wipes matching keys on exit.
    """
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
        except Exception:
            pass
    try:
        client.close()
    except Exception:
        pass
