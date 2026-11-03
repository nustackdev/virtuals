"""Private valkey / redis servers for tests.

- `private_valkey(tmp_dir, port=None)`: an embedded valkeylite server on a
  unix socket (or a TCP port when given). `.stop()` and `.start()` restart it
  at the same address, for reconnect tests.
- `private_redis6(tmp_dir)`: the system `redis-server` binary (red has 6.0.16,
  no FUNCTION command) on a private unix socket, for the EVALSHA fallback.
  Returns None when the binary is absent.
- `redis_url_or_private(tmp_path_factory)`: `TEST_REDIS_URL` if set, else a
  private valkey. Generator for session fixtures; skips when neither works.

Never point tests at a shared server by default: it silently skips or
collides with other users.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest


if TYPE_CHECKING:
    from collections.abc import Iterator


__all__ = [
    "PrivateRedis",
    "PrivateValkey",
    "private_redis6",
    "private_valkey",
    "redis_url_or_private",
]


def _sock_dir() -> Path:
    # unix socket paths are capped at ~107 bytes; pytest tmp dirs can be long.
    return Path(tempfile.mkdtemp(prefix="vk-", dir="/tmp"))


def _ping(url: str, timeout: float = 5.0) -> None:
    import redis

    deadline = time.monotonic() + timeout
    while True:
        try:
            redis.from_url(url, socket_connect_timeout=0.5).ping()
            return
        except Exception:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.05)


class PrivateValkey:
    """valkeylite server with a stable address across stop/start."""

    def __init__(self, tmp_dir: Path, port: int | None = None) -> None:
        from valkeylite import ValkeyServer

        self._sock_dir = None if port else _sock_dir()
        sock = None if port else self._sock_dir / "vk.sock"
        self._server: Any = ValkeyServer(
            port=port if port else 0,
            data_dir=Path(tmp_dir),
            unix_socket_path=sock,
            save="",
            appendonly="no",
        )
        self._server.start()
        self.port = self._server.port if port else None
        self.url = f"unix://{sock}" if sock else f"redis://127.0.0.1:{self.port}"

    def stop(self) -> None:
        self._server.stop()

    def start(self) -> None:
        if self.port:
            self._server._desired_port = self.port
        self._server.start()

    def close(self) -> None:
        if self._server.is_running():
            self._server.stop()
        if self._sock_dir is not None:
            shutil.rmtree(self._sock_dir, ignore_errors=True)


def private_valkey(tmp_dir: Path, port: int | None = None) -> PrivateValkey:
    """Start a private valkey. Call `.close()` when done."""
    return PrivateValkey(tmp_dir, port=port)


class PrivateRedis:
    """System redis-server on a private unix socket."""

    def __init__(self, binary: str) -> None:
        self._dir = _sock_dir()
        sock = self._dir / "r.sock"
        self._proc = subprocess.Popen(  # noqa: S603
            [
                binary,
                "--port",
                "0",
                "--unixsocket",
                str(sock),
                "--unixsocketperm",
                "700",
                "--save",
                "",
                "--appendonly",
                "no",
                "--dir",
                str(self._dir),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.url = f"unix://{sock}"
        _ping(self.url)

    def close(self) -> None:
        self._proc.terminate()
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        shutil.rmtree(self._dir, ignore_errors=True)


def private_redis6(binary: str = "/usr/bin/redis-server") -> PrivateRedis | None:
    """Start the system redis-server privately, or None if it is absent."""
    if not Path(binary).exists():
        return None
    return PrivateRedis(binary)


def redis_url_or_private(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """Yield TEST_REDIS_URL if set, else a private valkey's url (stopped after)."""
    try:
        import msgpack  # noqa: F401
        import redis
    except ImportError as e:
        pytest.skip(f"redis extras not installed: {e}")

    url = os.environ.get("TEST_REDIS_URL")
    if url:
        try:
            redis.from_url(url).ping()
        except Exception as e:
            pytest.skip(f"redis unreachable at {url}: {e}")
        yield url
        return

    try:
        server = private_valkey(tmp_path_factory.mktemp("valkey"))
    except ImportError as e:
        pytest.skip(f"valkeylite not installed and TEST_REDIS_URL unset: {e}")
    try:
        yield server.url
    finally:
        server.close()
