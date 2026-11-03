"""Loader and call wrapper for the `nu_notify_v1` Lua library.

Function mode (Redis >= 7, Valkey): `FUNCTION LOAD REPLACE`, then `FCALL`.
Script mode (servers without functions): `SCRIPT LOAD` of the same body with
an EVALSHA dispatch footer, then `EVALSHA sha 0 <name> <args...>`.

A call that hits "Function not found" or "NOSCRIPT" (server restart,
`FUNCTION FLUSH`, `SCRIPT FLUSH`) reloads the library once and retries once.
"""

from __future__ import annotations

import threading
from importlib.resources import files
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from redis import Redis


__all__ = ["LIBRARY", "LUA_SCRIPT_SOURCE", "LUA_SOURCE", "NotifyLua"]

LIBRARY = "nu_notify_v1"

LUA_SOURCE: str = (files(__package__) / "nu_notify.lua").read_text(encoding="utf-8")

_REGISTER_MARKER = "-- @@register"
_SCRIPT_FOOTER = (
    "local H = {nu1_reg = reg, nu1_route = route}\n"
    "local fn = H[ARGV[1]]\n"
    "if not fn then return redis.error_reply('ERR Function not found: ' .. tostring(ARGV[1])) end\n"
    "return fn(ARGV, 2)\n"
)


def _script_source(src: str) -> str:
    body = src.split(_REGISTER_MARKER, 1)[0]
    lines = body.splitlines(keepends=True)
    if lines and lines[0].startswith("#!"):
        lines = lines[1:]
    return "".join(lines) + _SCRIPT_FOOTER


LUA_SCRIPT_SOURCE: str = _script_source(LUA_SOURCE)


def _is_missing(err: Exception) -> bool:
    import redis

    # redis-py maps NOSCRIPT to NoScriptError and strips the code from the message.
    if isinstance(err, redis.exceptions.NoScriptError):
        return True
    msg = str(err)
    return "Function not found" in msg or "NOSCRIPT" in msg


class NotifyLua:
    """Loads the routing library on a connection and calls its functions.

    One instance per connection owner. `ensure` picks function mode, or
    falls back to script mode when the server has no FUNCTION command.
    """

    def __init__(self) -> None:
        self.mode: str | None = None  # "function" | "script"
        self.sha: str | None = None
        self._lock = threading.Lock()

    def ensure(self, conn: Redis) -> None:
        """Load (or reload) the library on the server behind `conn`."""
        import redis

        with self._lock:
            if self.mode != "script":
                try:
                    conn.function_load(LUA_SOURCE, replace=True)
                    self.mode = "function"
                    return
                except redis.ResponseError as e:
                    if "unknown command" not in str(e).lower():
                        raise
            self.sha = conn.script_load(LUA_SCRIPT_SOURCE)
            self.mode = "script"

    def call(self, conn: Redis, name: str, args: list[Any]) -> object:
        """Call library function `name` with flat `args` (no keys).

        Reloads the library and retries once if the server lost it.
        """
        import redis

        if self.mode is None:
            self.ensure(conn)
        try:
            return self._call(conn, name, args)
        except redis.ResponseError as e:
            if not _is_missing(e):
                raise
        self.ensure(conn)
        return self._call(conn, name, args)

    def _call(self, conn: Redis, name: str, args: list[Any]) -> object:
        if self.mode == "script":
            return conn.evalsha(self.sha, 0, name, *args)
        return conn.fcall(name, 0, *args)
