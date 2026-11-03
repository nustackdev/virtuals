"""Redis publisher: server-side routing of keys to per-filter-hash channels.

Each drained batch goes to the `nu1_route` Lua function (FCALL, or EVALSHA on
servers without functions) in chunks of at most MAX_KEYS_PER_CALL keys.
The function probes the routing index the observers registered (`nu1_reg`),
groups keys per subscriber hash and PUBLISHes one message per hash. The
publisher keeps no registry cache, control listener or refresh timer, so a
subscription registered before a write is always routed.

Python owns encoding (`notify.keyenc`): Lua only compares and concatenates
opaque strings, and Python's Filter.matches stays the only exact matcher
(the observer drops over-delivery).

Delivery is at least once per retry: a chunk retried after a lost reply runs
PUBLISH again, so an observer can see a key twice (see PublisherBase).

Wire format (unchanged for observers):
- Channel: `{prefix}:notif:{filter_hash}`.
- msgpack payload: `{"iid": pub_id, "keys": [[...], ...]}`.

Mixing publishers/observers from before server-side routing on one prefix is
unsupported: the routing index lives under `{prefix}:nx1:*`.
"""

from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING, Any

from virtuals._backends.notify.keyenc import enc_key
from virtuals._backends.notify.router import NotifyLua

from ._base import DEFAULT_PUBLISH_RETRY_SECONDS, DEFAULT_QUEUE_MAXSIZE, PublisherBase


try:
    import redis
except ImportError as e:
    raise ImportError(
        "redis is required for RedisPublisher. Install with: pip install virtuals-py[redis]"
    ) from e

try:
    import msgpack
except ImportError as e:
    raise ImportError(
        "msgpack is required for RedisPublisher. Install with: pip install virtuals-py[redis]"
    ) from e


if TYPE_CHECKING:
    from virtuals.tkv.types import Key


logger = getLogger(__name__)


__all__ = [
    "MAX_KEYS_PER_CALL",
    "RedisPublisher",
]


# Keys per nu1_route call. Bounds one server-side script run (valkey is
# single threaded, a 250-key call holds it about 3 ms at most) and the
# argument count of one call.
MAX_KEYS_PER_CALL: int = 250


class RedisPublisher(PublisherBase):
    """Publisher that routes writes server-side onto per-filter-hash channels."""

    MAX_KEYS_PER_CALL = MAX_KEYS_PER_CALL

    def __init__(
        self,
        *,
        redis_url: str = "redis://localhost:6379",
        channel_prefix: str = "nu",
        queue_maxsize: int = DEFAULT_QUEUE_MAXSIZE,
        publish_retry_seconds: float = DEFAULT_PUBLISH_RETRY_SECONDS,
    ) -> None:
        super().__init__(queue_maxsize=queue_maxsize, publish_retry_seconds=publish_retry_seconds)
        self._redis_url = redis_url
        self._channel_prefix = channel_prefix
        # Instance id: kept in the wire payload for backwards-compat; the
        # RedisObserver ignores it.
        self._instance_id = id(self)
        # Constant payload head: msgpack map of 2, "iid" -> id, "keys" -> <array>.
        self._hdr: bytes = (
            b"\x82"
            + msgpack.packb("iid")
            + msgpack.packb(self._instance_id)
            + msgpack.packb("keys")
        )
        self._redis_pub: Any = None
        self._lua = NotifyLua()

    # -- Lifecycle hooks ------------------------------------------------------

    def _on_connect(self) -> None:
        """Open the Redis connection and load the routing library."""
        self._redis_pub = redis.from_url(self._redis_url)
        self._redis_pub.ping()
        self._lua.ensure(self._redis_pub)
        logger.info(
            "Redis publisher started at %s (prefix=%s, iid=%d, mode=%s)",
            self._redis_url,
            self._channel_prefix,
            self._instance_id,
            self._lua.mode,
        )

    def _on_disconnect(self) -> None:
        """Close the Redis connection."""
        if self._redis_pub is not None:
            try:
                self._redis_pub.close()
            except Exception as e:
                logger.warning("Error closing redis connection: %s", e)
            self._redis_pub = None
        logger.info("Redis publisher stopped")

    # -- Publish routing ------------------------------------------------------

    def _route_args(self, batch: list[Key]) -> tuple[list[bytes], int]:
        """Build nu1_route ARGV: P, hdr, then per key packed, nseg, segs.

        Returns the args and the number of routable keys in them.
        """
        args: list[bytes] = [self._channel_prefix.encode(), self._hdr]
        n = 0
        for key in batch:
            try:
                segs = enc_key(key)
                packed = msgpack.packb(list(key), use_bin_type=True)
            except (TypeError, ValueError, OverflowError) as e:
                # Not a str/int key, an int outside msgpack's range, or a str
                # with lone surrogates. Logged, counted, raised by next flush().
                logger.error("Skipping unroutable key %r: %s", key, e)
                self._record_drop(1, e)
                continue
            args.append(packed)
            args.append(b"%d" % len(segs))
            args.extend(segs)
            n += 1
        return args, n

    def _prepare_chunk(self, chunk: list[Key]) -> tuple[list[bytes], int]:
        """Encode once per chunk; unroutable keys are recorded here, not per retry."""
        return self._route_args(chunk)

    def _send_chunk(self, prepared: list[bytes]) -> None:
        """One nu1_route call for a chunk of at most MAX_KEYS_PER_CALL keys."""
        if self._redis_pub is None:
            raise redis.ConnectionError("RedisPublisher not connected")
        self._lua.call(self._redis_pub, "nu1_route", prepared)

    def _publish_batch(self, batch: list[Key]) -> None:
        """Encode and send one chunk (no retry)."""
        args, n = self._route_args(batch)
        if n:
            self._send_chunk(args)


if TYPE_CHECKING:
    from virtuals.tkv.publisher import PublisherProtocol

    _p: type[PublisherProtocol] = RedisPublisher
