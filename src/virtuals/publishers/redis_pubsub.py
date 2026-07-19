"""Redis publisher (top-level shortcut)."""

from __future__ import annotations

from virtuals._backends.publishers.redis_pubsub import RedisPublisher


__all__ = [
    "RedisPublisher",
]
