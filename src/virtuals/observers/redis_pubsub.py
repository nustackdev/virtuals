"""Redis pub/sub observer with fire-and-forget notifications.

RedisObserver = Observer + RedisPublisher.
"""

from __future__ import annotations

from virtuals._backends.observers.redis_pubsub import RedisObserver, RedisPublisher


__all__ = [
    "RedisObserver",
    "RedisPublisher",
]
