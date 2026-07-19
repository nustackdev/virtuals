"""Redis pub/sub observer (top-level shortcut).

Re-exports the read-side observer that listens on per-filter-hash Redis
pubsub channels. Pair it with `virtuals.publishers.redis_pubsub.RedisPublisher`
(possibly in a different process) for a working write/notify/subscribe
loop.
"""

from __future__ import annotations

from virtuals._backends.observers.redis_pubsub import RedisObserver


__all__ = [
    "RedisObserver",
]
