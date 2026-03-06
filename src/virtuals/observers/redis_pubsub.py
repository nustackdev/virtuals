"""Redis pub/sub observer implementation for inter-process notification.

The RedisObserver provides distributed notification across processes using Redis
pub/sub. When a key changes in one process, all other processes subscribed via
Redis will receive the notification.

Features:
- Inter-process notification via Redis pub/sub
- Thread-safe subscription management (inherited from BaseObserver)
- Automatic reconnection on connection loss
- Configurable channel prefix
"""

from __future__ import annotations

from virtuals._backends.observers.redis_pubsub import RedisObserver


__all__ = [
    "RedisObserver",
]
