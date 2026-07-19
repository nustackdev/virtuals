"""Read-side Observer backends.

Import directly from submodules:
    from virtuals._backends.observers.mem import InMemoryObserver
    from virtuals._backends.observers.redis_pubsub import RedisObserver
"""

from __future__ import annotations

from ._base import ObserverBase
from .mem import InMemoryObserver
from .redis_pubsub import RedisObserver


__all__ = [
    "InMemoryObserver",
    "ObserverBase",
    "RedisObserver",
]
