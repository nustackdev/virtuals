"""Publisher backends.

Import directly from submodules:
    from virtuals._backends.publishers.mem import InMemoryPublisher
    from virtuals._backends.publishers.redis_pubsub import RedisPublisher
"""

from __future__ import annotations

from ._base import PublisherBase


__all__ = ["PublisherBase"]
