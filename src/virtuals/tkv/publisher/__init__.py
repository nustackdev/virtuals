"""Publisher module.

Write-side of change notifications. A publisher is attached to a storage
and forwards keys onto a transport (in-mem bus or Redis pubsub). It knows
nothing about local subscribers -- that is the observer's role, on the
other side of the transport.
"""

from __future__ import annotations

from .exceptions import (
    PublisherConnectionError,
    PublisherError,
    PublisherOperationError,
)
from .publisher import PublisherProtocol


__all__ = [
    "PublisherConnectionError",
    "PublisherError",
    "PublisherOperationError",
    "PublisherProtocol",
]
