"""Transport module.

The shared meeting point between a publisher and observer. For the in-mem
backend, transport is an explicit shared object (InMemoryTransport). For
the Redis backend, transport IS Redis -- no in-process class needed.
"""

from __future__ import annotations

from .transport import InMemoryTransport


__all__ = [
    "InMemoryTransport",
]
