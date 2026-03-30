"""In-memory observer with fire-and-forget notifications.

InMemoryObserver = Observer + InMemoryPublisher.
"""

from __future__ import annotations

from virtuals._backends.observers.mem import InMemoryObserver, InMemoryPublisher


__all__ = [
    "InMemoryObserver",
    "InMemoryPublisher",
]
