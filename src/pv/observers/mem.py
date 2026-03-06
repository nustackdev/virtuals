"""In-memory observer implementation with thread-safe subscription management.

The InMemoryObserver provides efficient pattern matching using the
SubscriptionRegistry from the base class. All subscription logic is
handled by BaseObserver - this class only provides connection management.
"""

from __future__ import annotations

from tkv._observers.mem import InMemoryObserver


__all__ = [
    "InMemoryObserver",
]
