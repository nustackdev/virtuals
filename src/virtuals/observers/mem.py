"""In-memory observer (top-level shortcut).

Re-exports the read-side observer that listens on a shared
`InMemoryTransport`. Pair it with `virtuals.publishers.mem.InMemoryPublisher`
on the same transport for a working write/notify/subscribe loop.
"""

from __future__ import annotations

from virtuals._backends.observers.mem import InMemoryObserver


__all__ = [
    "InMemoryObserver",
]
