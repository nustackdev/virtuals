"""In-process transport for the in-mem publisher/observer pair.

Publisher.notify -> Publisher._publish_batch -> transport.publish(keys)
-> all registered observer listeners -> Observer._dispatch_incoming.

Design choice: explicit shared object passed to both Publisher and
Observer. Not a module-level singleton -- tests and multi-tenant callers
need isolated (publisher, observer) pairs without cross-contamination.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from ..types import Key


__all__ = [
    "InMemoryTransport",
]


class InMemoryTransport:
    """Minimal in-process pub/sub bus for the mem backend.

    Thread-safe register/unregister/publish. All listeners are invoked
    synchronously on the publish call's thread.
    """

    def __init__(self) -> None:
        """Initialize with no listeners."""
        self._listeners: list[Callable[[list[Key]], None]] = []
        self._lock = threading.Lock()

    def register(self, listener: Callable[[list[Key]], None]) -> None:
        """Add a listener. Duplicates are permitted (kept separate)."""
        with self._lock:
            self._listeners.append(listener)

    def unregister(self, listener: Callable[[list[Key]], None]) -> None:
        """Remove one occurrence of listener. No-op if absent."""
        with self._lock:
            try:
                self._listeners.remove(listener)
            except ValueError:
                pass

    def publish(self, keys: Iterable[Key]) -> None:
        """Fan out keys to every registered listener.

        Empty batch is a no-op. Snapshot the listener list under the lock
        so a concurrent unregister mid-fanout is safe.
        """
        batch = list(keys)
        if not batch:
            return
        with self._lock:
            listeners = list(self._listeners)
        for listener in listeners:
            listener(batch)
