"""In-memory storage backend.

Ephemeral key-value storage for tests, prototypes and per-process state.
Serializable transactions through a writer slot, snapshots pinned at a
commit sequence (repeatable, never torn), ordered keys with native seek,
and notifications after commit. Single-process; all data is lost on close.
"""

from __future__ import annotations

from virtuals._backends.storages.mem import InMemoryStorage


__all__ = [
    "InMemoryStorage",
]
