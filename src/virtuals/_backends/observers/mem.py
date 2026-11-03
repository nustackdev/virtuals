"""In-memory observer.

Registers a listener on a shared InMemoryTransport. On every publish, the
transport calls back into `_dispatch_incoming(batch)` and the observer's
worker matches against its local SubscriptionRegistry.

The transport outlives observers (it is the shared bus), so the listener
it holds references the observer weakly: an observer dropped without
disconnect() is collected, and the transport's call becomes a no-op.
"""

from __future__ import annotations

import weakref
from logging import getLogger
from typing import TYPE_CHECKING

from ._base import DEFAULT_DISPATCH_QUEUE_MAXSIZE, ObserverBase


if TYPE_CHECKING:
    from collections.abc import Callable

    from virtuals.tkv.transport import InMemoryTransport
    from virtuals.tkv.types import Key


logger = getLogger(__name__)


__all__ = [
    "InMemoryObserver",
]


def _weak_listener(ref: weakref.ref[InMemoryObserver]) -> Callable[[list[Key]], None]:
    def listener(batch: list[Key]) -> None:
        obs = ref()
        if obs is not None:
            obs._dispatch_incoming(batch)

    return listener


def _weak_resync(ref: weakref.ref[InMemoryObserver]) -> Callable[[], None]:
    def on_resync() -> None:
        obs = ref()
        if obs is not None:
            obs._dispatch_resync_all()

    return on_resync


class InMemoryObserver(ObserverBase):
    """Observer that receives batches from a shared InMemoryTransport."""

    def __init__(
        self,
        transport: InMemoryTransport,
        *,
        dispatch_queue_maxsize: int = DEFAULT_DISPATCH_QUEUE_MAXSIZE,
    ) -> None:
        super().__init__(dispatch_queue_maxsize=dispatch_queue_maxsize)
        self._transport = transport
        self._listener: Callable[[list[Key]], None] | None = None
        self._on_resync: Callable[[], None] | None = None

    def _on_connect(self) -> None:
        ref = weakref.ref(self)
        self._listener = _weak_listener(ref)
        self._on_resync = _weak_resync(ref)
        self._transport.register(self._listener, self._on_resync)

    def _on_disconnect(self) -> None:
        if self._listener is not None:
            self._transport.unregister(self._listener, self._on_resync)
            self._listener = None
            self._on_resync = None
