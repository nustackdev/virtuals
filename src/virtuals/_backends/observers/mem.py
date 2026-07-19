"""In-memory observer.

Registers a listener on a shared InMemoryTransport. On every publish, the
transport calls back into `_dispatch_incoming(batch)` and the observer's
worker matches against its local SubscriptionRegistry.
"""

from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING

from ._base import DEFAULT_DISPATCH_QUEUE_MAXSIZE, ObserverBase


if TYPE_CHECKING:
    from virtuals.tkv.transport import InMemoryTransport


logger = getLogger(__name__)


__all__ = [
    "InMemoryObserver",
]


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

    def _on_connect(self) -> None:
        self._transport.register(self._dispatch_incoming)

    def _on_disconnect(self) -> None:
        self._transport.unregister(self._dispatch_incoming)
