"""In-memory publisher.

Sends batches onto a shared InMemoryTransport. The transport fans out to
every registered observer listener synchronously on this publisher's
worker thread.

There is no stalled server to protect a writer against, so notify() keeps
pure backpressure by default (`notify_timeout=None`: it blocks until the
queue has room, never drops while connected; a notify while disconnected
is a counted drop like on any publisher). With a timeout set, a full queue
drops keys too: counted and logged, never delivered.
"""

from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING

from ._base import DEFAULT_QUEUE_MAXSIZE, PublisherBase


if TYPE_CHECKING:
    from virtuals.tkv.transport import InMemoryTransport
    from virtuals.tkv.types import Key


logger = getLogger(__name__)


__all__ = [
    "InMemoryPublisher",
]


class InMemoryPublisher(PublisherBase):
    """Publisher that hands batches to a shared in-process transport."""

    def __init__(
        self,
        transport: InMemoryTransport,
        *,
        queue_maxsize: int = DEFAULT_QUEUE_MAXSIZE,
        notify_timeout: float | None = None,
    ) -> None:
        super().__init__(queue_maxsize=queue_maxsize, notify_timeout=notify_timeout)
        self._transport = transport

    def _publish_batch(self, batch: list[Key]) -> None:
        self._transport.publish(batch)


if TYPE_CHECKING:
    from virtuals.tkv.publisher import PublisherProtocol

    _p: type[PublisherProtocol] = InMemoryPublisher
