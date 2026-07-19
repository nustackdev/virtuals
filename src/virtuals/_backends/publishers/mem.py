"""In-memory publisher.

Sends batches onto a shared InMemoryTransport. The transport fans out to
every registered observer listener synchronously on this publisher's
worker thread.
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
    ) -> None:
        super().__init__(queue_maxsize=queue_maxsize)
        self._transport = transport

    def _publish_batch(self, batch: list[Key]) -> None:
        self._transport.publish(batch)


if TYPE_CHECKING:
    from virtuals.tkv.publisher import PublisherProtocol

    _p: type[PublisherProtocol] = InMemoryPublisher
