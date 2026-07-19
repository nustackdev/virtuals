"""Publisher protocol.

Defines the write-side of change notifications: storage hands modified
keys to publisher.notify(keys), publisher takes ownership and routes them
onto its transport. Publishers do not know about local subscribers or
callbacks -- that is entirely the observer's concern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable


if TYPE_CHECKING:
    from collections.abc import Iterable
    from types import TracebackType

    from ..types import Key


__all__ = [
    "PublisherProtocol",
]


@runtime_checkable
class PublisherProtocol(Protocol):
    """Protocol for the write-side of change notifications.

    A publisher is attached to a storage. On every storage write, storage
    hands the modified keys to publisher.notify(keys). The publisher takes
    ownership from that instant -- it decides when and how to route the
    keys onto the transport (Redis pubsub, in-mem bus, etc.).

    Publishers know NOTHING about local subscriptions. They never call
    callbacks. They never touch a SubscriptionRegistry. That is the
    observer's job, on the other side of the transport.
    """

    def connect(self) -> None:
        """Connect publisher: open transport, start worker thread."""
        ...

    def disconnect(self) -> None:
        """Disconnect publisher: stop worker, close transport."""
        ...

    def notify(self, keys: Key | Iterable[Key]) -> None:
        """Enqueue keys for publish. Fire-and-forget, returns immediately."""
        ...

    def flush(self, timeout: float = 1.0) -> None:
        """Wait for the write queue to drain."""
        ...

    def __enter__(self) -> Self:
        """Enter context; connect."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context; disconnect."""
        ...
