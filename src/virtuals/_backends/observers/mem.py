"""In-memory observer and publisher.

InMemoryPublisher delivers notifications by calling subscriber callbacks
directly on the observer's background thread.

InMemoryObserver is a convenience class: Observer + InMemoryPublisher.
"""

from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING, Any

from virtuals.tkv.observer.publisher import deliver_local

from ._base import Observer


if TYPE_CHECKING:
    from virtuals.tkv.codec import CodecProtocol
    from virtuals.tkv.observer import SubscriptionRegistry
    from virtuals.tkv.observer.publisher import PublisherProtocol
    from virtuals.tkv.observer.subscription import Subscription
    from virtuals.tkv.types import Key


logger = getLogger(__name__)


__all__ = [
    "InMemoryObserver",
    "InMemoryPublisher",
]


class InMemoryPublisher:
    """Publisher that delivers notifications via local callbacks.

    Calls subscriber callbacks directly. No external broadcast.
    """

    def start(self, registry: SubscriptionRegistry) -> None:
        """No-op. No external resources to start."""

    def stop(self) -> None:
        """No-op. No external resources to stop."""

    def deliver(
        self,
        keys: list[Key],
        notifications: list[tuple[Key, list[Subscription]]],
    ) -> None:
        """Deliver matched notifications to local subscribers."""
        deliver_local(notifications)


class InMemoryObserver(Observer[str]):
    """Observer with in-memory local delivery.

    Convenience class: Observer + InMemoryPublisher.
    """

    def __init__(self, codec: CodecProtocol[str, Any]) -> None:
        """Initialize with InMemoryPublisher."""
        super().__init__(codec=codec, publisher=InMemoryPublisher())


if TYPE_CHECKING:
    from virtuals.tkv.observer import ObserverProtocol

    _: type[ObserverProtocol[str]] = InMemoryObserver
    _p: type[PublisherProtocol] = InMemoryPublisher
