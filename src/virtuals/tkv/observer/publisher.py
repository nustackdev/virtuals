"""Publisher protocol for observer notification delivery.

Publishers handle how matched notifications are delivered to subscribers.
The observer handles queueing and matching; the publisher handles delivery.
"""

from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING, Protocol, runtime_checkable


if TYPE_CHECKING:
    from ..types import Key
    from .registry import SubscriptionRegistry
    from .subscription import Subscription


logger = getLogger(__name__)


__all__ = [
    "PublisherProtocol",
    "deliver_local",
]


@runtime_checkable
class PublisherProtocol(Protocol):
    """Protocol for notification delivery backends.

    Publishers receive batches of keys and matched notifications from the
    observer's background thread and deliver them to subscribers.

    Implementations:
    - InMemoryPublisher: calls callbacks directly
    - RedisPublisher: calls callbacks + broadcasts to Redis
    """

    def start(self, registry: SubscriptionRegistry) -> None:
        """Start publisher (lifecycle hook).

        Called when observer connects. Publisher may start background
        threads (e.g. Redis listener) here.

        Args:
            registry: Subscription registry for matching (used by Redis listener).
        """
        ...

    def stop(self) -> None:
        """Stop publisher (lifecycle hook).

        Called when observer disconnects.
        """
        ...

    def deliver(
        self,
        keys: list[Key],
        notifications: list[tuple[Key, list[Subscription]]],
    ) -> None:
        """Deliver a batch of notifications.

        Called by the observer's background thread after matching.

        Args:
            keys: All keys in the batch (for broadcasting).
            notifications: Matched (key, subscriptions) pairs (for local delivery).
        """
        ...


def deliver_local(
    notifications: list[tuple[Key, list[Subscription]]],
) -> None:
    """Deliver notifications to local subscribers.

    Shared helper used by publisher implementations.

    Args:
        notifications: Matched (key, subscriptions) pairs.
    """
    for key, subs in notifications:
        for sub in subs:
            for error in sub.notify(key):
                logger.error("Callback failed for %s: %s", key, error)
