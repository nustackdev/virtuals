"""Protocol definitions for observer system.

Defines the abstract interfaces for observers and subscriptions.
The observer system provides:
- Fire-and-forget notifications (non-blocking enqueue)
- Background thread for matching and delivery
- Pluggable delivery backends (Publisher)
- Flexible filtering (prefix, suffix, wildcard, length, composite)
- Decoupled subscription from callbacks (subscribe once, bind/unbind callbacks)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable


if TYPE_CHECKING:
    from collections.abc import Iterable

    from virtuals.tkv.codec import CodecProtocol

    from ..types import Key
    from .subscription import Subscription
    from .types import SubscriptionOptions


__all__ = [
    "ObserverProtocol",
]


@runtime_checkable
class ObserverProtocol[EncodedKeyT](Protocol):
    """Protocol for observable adapters.

    Observers provide fire-and-forget notification for storage changes:
    - notify() enqueues keys and returns immediately
    - Background thread matches and delivers via pluggable Publisher
    - subscribe() registers subscriptions with flexible filtering

    Examples:
        >>> sub = observer.subscribe(
        ...     SubscriptionOptions(filter=PrefixFilter(prefix=("users",)))
        ... )
        >>> sub.bind(lambda key: print(f"Changed: {key}"))
    """

    @property
    def codec(self) -> CodecProtocol[EncodedKeyT, Any]:
        """Get key codec for encoding topics."""
        ...

    def subscribe(self, options: SubscriptionOptions) -> Subscription:
        """Subscribe to key changes with flexible filtering.

        Args:
            options: Subscription options including filter specification.

        Returns:
            Subscription object for binding callbacks and managing lifecycle.
        """
        ...

    def notify(self, keys: Key | Iterable[Key]) -> None:
        """Enqueue keys for notification. Returns immediately.

        Accepts a single key (tuple) or a batch of keys (set, list, etc).
        Fire-and-forget: background thread handles matching and delivery.

        Args:
            keys: Single key or batch of keys to notify about.
        """
        ...

    def flush(self, timeout: float = 1.0) -> None:
        """Wait for pending notifications to be delivered.

        Blocks until the queue is drained. Useful for testing.
        """
        ...

    def _close_subscription(self, subscription: Subscription) -> None:
        """Remove subscription from registry. Called by Subscription.close()."""
        ...
