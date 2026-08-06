"""Protocol definitions for the observer (read-side) system.

An observer lives at process scope. It owns a SubscriptionRegistry and a
listener on the transport. When something arrives on the transport, the
observer matches it against the registry and calls the bound callbacks.

Observers know NOTHING about local writes or storage. They never call
notify(). They meet publishers via the transport only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from typing_extensions import Self


if TYPE_CHECKING:
    from types import TracebackType

    from .subscription import Subscription
    from .types import SubscriptionOptions


__all__ = [
    "ObserverProtocol",
]


@runtime_checkable
class ObserverProtocol(Protocol):
    """Protocol for read-side observer backends.

    Observers subscribe local callbacks against filters, listen on a
    transport for inbound keys, match them against their local
    `SubscriptionRegistry`, and fan out to the bound callbacks.

    Examples:
        >>> sub = observer.subscribe(
        ...     SubscriptionOptions(filter=PrefixFilter(prefix=("users",)))
        ... )
        >>> sub.bind(lambda key: print(f"Changed: {key}"))
    """

    def connect(self) -> None:
        """Open the transport, start the listener, ready the registry."""
        ...

    def disconnect(self) -> None:
        """Symmetric shutdown: stop the listener, close the transport, clear registry."""
        ...

    def subscribe(self, options: SubscriptionOptions) -> Subscription:
        """Subscribe to key changes with flexible filtering.

        Args:
            options: Subscription options including filter specification.

        Returns:
            Subscription object for binding callbacks and managing lifecycle.
        """
        ...

    def _close_subscription(self, subscription: Subscription) -> None:
        """Remove subscription from registry. Called by Subscription.close()."""
        ...

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None: ...
