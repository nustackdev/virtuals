"""Observer implementation with fire-and-forget notifications.

The Observer provides:
- Non-blocking notify(): enqueues keys and returns immediately
- Background thread for matching and delivery
- Pluggable Publisher for delivery backend (InMemory, Redis, etc.)
- SubscriptionRegistry for efficient pattern matching
"""

from __future__ import annotations

import threading
from collections import deque
from logging import getLogger
from typing import TYPE_CHECKING, Any, Self

from virtuals.tkv.observer import (
    ObserverConnectionError,
    Subscription,
    SubscriptionOptions,
    SubscriptionRegistry,
)


if TYPE_CHECKING:
    from collections.abc import Iterable
    from types import TracebackType

    from virtuals.tkv.codec import CodecProtocol
    from virtuals.tkv.observer.publisher import PublisherProtocol
    from virtuals.tkv.types import Key


logger = getLogger(__name__)


__all__ = [
    "Observer",
]


class Observer[EncodedKeyT]:
    """Observer with fire-and-forget notifications.

    Owns a queue, background thread, subscription registry, and a pluggable
    Publisher for delivery. notify() enqueues keys and returns immediately.
    The background thread drains the queue, matches against the registry,
    and hands matched notifications to the publisher for delivery.

    Type Parameters:
        EncodedKeyT: Encoded topic type (e.g., str for in-memory, bytes for RocksDB)
    """

    def __init__(
        self,
        codec: CodecProtocol[EncodedKeyT, Any],
        publisher: PublisherProtocol,
    ) -> None:
        self._codec = codec
        self._publisher = publisher
        self._connected: bool = False

        # Queue and threading (initialized on connect)
        # Queue holds Key tuples and flush sentinels (threading.Event)
        self._queue: deque[Key | threading.Event] = deque()
        self._registry: SubscriptionRegistry | None = None
        self._event = threading.Event()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def codec(self) -> CodecProtocol[EncodedKeyT, Any]:
        """Get codec for encoding/decoding topics."""
        return self._codec

    def _ensure_connected(self) -> None:
        if not self._connected:
            raise ObserverConnectionError("Observer not connected")

    def connect(self) -> None:
        """Connect observer: init registry, start bg thread, start publisher."""
        if self._connected:
            return
        try:
            self._registry = SubscriptionRegistry()
            self._stop_event.clear()
            self._queue.clear()

            # Start publisher
            self._publisher.start(self._registry)

            # Start background worker
            self._thread = threading.Thread(
                target=self._worker,
                name=f"Observer-{id(self)}",
                daemon=True,
            )
            self._thread.start()

            self._connected = True
        except Exception as e:
            raise ObserverConnectionError(f"Failed to connect: {e}") from e

    def disconnect(self) -> None:
        """Disconnect observer: stop bg thread, stop publisher, clear registry."""
        if not self._connected:
            return
        try:
            # Signal worker to stop and wake it
            self._stop_event.set()
            self._event.set()

            if self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=2.0)
                self._thread = None

            # Stop publisher
            self._publisher.stop()
        finally:
            if self._registry is not None:
                self._registry.clear()
                self._registry = None
            self._connected = False
            self._queue.clear()

    def notify(self, keys: Key | Iterable[Key]) -> None:
        """Enqueue keys for notification. Returns immediately.

        Accepts a single key (tuple) or a batch of keys (set, list, etc).
        Fire-and-forget: the background thread handles matching and delivery.
        """
        self._ensure_connected()
        if isinstance(keys, tuple):
            self._queue.append(keys)
        else:
            self._queue.extend(keys)
        self._event.set()

    def subscribe(self, options: SubscriptionOptions) -> Subscription:
        """Subscribe to key changes with flexible filtering."""
        self._ensure_connected()

        subscription = Subscription(
            _options=options,
            _observer=self,
        )

        if self._registry is not None:
            self._registry.add(subscription)

        return subscription

    def _close_subscription(self, subscription: Subscription) -> None:
        """Remove subscription from registry. Called by Subscription.close()."""
        if self._registry is not None:
            self._registry.remove(subscription)

    def flush(self, timeout: float = 1.0) -> None:
        """Wait for pending notifications to be delivered.

        Places a sentinel in the queue. When the worker reaches it,
        it signals completion. Guarantees all prior items are processed.

        Args:
            timeout: Maximum seconds to wait.
        """
        if not self._connected:
            return
        done = threading.Event()
        self._queue.append(done)  # sentinel after all pending keys
        self._event.set()  # wake worker
        done.wait(timeout=timeout)

    def _worker(self) -> None:
        """Background thread: drain queue, match, deliver via publisher."""
        while not self._stop_event.is_set():
            self._event.wait(timeout=0.1)
            self._event.clear()

            # Drain queue, separating keys from flush sentinels
            batch: list[Key] = []
            while self._queue:
                try:
                    item = self._queue.popleft()
                except IndexError:
                    break
                if isinstance(item, threading.Event):
                    # Process keys accumulated so far, then signal
                    self._deliver_batch(batch)
                    batch = []
                    item.set()
                else:
                    batch.append(item)

            # Deliver remaining keys
            self._deliver_batch(batch)

    def _deliver_batch(self, batch: list[Key]) -> None:
        """Match and deliver a batch of keys via publisher."""
        if not batch:
            return

        registry = self._registry
        if registry is None:
            return

        if len(registry) == 0:
            self._publisher.deliver(batch, [])
            return

        notifications: list[tuple[Key, list[Subscription]]] = []
        for key in batch:
            matched = registry.match(key)
            if matched:
                notifications.append((key, matched))

        self._publisher.deliver(batch, notifications)

    def __enter__(self) -> Self:
        """Enter context manager."""
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager."""
        self.disconnect()
