"""Observer implementation with fire-and-forget notifications.

The Observer provides:
- Non-blocking notify(): enqueues keys and returns. Bounded queue provides
  natural backpressure -- notify() blocks briefly when the queue is full
  rather than dropping keys or growing memory unbounded.
- Background thread for matching and delivery.
- Pluggable Publisher for delivery backend (InMemory, Redis, etc.).
- SubscriptionRegistry for efficient pattern matching.

The storage write path calls only notify() -- it no longer blocks on flush().
flush() remains public for tests, shutdown, and callers that explicitly want
a delivery barrier.
"""

from __future__ import annotations

import queue
import threading
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


# Default queue capacity. At 1200 keys/s target throughput this is ~80s of
# buffer -- comfortable headroom for bursts, small enough that a stuck
# subscriber surfaces as writer backpressure well before OOM.
DEFAULT_QUEUE_MAXSIZE: int = 100_000

# Sentinel used to unblock a worker sleeping on Queue.get() during disconnect.
_STOP_MARKER: object = object()


class Observer[EncodedKeyT]:
    """Observer with fire-and-forget notifications.

    Owns a bounded queue, background thread, subscription registry, and a
    pluggable Publisher. notify() enqueues keys; the background thread drains,
    matches, and hands matches to the publisher.

    Backpressure: notify() blocks briefly when the queue is full (default
    capacity DEFAULT_QUEUE_MAXSIZE). No silent drops.

    Type Parameters:
        EncodedKeyT: Encoded topic type (e.g., str for in-memory, bytes for RocksDB)
    """

    def __init__(
        self,
        codec: CodecProtocol[EncodedKeyT, Any],
        publisher: PublisherProtocol,
        queue_maxsize: int = DEFAULT_QUEUE_MAXSIZE,
    ) -> None:
        self._codec = codec
        self._publisher = publisher
        self._queue_maxsize = queue_maxsize
        self._connected: bool = False

        self._queue: queue.Queue[Any] = queue.Queue(maxsize=queue_maxsize)
        self._registry: SubscriptionRegistry | None = None
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
            # Fresh queue on each connect so re-connect doesn't inherit stale items.
            self._queue = queue.Queue(maxsize=self._queue_maxsize)

            self._publisher.start(self._registry)

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
            self._stop_event.set()
            # Best-effort wake if worker is blocked on Queue.get().
            try:
                self._queue.put_nowait(_STOP_MARKER)
            except queue.Full:
                pass  # worker will wake via get() timeout instead

            if self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=2.0)
                self._thread = None

            self._publisher.stop()
        finally:
            if self._registry is not None:
                self._registry.clear()
                self._registry = None
            self._connected = False

    def notify(self, keys: Key | Iterable[Key]) -> None:
        """Enqueue keys for notification.

        Fire-and-forget: writer just enqueues; the background worker handles
        matching and delivery. When the queue is full, notify() blocks
        (backpressure) rather than dropping keys.

        Accepts a single key (tuple) or a batch of keys (set, list, etc.).
        """
        self._ensure_connected()
        if isinstance(keys, tuple):
            self._queue.put(keys)
        else:
            for k in keys:
                self._queue.put(k)

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

        Places a sentinel in the queue. When the worker reaches it, all
        previously-enqueued keys have been processed and dispatched to
        callbacks. Public API for tests, shutdown, and callers that
        explicitly want a delivery barrier. Storage write paths do NOT
        call this -- they enqueue and return.

        Args:
            timeout: Maximum seconds to wait.
        """
        if not self._connected:
            return
        done = threading.Event()
        self._queue.put(done)  # sentinel after all pending keys
        done.wait(timeout=timeout)

    def _worker(self) -> None:
        """Background thread: drain queue, match, deliver via publisher."""
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Drain what's queued right now into a single batch. Sentinels
            # flush the accumulated batch first, then signal.
            batch: list[Key] = []
            while True:
                if item is _STOP_MARKER:
                    self._deliver_batch(batch)
                    return
                if isinstance(item, threading.Event):
                    self._deliver_batch(batch)
                    batch = []
                    item.set()
                else:
                    batch.append(item)  # Key tuple
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    break

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
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.disconnect()
