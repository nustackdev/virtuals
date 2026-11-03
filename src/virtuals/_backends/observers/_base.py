"""Observer base: registry + optional internal queue for fire-and-forget dispatch.

Owns the SubscriptionRegistry, subscribe / _close_subscription, and the
transport-facing `_dispatch_incoming(keys)` entry point. Subclasses
provide `_on_connect / _on_disconnect / _on_subscribe / _on_unsubscribe`
hooks for backend-specific setup (registering with an in-mem transport,
a confirmed SUBSCRIBE plus routing registration on Redis, etc.).

Includes an internal queue + worker so `_dispatch_incoming` returns as
soon as the keys are queued: a slow subscriber backs up the OBSERVER's
queue first. When that queue is full, `_dispatch_incoming` blocks
(backpressure into the transport) instead of dropping keys.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from logging import getLogger
from typing import TYPE_CHECKING, Any

from typing_extensions import Self

from virtuals.tkv.observer import (
    ObserverConnectionError,
    Subscription,
    SubscriptionOptions,
    SubscriptionRegistry,
    deliver_local,
)


if TYPE_CHECKING:
    from collections.abc import Iterable
    from types import TracebackType

    from virtuals.tkv.types import Key


@dataclass(frozen=True, slots=True)
class _TaggedKey:
    """A key tagged with a filter hash for hash-scoped delivery."""

    key: Any  # tkv.Key at runtime; kept loose to avoid a circular import
    filter_hash: str


logger = getLogger(__name__)


__all__ = [
    "DEFAULT_DISPATCH_QUEUE_MAXSIZE",
    "ObserverBase",
]


# Default dispatch queue capacity: buffers incoming batches so transport
# listeners rarely block on slow subscribers. When full, they block.
DEFAULT_DISPATCH_QUEUE_MAXSIZE: int = 100_000

# Sentinel to unblock the dispatch worker on Queue.get() during disconnect.
_STOP_MARKER: object = object()

# How often a producer blocked on a full dispatch queue rechecks for shutdown.
_PUT_POLL_SECONDS: float = 0.1


class ObserverBase:
    """Read-side base: registry, subscribe, dispatch worker."""

    def __init__(
        self,
        *,
        dispatch_queue_maxsize: int = DEFAULT_DISPATCH_QUEUE_MAXSIZE,
    ) -> None:
        self._dispatch_queue_maxsize = dispatch_queue_maxsize
        self._connected: bool = False
        self._registry: SubscriptionRegistry | None = None

        self._dispatch_queue: queue.Queue[Any] = queue.Queue(maxsize=dispatch_queue_maxsize)
        self._stop_event = threading.Event()
        self._dispatch_thread: threading.Thread | None = None

    def _ensure_connected(self) -> None:
        if not self._connected:
            raise ObserverConnectionError("Observer not connected")

    def connect(self) -> None:
        """Init registry + dispatch worker, then call the subclass _on_connect hook."""
        if self._connected:
            return
        try:
            self._registry = SubscriptionRegistry()
            self._stop_event.clear()
            self._dispatch_queue = queue.Queue(maxsize=self._dispatch_queue_maxsize)

            self._on_connect()

            self._dispatch_thread = threading.Thread(
                target=self._dispatch_worker,
                name=f"Observer-{id(self)}",
                daemon=True,
            )
            self._dispatch_thread.start()

            self._connected = True
        except Exception as e:
            raise ObserverConnectionError(f"Failed to connect: {e}") from e

    def disconnect(self) -> None:
        """Symmetric shutdown. _on_disconnect hook runs after the worker joins."""
        if not self._connected:
            return
        try:
            self._stop_event.set()
            try:
                self._dispatch_queue.put_nowait(_STOP_MARKER)
            except queue.Full:
                pass

            if self._dispatch_thread is not None and self._dispatch_thread.is_alive():
                self._dispatch_thread.join(timeout=2.0)
                self._dispatch_thread = None

            self._on_disconnect()
        finally:
            if self._registry is not None:
                self._registry.clear()
                self._registry = None
            self._connected = False

    def subscribe(self, options: SubscriptionOptions) -> Subscription:
        """Register a subscription on the local registry and call _on_subscribe hook."""
        self._ensure_connected()

        sub = Subscription(_options=options, _observer=self)
        registry = self._registry
        if registry is None:  # pragma: no cover -- guarded by _ensure_connected
            raise ObserverConnectionError("Observer registry not initialized")
        registry.add(sub)
        try:
            self._on_subscribe(sub)
        except Exception:
            # roll back the registry insert so state stays consistent
            registry.remove(sub)
            raise
        return sub

    def _close_subscription(self, subscription: Subscription) -> None:
        """Remove from registry and call _on_unsubscribe hook."""
        if self._registry is not None:
            self._registry.remove(subscription)
        try:
            self._on_unsubscribe(subscription)
        except Exception as e:
            logger.warning("Observer _on_unsubscribe hook raised: %s", e)

    def _dispatch_incoming(self, keys: Iterable[Key], filter_hash: str | None = None) -> None:
        """Transport-facing entry point.

        Enqueues keys onto the dispatch queue. When the queue is full the
        call blocks until the dispatch worker makes room: keys are never
        dropped silently. Backpressure then reaches the transport (for Redis:
        the listener stops reading, valkey buffers, and past its pubsub
        output limit disconnects the client, which is logged and recovered
        by re-registration). If the observer isn't connected, or disconnects
        while blocked, the keys are dropped -- that is lifecycle, not load.

        `filter_hash` (optional) narrows matching at delivery time to subs
        whose canonical filter hashes to that value. Used by hash-routed
        transports (Redis pubsub) to prevent duplicate fanout when a key
        matches multiple registered filters.
        """
        if not self._connected:
            return
        q = self._dispatch_queue
        for k in keys:
            item = k if filter_hash is None else _TaggedKey(k, filter_hash)
            try:
                q.put_nowait(item)
                continue
            except queue.Full:
                pass
            while True:
                try:
                    q.put(item, timeout=_PUT_POLL_SECONDS)
                    break
                except queue.Full:
                    if self._stop_event.is_set() or not self._connected:
                        logger.warning("Observer stopping; dropping undelivered keys")
                        return

    def flush(self, timeout: float = 1.0) -> None:
        """Wait for the dispatch queue to drain. Test helper."""
        if not self._connected:
            return
        done = threading.Event()
        self._dispatch_queue.put(done)
        done.wait(timeout=timeout)

    def _dispatch_worker(self) -> None:
        """Drain the dispatch queue: match each key + call bound callbacks."""
        while not self._stop_event.is_set():
            try:
                item = self._dispatch_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            batch: list[Key | _TaggedKey] = []
            while True:
                if item is _STOP_MARKER:
                    self._deliver_batch(batch)
                    return
                if isinstance(item, threading.Event):
                    self._deliver_batch(batch)
                    batch = []
                    item.set()
                else:
                    batch.append(item)
                try:
                    item = self._dispatch_queue.get_nowait()
                except queue.Empty:
                    break

            self._deliver_batch(batch)

    def _deliver_batch(self, batch: list[Key | _TaggedKey]) -> None:
        if not batch:
            return
        registry = self._registry
        if registry is None:
            return
        if len(registry) == 0:
            return
        notifications: list[tuple[Key, list[Subscription]]] = []
        for item in batch:
            if isinstance(item, _TaggedKey):
                key = item.key
                h = item.filter_hash
                matched = [s for s in registry.match(key) if s.filter_hash == h]
            else:
                key = item
                matched = registry.match(key)
            if matched:
                notifications.append((key, matched))
        if notifications:
            deliver_local(notifications)

    # -- Subclass hooks -------------------------------------------------------

    def _on_connect(self) -> None:
        """Hook called from connect() before the dispatch worker starts."""
        return None

    def _on_disconnect(self) -> None:
        """Hook called from disconnect() after the dispatch worker joins."""
        return None

    def _on_subscribe(self, subscription: Subscription) -> None:
        """Hook called after a subscription is added to the registry."""
        return None

    def _on_unsubscribe(self, subscription: Subscription) -> None:
        """Hook called after a subscription is removed from the registry."""
        return None

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
