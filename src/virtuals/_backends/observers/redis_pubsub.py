"""Redis pub/sub observer and publisher.

RedisPublisher delivers notifications by:
1. Calling local subscriber callbacks (same as InMemory)
2. Broadcasting batched key changes to Redis pub/sub (one message per batch)

RedisObserver is a convenience class: Observer + RedisPublisher.
"""

from __future__ import annotations

import json
import threading
from logging import getLogger
from typing import TYPE_CHECKING, Any

from virtuals.tkv.observer.publisher import deliver_local

from ._base import Observer


try:
    import redis
except ImportError as e:
    raise ImportError(
        "redis is required for RedisObserver. Install with: pip install virtuals-py[redis]"
    ) from e


if TYPE_CHECKING:
    from virtuals.tkv.codec import CodecProtocol
    from virtuals.tkv.observer import SubscriptionRegistry
    from virtuals.tkv.observer.publisher import PublisherProtocol
    from virtuals.tkv.observer.subscription import Subscription
    from virtuals.tkv.types import Key


__all__ = [
    "RedisObserver",
    "RedisPublisher",
]

logger = getLogger(__name__)


class RedisPublisher:
    """Publisher that delivers locally + broadcasts to Redis.

    Local delivery: calls subscriber callbacks directly.
    Remote delivery: batched redis.publish() - one message per deliver() call.

    Also runs a listener thread to receive remote notifications from other
    processes and deliver them to local subscribers.
    """

    def __init__(
        self,
        *,
        redis_url: str = "redis://localhost:6379",
        channel_prefix: str = "everyshape",
    ) -> None:
        """Initialize Redis publisher with connection settings."""
        self._redis_url = redis_url
        self._channel_prefix = channel_prefix
        self._channel = f"{channel_prefix}:notifications"
        self._instance_id = id(self)

        # Redis connections (created on start)
        self._redis_pub: Any = None
        self._redis_sub: Any = None
        self._pubsub: Any = None

        # Listener thread
        self._registry: SubscriptionRegistry | None = None
        self._listener_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self, registry: SubscriptionRegistry) -> None:
        """Connect to Redis and start listener thread."""
        self._registry = registry

        # Create Redis connections
        self._redis_pub = redis.from_url(self._redis_url)
        self._redis_sub = redis.from_url(self._redis_url)

        # Test connection
        self._redis_pub.ping()

        # Subscribe to notification channel
        self._pubsub = self._redis_sub.pubsub()
        self._pubsub.subscribe(self._channel)

        # Start listener thread
        self._stop_event.clear()
        self._listener_thread = threading.Thread(
            target=self._listener_loop,
            name=f"RedisPublisher-{self._instance_id}",
            daemon=True,
        )
        self._listener_thread.start()

        logger.info("Redis publisher started at %s, channel: %s", self._redis_url, self._channel)

    def stop(self) -> None:
        """Stop listener thread and disconnect from Redis."""
        self._stop_event.set()

        if self._listener_thread is not None and self._listener_thread.is_alive():
            self._listener_thread.join(timeout=2.0)
            self._listener_thread = None

        if self._pubsub is not None:
            try:
                self._pubsub.unsubscribe()
                self._pubsub.close()
            except Exception as e:
                logger.warning("Error closing pubsub: %s", e)
            self._pubsub = None

        if self._redis_sub is not None:
            try:
                self._redis_sub.close()
            except Exception as e:
                logger.warning("Error closing subscriber connection: %s", e)
            self._redis_sub = None

        if self._redis_pub is not None:
            try:
                self._redis_pub.close()
            except Exception as e:
                logger.warning("Error closing publisher connection: %s", e)
            self._redis_pub = None

        self._registry = None
        logger.info("Redis publisher stopped")

    def deliver(
        self,
        keys: list[Key],
        notifications: list[tuple[Key, list[Subscription]]],
    ) -> None:
        """Deliver locally + broadcast to Redis.

        Local: call subscriber callbacks for matched notifications.
        Remote: single batched redis.publish() for all keys.
        """
        # Local delivery
        deliver_local(notifications)

        # Remote broadcast (batched: one message for all keys)
        if keys and self._redis_pub is not None:
            try:
                message = json.dumps(
                    {
                        "instance_id": self._instance_id,
                        "keys": [list(k) for k in keys],
                    }
                )
                self._redis_pub.publish(self._channel, message)
            except Exception as e:
                logger.error("Failed to publish to Redis: %s", e)

    def _listener_loop(self) -> None:
        """Background thread: receive remote notifications from Redis."""
        logger.debug("Redis listener started for channel: %s", self._channel)

        while not self._stop_event.is_set():
            try:
                message = self._pubsub.get_message(timeout=0.1)

                if message is None or message["type"] != "message":
                    continue

                try:
                    raw_data = message["data"]
                    if isinstance(raw_data, bytes):
                        raw_data = raw_data.decode("utf-8")
                    data = json.loads(raw_data)
                    sender_id = data.get("instance_id")

                    # Skip self-notifications (already delivered locally)
                    if sender_id == self._instance_id:
                        continue

                    # Match and deliver remote notifications
                    if self._registry is not None:
                        remote_keys = [tuple(k) for k in data["keys"]]
                        notifications: list[tuple[Key, list[Subscription]]] = []
                        for key in remote_keys:
                            matched = self._registry.match(key)
                            if matched:
                                notifications.append((key, matched))
                        deliver_local(notifications)

                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    logger.warning("Invalid message received: %s, error: %s", message, e)

            except Exception as e:
                if not self._stop_event.is_set():
                    logger.error("Error in listener loop: %s", e)
                    self._stop_event.wait(0.5)

        logger.debug("Redis listener stopped")


class RedisObserver(Observer[str]):
    """Observer with Redis pub/sub delivery.

    Convenience class: Observer + RedisPublisher.
    """

    def __init__(
        self,
        codec: CodecProtocol[str, Any],
        *,
        redis_url: str = "redis://localhost:6379",
        channel_prefix: str = "everyshape",
        notify_self: bool = True,  # kept for API compat, ignored (always local + remote)
    ) -> None:
        """Initialize with RedisPublisher."""
        publisher = RedisPublisher(
            redis_url=redis_url,
            channel_prefix=channel_prefix,
        )
        super().__init__(codec=codec, publisher=publisher)


if TYPE_CHECKING:
    from virtuals.tkv.observer import ObserverProtocol

    _: type[ObserverProtocol[str]] = RedisObserver
    _p: type[PublisherProtocol] = RedisPublisher
