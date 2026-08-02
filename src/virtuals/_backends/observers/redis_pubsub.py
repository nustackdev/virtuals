"""Redis observer: listens to per-filter-hash pubsub channels.

Owns:
- `_local_hash_refcount`: per-hash refcount so we only SUBSCRIBE once
  regardless of how many local subs share a filter shape.
- HSET / HDEL of the registry HASH (`{prefix}:subs:registry`).
- PUBLISH of the control channel (`{prefix}:subs:changed`) after every
  HSET / HDEL, so cluster publishers refresh their caches.
- The listener thread that owns `self._pubsub` and dispatches inbound
  notif messages via `_dispatch_incoming(keys)`.
- A cleanup thread that periodically sweeps the registry HASH: for every
  entry with zero cluster subscribers (PUBSUB NUMSUB), HDEL + PUBLISH.

Does NOT own the publish queue, publish pipeline, or the publisher-side
remote-registry cache -- those live in RedisPublisher.

Self-echo: per the split design, this observer does NOT suppress
messages whose `iid` matches. In the split architecture, the observer
never publishes, so a same-process publisher's echo is simply an
ordinary inbound message. This is a behaviour change from the old fused
code (which self-suppressed), noted in the design doc.

Wire format is identical to the fused code so the old and new pair are
interoperable during migration:
- Channels: `{prefix}:notif:{filter_hash}`, `{prefix}:subs:changed`.
- HASH key: `{prefix}:subs:registry`.
- msgpack payload: `{"iid": pub_id, "keys": [[...], ...]}`.
"""

from __future__ import annotations

import json
import queue
import threading
from logging import getLogger
from typing import TYPE_CHECKING, Any

from virtuals.tkv.filter import Filter, filter_hash

from ._base import DEFAULT_DISPATCH_QUEUE_MAXSIZE, ObserverBase


try:
    import redis
except ImportError as e:
    raise ImportError(
        "redis is required for RedisObserver. Install with: pip install virtuals-py[redis]"
    ) from e

try:
    import msgpack
except ImportError as e:
    raise ImportError(
        "msgpack is required for RedisObserver. Install with: pip install virtuals-py[redis]"
    ) from e


if TYPE_CHECKING:
    from virtuals.tkv.observer import Subscription
    from virtuals.tkv.types import Key


logger = getLogger(__name__)


__all__ = [
    "RedisObserver",
]


# Default HASH cleanup sweep interval (seconds). Lazy; correctness never hurts.
DEFAULT_CLEANUP_INTERVAL_SECONDS: float = 30.0

# Pubsub polling timeout (seconds).
_PUBSUB_POLL_TIMEOUT: float = 0.05


class RedisObserver(ObserverBase):
    """Observer that receives per-hash routed notifications over Redis pubsub."""

    def __init__(
        self,
        *,
        redis_url: str = "redis://localhost:6379",
        channel_prefix: str = "nu",
        cleanup_interval_seconds: float = DEFAULT_CLEANUP_INTERVAL_SECONDS,
        dispatch_queue_maxsize: int = DEFAULT_DISPATCH_QUEUE_MAXSIZE,
    ) -> None:
        super().__init__(dispatch_queue_maxsize=dispatch_queue_maxsize)
        self._redis_url = redis_url
        self._channel_prefix = channel_prefix
        self._instance_id = id(self)
        self._cleanup_interval = cleanup_interval_seconds

        self._redis: Any = None  # HSET/HDEL/HGETALL/PUBSUB NUMSUB/PUBLISH
        self._redis_sub: Any = None  # dedicated pubsub connection
        self._pubsub: Any = None
        # NOTE: `self._pubsub` is touched EXCLUSIVELY by the listener thread.
        # Sub/unsub from other threads goes via `_control_cmd_queue`.

        self._local_hash_refcount: dict[str, int] = {}
        self._state_lock = threading.Lock()

        self._obs_stop_event = threading.Event()
        self._listener_thread: threading.Thread | None = None
        self._cleanup_thread: threading.Thread | None = None
        # (action, channel, ack_event). action in {'sub', 'unsub', 'shutdown'}.
        self._control_cmd_queue: (
            queue.Queue[tuple[str, str | None, threading.Event | None]] | None
        ) = None

    # -- Channel / key helpers ------------------------------------------------

    def _control_channel(self) -> str:
        return f"{self._channel_prefix}:subs:changed"

    def _notif_channel(self, h: str) -> str:
        return f"{self._channel_prefix}:notif:{h}"

    def _registry_key(self) -> str:
        return f"{self._channel_prefix}:subs:registry"

    # -- Lifecycle hooks ------------------------------------------------------

    def _on_connect(self) -> None:
        """Open connections + pubsub, start listener and cleanup threads."""
        self._redis = redis.from_url(self._redis_url)
        self._redis_sub = redis.from_url(self._redis_url)
        self._redis.ping()

        self._pubsub = self._redis_sub.pubsub()

        self._obs_stop_event.clear()
        self._control_cmd_queue = queue.Queue()

        self._listener_thread = threading.Thread(
            target=self._listener_loop,
            name=f"RedisObserverListener-{self._instance_id}",
            daemon=True,
        )
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            name=f"RedisObserverCleanup-{self._instance_id}",
            daemon=True,
        )
        self._listener_thread.start()
        self._cleanup_thread.start()

        logger.info(
            "Redis observer started at %s (prefix=%s, iid=%d)",
            self._redis_url,
            self._channel_prefix,
            self._instance_id,
        )

    def _on_disconnect(self) -> None:
        """Signal threads, wait for them to exit, close connections."""
        self._obs_stop_event.set()

        if self._control_cmd_queue is not None:
            try:
                self._control_cmd_queue.put_nowait(("shutdown", None, None))
            except queue.Full:
                pass

        for t in (self._listener_thread, self._cleanup_thread):
            if t is not None and t.is_alive():
                t.join(timeout=2.0)
        self._listener_thread = None
        self._cleanup_thread = None

        if self._pubsub is not None:
            try:
                self._pubsub.unsubscribe()
                self._pubsub.close()
            except Exception as e:
                logger.warning("Error closing pubsub: %s", e)
            self._pubsub = None

        for conn_attr in ("_redis_sub", "_redis"):
            conn = getattr(self, conn_attr)
            if conn is not None:
                try:
                    conn.close()
                except Exception as e:
                    logger.warning("Error closing %s: %s", conn_attr, e)
                setattr(self, conn_attr, None)

        with self._state_lock:
            self._local_hash_refcount = {}
        logger.info("Redis observer stopped")

    # -- Subscribe / unsubscribe hooks ---------------------------------------

    def _on_subscribe(self, subscription: Subscription) -> None:
        """First-time local sub for a hash -> SUBSCRIBE + HSET + PUBLISH."""
        self._register_local_filter(subscription.filter)

    def _on_unsubscribe(self, subscription: Subscription) -> None:
        """Last local sub for a hash -> UNSUBSCRIBE. HDEL is lazy (cleanup thread)."""
        self._unregister_local_filter(subscription.filter)

    def _register_local_filter(self, f: Filter) -> None:
        h = filter_hash(f)
        with self._state_lock:
            prior = self._local_hash_refcount.get(h, 0)
            self._local_hash_refcount[h] = prior + 1
            is_new = prior == 0

        if not is_new:
            return

        # SUBSCRIBE FIRST so we don't miss messages between our HSET and other
        # publishers routing to us. Delegated to listener thread; wait for ack.
        self._dispatch_pubsub_cmd("sub", self._notif_channel(h), wait=True)

        if self._redis is not None:
            try:
                self._redis.hset(
                    self._registry_key(),
                    h,
                    json.dumps(f.to_dict(), separators=(",", ":")),
                )
                self._redis.publish(self._control_channel(), b"")
            except Exception as e:
                logger.error("Register HSET/publish for hash %s failed: %s", h, e)

    def _unregister_local_filter(self, f: Filter) -> None:
        h = filter_hash(f)
        with self._state_lock:
            prior = self._local_hash_refcount.get(h, 0)
            if prior <= 1:
                self._local_hash_refcount.pop(h, None)
                is_last = True
            else:
                self._local_hash_refcount[h] = prior - 1
                is_last = False

        if not is_last:
            return
        # Fire-and-forget: caller doesn't need to wait for the UNSUBSCRIBE.
        # HASH cleanup happens in the periodic sweep, not on close.
        self._dispatch_pubsub_cmd("unsub", self._notif_channel(h), wait=False)

    def _dispatch_pubsub_cmd(self, action: str, channel: str, wait: bool = True) -> None:
        """Queue a sub/unsub for the listener thread and optionally wait for ack."""
        if self._control_cmd_queue is None:
            return
        ack = threading.Event() if wait else None
        try:
            self._control_cmd_queue.put_nowait((action, channel, ack))
        except queue.Full:
            logger.warning("pubsub cmd queue full; dropping %s(%s)", action, channel)
            return
        if ack is not None:
            ack.wait(timeout=1.0)

    # -- Listener thread ------------------------------------------------------

    def _listener_loop(self) -> None:
        """Owns `self._pubsub`; drains cmd queue; polls messages; fans out notifs."""
        notif_prefix = f"{self._channel_prefix}:notif:"

        while not self._obs_stop_event.is_set():
            # 1. Drain pending sub/unsub commands.
            drained_shutdown = False
            while True:
                try:
                    cmd = self._control_cmd_queue.get_nowait()
                except queue.Empty:
                    break
                action, channel, ack = cmd
                try:
                    if action == "sub" and channel is not None:
                        self._pubsub.subscribe(channel)
                    elif action == "unsub" and channel is not None:
                        self._pubsub.unsubscribe(channel)
                    elif action == "shutdown":
                        drained_shutdown = True
                except Exception as e:
                    logger.warning("Listener cmd %s(%s) failed: %s", action, channel, e)
                if ack is not None:
                    ack.set()
            if drained_shutdown or self._obs_stop_event.is_set():
                return

            # 2. Poll for one message.
            try:
                msg = self._pubsub.get_message(timeout=_PUBSUB_POLL_TIMEOUT)
            except Exception as e:
                logger.warning("Listener get_message failed: %s", e)
                self._obs_stop_event.wait(0.5)
                continue

            if msg is None or msg.get("type") not in ("message", "pmessage"):
                continue

            channel = msg["channel"]
            if isinstance(channel, bytes):
                channel = channel.decode("utf-8")

            if channel.startswith(notif_prefix):
                filter_hash_str = channel[len(notif_prefix) :]
                self._handle_notif(msg, filter_hash_str)

    def _handle_notif(self, msg: dict, filter_hash_str: str) -> None:
        try:
            payload = msgpack.unpackb(msg["data"], raw=False)
        except Exception as e:
            logger.warning("Bad msgpack payload on %s: %s", msg.get("channel"), e)
            return
        # Self-echo suppression is intentionally dropped in the split design:
        # the observer never publishes, so any inbound message is a real event.
        # Hash-scoped dispatch prevents duplicate delivery when a key matches
        # multiple registered filters (arriving on multiple notif channels).
        raw_keys = payload.get("keys", [])
        keys: list[Key] = [tuple(k) for k in raw_keys]
        if keys:
            self._dispatch_incoming(keys, filter_hash=filter_hash_str)

    # -- Cleanup thread -------------------------------------------------------

    def _cleanup_loop(self) -> None:
        """Periodic sweep: HDEL registry entries with zero cluster subscribers."""
        while not self._obs_stop_event.wait(self._cleanup_interval):
            if self._obs_stop_event.is_set():
                return
            self._cleanup_orphans()

    def _cleanup_orphans(self) -> None:
        if self._redis is None:
            return
        try:
            data = self._redis.hgetall(self._registry_key())
        except Exception as e:
            logger.warning("Cleanup HGETALL failed: %s", e)
            return
        if not data:
            return

        hashes = [h.decode("utf-8") if isinstance(h, bytes) else h for h in data.keys()]
        channels = [self._notif_channel(h) for h in hashes]
        try:
            counts = self._redis.pubsub_numsub(*channels)
        except Exception as e:
            logger.warning("Cleanup PUBSUB NUMSUB failed: %s", e)
            return

        orphans: list[str] = []
        for (_ch, count), h in zip(counts, hashes, strict=True):
            if count == 0:
                orphans.append(h)

        if not orphans:
            return
        try:
            self._redis.hdel(self._registry_key(), *orphans)
            self._redis.publish(self._control_channel(), b"")
            logger.debug("Cleaned %d orphan filter hashes", len(orphans))
        except Exception as e:
            logger.warning("Cleanup HDEL/publish failed: %s", e)
