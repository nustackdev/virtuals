"""Redis pub/sub observer with per-hash channel routing.

Design:

- HASH `{prefix}:subs:registry` is the source of truth for active filter
  shapes across the whole cluster. Maps `filter_hash -> filter_json`.
- Channel `{prefix}:subs:changed` carries a signal (no payload) whenever
  the registry HASH mutates. Every observer subscribes to it and refreshes
  its local cache via HGETALL on receipt (debounced to coalesce bursts).
- Per-hash channels `{prefix}:notif:{hash}` carry actual key notifications
  (msgpack-encoded `{"iid": instance_id, "keys": [[seg, ...], ...]}`).
  Observers only publish to hashes that have at least one subscriber
  somewhere in the cluster (checked against the local cache of the HASH).

Zero-interest publish is impossible by construction: an empty local cache
means no publishes happen. Cleanup of dead hashes is lazy via a periodic
sweep that calls `PUBSUB NUMSUB` per hash and `HDEL`s any with zero global
subscribers.

Threading:

- Base `Observer` worker thread calls `deliver()` from `_deliver_batch`.
  Local delivery is inline; remote publishes are enqueued to a separate
  publish thread that pipelines them (writer thread never pays network
  RTT for publishes).
- Listener thread pumps redis pubsub messages: control-channel signals
  trigger debounced HGETALL; notif-channel messages fan out to local
  subscribers via `registry.match(key)` (same code path as local delivery).
- Cleanup thread runs every N seconds, HDEL-ing hashes with zero remote
  subscribers.
- All redis-pubsub state (`self._pubsub`) is guarded by `_pubsub_lock`.

RedisObserver = Observer + RedisPublisher (thin wrapper that hooks
subscribe/unsubscribe into publisher-side hash registration).
"""

from __future__ import annotations

import json
import queue
import threading
from logging import getLogger
from typing import TYPE_CHECKING, Any

from virtuals.tkv.filter import Filter, filter_from_dict, filter_hash
from virtuals.tkv.observer.publisher import deliver_local

from ._base import DEFAULT_QUEUE_MAXSIZE, Observer


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
    from virtuals.tkv.codec import CodecProtocol
    from virtuals.tkv.observer import Subscription, SubscriptionOptions, SubscriptionRegistry
    from virtuals.tkv.observer.publisher import PublisherProtocol
    from virtuals.tkv.types import Key


__all__ = [
    "RedisObserver",
    "RedisPublisher",
]

logger = getLogger(__name__)


# Default publish queue capacity. Batches accumulate here between drains.
# Values larger than DEFAULT_PUBLISH_QUEUE_MAXSIZE will block the observer
# worker briefly (backpressure), which is the intended overload behavior.
DEFAULT_PUBLISH_QUEUE_MAXSIZE: int = 10_000

# Default HASH cleanup sweep interval (seconds). Lazy; correctness never hurts.
DEFAULT_CLEANUP_INTERVAL_SECONDS: float = 30.0

# Default HGETALL refresh debounce window (seconds). Coalesces bursts of
# "changed" signals from many observers subscribing at once.
DEFAULT_REFRESH_DEBOUNCE_SECONDS: float = 0.01

# Pubsub polling timeout (seconds). Bounds the max latency a subscribe/
# unsubscribe waits for the pubsub lock.
_PUBSUB_POLL_TIMEOUT: float = 0.05


class RedisPublisher:
    """Publisher: local callbacks + per-hash routed redis pub/sub."""

    def __init__(
        self,
        *,
        redis_url: str = "redis://localhost:6379",
        channel_prefix: str = "everyshape",
        publish_queue_maxsize: int = DEFAULT_PUBLISH_QUEUE_MAXSIZE,
        cleanup_interval_seconds: float = DEFAULT_CLEANUP_INTERVAL_SECONDS,
        refresh_debounce_seconds: float = DEFAULT_REFRESH_DEBOUNCE_SECONDS,
    ) -> None:
        self._redis_url = redis_url
        self._channel_prefix = channel_prefix
        self._instance_id = id(self)
        self._publish_queue_maxsize = publish_queue_maxsize
        self._cleanup_interval = cleanup_interval_seconds
        self._refresh_debounce = refresh_debounce_seconds

        # Redis connections (created on start)
        self._redis_pub: Any = None  # for publish + HSET/HDEL/HGETALL/PUBSUB NUMSUB
        self._redis_sub: Any = None  # dedicated connection for pubsub
        self._pubsub: Any = None
        # NOTE: `self._pubsub` is touched EXCLUSIVELY on the listener thread.
        # Subscribe/unsubscribe from other threads goes via `_control_cmd_queue`
        # and awaits an ack event. This is race-free and avoids the pubsub-lock
        # contention we saw when serialising `get_message` with sub/unsub calls.

        # Registry + local caches
        self._registry: SubscriptionRegistry | None = None
        # remote_registry: filter_hash -> reconstructed Filter. Source of routing decisions.
        self._remote_registry: dict[str, Filter] = {}
        # local_hash_refcount: filter_hash -> count of local subs sharing that hash.
        # Determines when to (un)SUBSCRIBE our notif channel.
        self._local_hash_refcount: dict[str, int] = {}
        self._state_lock = threading.Lock()

        # Threads
        self._stop_event = threading.Event()
        self._listener_thread: threading.Thread | None = None
        self._publish_thread: threading.Thread | None = None
        self._cleanup_thread: threading.Thread | None = None
        self._publish_queue: queue.Queue[tuple[str, bytes] | None] | None = None
        # Cross-thread pubsub commands: (action, channel, ack_event)
        # action is 'sub' or 'unsub'; ack_event is set by listener after redis command
        # is issued. 'shutdown' with None channel/event tells listener to exit.
        self._control_cmd_queue: queue.Queue[tuple[str, str | None, threading.Event | None]] | None = None
        # Debounced refresh: single-shot timer, replaced when new signals arrive.
        self._refresh_timer: threading.Timer | None = None
        self._refresh_timer_lock = threading.Lock()

    # -- Channel / key helpers ------------------------------------------------

    def _control_channel(self) -> str:
        return f"{self._channel_prefix}:subs:changed"

    def _notif_channel(self, h: str) -> str:
        return f"{self._channel_prefix}:notif:{h}"

    def _registry_key(self) -> str:
        return f"{self._channel_prefix}:subs:registry"

    # -- Lifecycle ------------------------------------------------------------

    def start(self, registry: SubscriptionRegistry) -> None:
        """Connect to Redis, subscribe to control channel, seed remote cache, start threads."""
        self._registry = registry
        self._redis_pub = redis.from_url(self._redis_url)
        self._redis_sub = redis.from_url(self._redis_url)
        self._redis_pub.ping()

        self._pubsub = self._redis_sub.pubsub()
        # SUBSCRIBE control channel FIRST so we don't miss changes during HGETALL.
        # Safe: listener hasn't started yet, only this thread touches _pubsub.
        self._pubsub.subscribe(self._control_channel())

        # Seed remote registry from current HASH state.
        self._refresh_remote_registry()

        # Threads
        self._stop_event.clear()
        self._publish_queue = queue.Queue(maxsize=self._publish_queue_maxsize)
        self._control_cmd_queue = queue.Queue()

        self._listener_thread = threading.Thread(
            target=self._listener_loop,
            name=f"RedisListener-{self._instance_id}",
            daemon=True,
        )
        self._publish_thread = threading.Thread(
            target=self._publish_loop,
            name=f"RedisPublisher-{self._instance_id}",
            daemon=True,
        )
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            name=f"RedisCleanup-{self._instance_id}",
            daemon=True,
        )
        self._listener_thread.start()
        self._publish_thread.start()
        self._cleanup_thread.start()

        logger.info(
            "Redis publisher started at %s (prefix=%s, iid=%d)",
            self._redis_url,
            self._channel_prefix,
            self._instance_id,
        )

    def stop(self) -> None:
        """Signal all threads, cancel debounced refresh, close connections."""
        self._stop_event.set()

        # Cancel debounce timer
        with self._refresh_timer_lock:
            if self._refresh_timer is not None:
                self._refresh_timer.cancel()
                self._refresh_timer = None

        # Wake publish thread
        if self._publish_queue is not None:
            try:
                self._publish_queue.put_nowait(None)  # stop sentinel
            except queue.Full:
                pass

        # Wake listener thread (drains cmd queue between polls)
        if self._control_cmd_queue is not None:
            try:
                self._control_cmd_queue.put_nowait(("shutdown", None, None))
            except queue.Full:
                pass

        for t in (self._listener_thread, self._publish_thread, self._cleanup_thread):
            if t is not None and t.is_alive():
                t.join(timeout=2.0)
        self._listener_thread = None
        self._publish_thread = None
        self._cleanup_thread = None

        if self._pubsub is not None:
            # Listener thread has exited, safe to touch pubsub from here.
            try:
                self._pubsub.unsubscribe()
                self._pubsub.close()
            except Exception as e:
                logger.warning("Error closing pubsub: %s", e)
            self._pubsub = None

        for conn_attr in ("_redis_sub", "_redis_pub"):
            conn = getattr(self, conn_attr)
            if conn is not None:
                try:
                    conn.close()
                except Exception as e:
                    logger.warning("Error closing %s: %s", conn_attr, e)
                setattr(self, conn_attr, None)

        with self._state_lock:
            self._remote_registry = {}
            self._local_hash_refcount = {}
        self._registry = None
        logger.info("Redis publisher stopped")

    # -- deliver (called from Observer worker thread) -------------------------

    def deliver(
        self,
        keys: list[Key],
        notifications: list[tuple[Key, list[Subscription]]],
    ) -> None:
        """Local delivery + route to per-hash channels for remote observers."""
        # Local delivery first (inline, fast)
        deliver_local(notifications)

        if not keys or self._publish_queue is None:
            return

        # Snapshot remote registry so publish routing doesn't hold the state lock.
        with self._state_lock:
            remote = dict(self._remote_registry)

        if not remote:
            return  # nobody in the cluster is interested; publish nothing

        # Bucket keys per matching hash.
        per_hash: dict[str, list[Key]] = {}
        for key in keys:
            for h, f in remote.items():
                if f.matches(key):
                    per_hash.setdefault(h, []).append(key)

        if not per_hash:
            return

        # Enqueue one publish per bucket. Publish thread will pipeline them.
        for h, ks in per_hash.items():
            payload = msgpack.packb(
                {
                    "iid": self._instance_id,
                    "keys": [list(k) for k in ks],
                },
                use_bin_type=True,
            )
            try:
                self._publish_queue.put_nowait((self._notif_channel(h), payload))
            except queue.Full:
                logger.warning(
                    "Publish queue full; dropping %d keys for hash %s", len(ks), h
                )

    # -- Publish thread -------------------------------------------------------

    def _publish_loop(self) -> None:
        """Drain publish queue, batch into redis pipelines, execute."""
        while not self._stop_event.is_set():
            try:
                item = self._publish_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                return

            batch: list[tuple[str, bytes]] = [item]
            # Drain whatever is ready right now into one pipeline.
            while True:
                try:
                    nxt = self._publish_queue.get_nowait()
                except queue.Empty:
                    break
                if nxt is None:
                    self._flush_pipeline(batch)
                    return
                batch.append(nxt)

            self._flush_pipeline(batch)

    def _flush_pipeline(self, batch: list[tuple[str, bytes]]) -> None:
        if not batch or self._redis_pub is None:
            return
        try:
            pipe = self._redis_pub.pipeline(transaction=False)
            for channel, payload in batch:
                pipe.publish(channel, payload)
            pipe.execute()
        except Exception as e:
            logger.error("Publish pipeline failed (%d msgs): %s", len(batch), e)

    # -- Listener thread ------------------------------------------------------

    def _listener_loop(self) -> None:
        """Sole owner of `self._pubsub`.

        Drains pubsub command queue (sub/unsub requests from other threads),
        polls redis pubsub for messages, dispatches control -> refresh,
        notif -> local fanout.
        """
        control_channel = self._control_channel()
        notif_prefix = f"{self._channel_prefix}:notif:"

        while not self._stop_event.is_set():
            # 1. Drain pending sub/unsub commands (fast, non-blocking)
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
            if drained_shutdown or self._stop_event.is_set():
                return

            # 2. Poll for one message
            try:
                msg = self._pubsub.get_message(timeout=_PUBSUB_POLL_TIMEOUT)
            except Exception as e:
                logger.warning("Listener get_message failed: %s", e)
                # Brief pause to avoid tight-looping on a persistent error
                self._stop_event.wait(0.5)
                continue

            if msg is None or msg.get("type") not in ("message", "pmessage"):
                continue

            channel = msg["channel"]
            if isinstance(channel, bytes):
                channel = channel.decode("utf-8")

            if channel == control_channel:
                self._schedule_refresh()
            elif channel.startswith(notif_prefix):
                self._handle_notif(msg)

    def _handle_notif(self, msg: dict) -> None:
        try:
            payload = msgpack.unpackb(msg["data"], raw=False)
        except Exception as e:
            logger.warning("Bad msgpack payload on %s: %s", msg.get("channel"), e)
            return
        if payload.get("iid") == self._instance_id:
            return  # self-echo suppression
        registry = self._registry
        if registry is None:
            return
        for raw_key in payload.get("keys", []):
            key: Key = tuple(raw_key)
            matched = registry.match(key)
            if matched:
                deliver_local([(key, matched)])

    # -- Debounced refresh ----------------------------------------------------

    def _schedule_refresh(self) -> None:
        """Coalesce multiple 'changed' signals into a single HGETALL."""
        with self._refresh_timer_lock:
            if self._refresh_timer is not None:
                return  # one already pending
            t = threading.Timer(self._refresh_debounce, self._do_refresh)
            t.daemon = True
            self._refresh_timer = t
            t.start()

    def _do_refresh(self) -> None:
        with self._refresh_timer_lock:
            self._refresh_timer = None
        if not self._stop_event.is_set():
            self._refresh_remote_registry()

    def _refresh_remote_registry(self) -> None:
        """HGETALL registry HASH -> rebuild `_remote_registry`."""
        if self._redis_pub is None:
            return
        try:
            data = self._redis_pub.hgetall(self._registry_key())
        except Exception as e:
            logger.warning("Refresh HGETALL failed: %s", e)
            return

        new_registry: dict[str, Filter] = {}
        for h_raw, fj_raw in data.items():
            h = h_raw.decode("utf-8") if isinstance(h_raw, bytes) else h_raw
            fj = fj_raw.decode("utf-8") if isinstance(fj_raw, bytes) else fj_raw
            try:
                new_registry[h] = filter_from_dict(json.loads(fj))
            except Exception as e:
                logger.warning("Skipping unreadable registry entry %s: %s", h, e)

        with self._state_lock:
            self._remote_registry = new_registry

    # -- Cleanup thread -------------------------------------------------------

    def _cleanup_loop(self) -> None:
        """Periodic sweep: HDEL registry entries with zero global subscribers."""
        while not self._stop_event.wait(self._cleanup_interval):
            if self._stop_event.is_set():
                return
            self._cleanup_orphans()

    def _cleanup_orphans(self) -> None:
        if self._redis_pub is None:
            return
        try:
            data = self._redis_pub.hgetall(self._registry_key())
        except Exception as e:
            logger.warning("Cleanup HGETALL failed: %s", e)
            return
        if not data:
            return

        hashes = [
            h.decode("utf-8") if isinstance(h, bytes) else h for h in data.keys()
        ]
        channels = [self._notif_channel(h) for h in hashes]
        try:
            counts = self._redis_pub.pubsub_numsub(*channels)
        except Exception as e:
            logger.warning("Cleanup PUBSUB NUMSUB failed: %s", e)
            return

        # counts: list[(channel_bytes_or_str, int)]
        orphans: list[str] = []
        for (ch, count), h in zip(counts, hashes, strict=True):
            if count == 0:
                orphans.append(h)

        if not orphans:
            return
        try:
            self._redis_pub.hdel(self._registry_key(), *orphans)
            self._redis_pub.publish(self._control_channel(), b"")
            logger.debug("Cleaned %d orphan filter hashes", len(orphans))
        except Exception as e:
            logger.warning("Cleanup HDEL/publish failed: %s", e)

    # -- Called by RedisObserver on local subscribe/unsubscribe --------------

    def _dispatch_pubsub_cmd(self, action: str, channel: str, wait: bool = True) -> None:
        """Queue a sub/unsub command for the listener thread and optionally wait."""
        if self._control_cmd_queue is None:
            return
        ack = threading.Event() if wait else None
        try:
            self._control_cmd_queue.put_nowait((action, channel, ack))
        except queue.Full:
            logger.warning("pubsub cmd queue full; dropping %s(%s)", action, channel)
            return
        if ack is not None:
            # Bound the wait so a broken listener never permanently blocks a caller.
            ack.wait(timeout=1.0)

    def register_local_filter(self, f: Filter) -> None:
        """Called when a local subscription is created.

        Refcounts the filter hash. On first local sub for a given hash:
        - SUBSCRIBE the notif channel (so we receive remote publishes).
        - HSET the registry HASH.
        - PUBLISH the control channel so others refresh.
        """
        h = filter_hash(f)
        with self._state_lock:
            prior = self._local_hash_refcount.get(h, 0)
            self._local_hash_refcount[h] = prior + 1
            is_new = prior == 0

        if not is_new:
            return  # already SUBSCRIBEd, already HSET

        # SUBSCRIBE FIRST so we don't miss messages between our HSET and other
        # publishers routing to us. Delegated to listener thread; wait for ack.
        self._dispatch_pubsub_cmd("sub", self._notif_channel(h), wait=True)

        if self._redis_pub is not None:
            try:
                self._redis_pub.hset(
                    self._registry_key(), h, json.dumps(f.to_dict(), separators=(",", ":"))
                )
                self._redis_pub.publish(self._control_channel(), b"")
            except Exception as e:
                logger.error("Register HSET/publish for hash %s failed: %s", h, e)

    def unregister_local_filter(self, f: Filter) -> None:
        """Called when a local subscription is closed.

        Refcount down. On last local sub for that hash:
        - UNSUBSCRIBE the notif channel (fire-and-forget; no ack needed).

        HASH cleanup happens in the periodic sweep (checks PUBSUB NUMSUB
        across the whole cluster), not on close, so we don't race with
        other observers that might still be interested in this hash.
        """
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
        # Fire-and-forget: caller doesn't need to wait for the UNSUBSCRIBE
        # to hit redis; delivery filtering happens against the local
        # SubscriptionRegistry, which was already updated on close().
        self._dispatch_pubsub_cmd("unsub", self._notif_channel(h), wait=False)


class RedisObserver(Observer[Any]):
    """Observer with per-hash redis pub/sub delivery.

    On `subscribe()`: also registers the filter hash with the publisher so
    remote publishers can route to us. On `close()`: unregisters.
    """

    def __init__(
        self,
        codec: CodecProtocol[Any, Any],
        *,
        redis_url: str = "redis://localhost:6379",
        channel_prefix: str = "everyshape",
        queue_maxsize: int = DEFAULT_QUEUE_MAXSIZE,
        publish_queue_maxsize: int = DEFAULT_PUBLISH_QUEUE_MAXSIZE,
        cleanup_interval_seconds: float = DEFAULT_CLEANUP_INTERVAL_SECONDS,
        refresh_debounce_seconds: float = DEFAULT_REFRESH_DEBOUNCE_SECONDS,
        **_ignored_kwargs: Any,
    ) -> None:
        """Initialize with RedisPublisher.

        `**_ignored_kwargs` silently swallows legacy kwargs (e.g. `notify_self`)
        so older callers keep working through a deprecation window.
        """
        publisher = RedisPublisher(
            redis_url=redis_url,
            channel_prefix=channel_prefix,
            publish_queue_maxsize=publish_queue_maxsize,
            cleanup_interval_seconds=cleanup_interval_seconds,
            refresh_debounce_seconds=refresh_debounce_seconds,
        )
        super().__init__(codec=codec, publisher=publisher, queue_maxsize=queue_maxsize)
        self._redis_publisher: RedisPublisher = publisher

    def subscribe(self, options: SubscriptionOptions) -> Subscription:
        sub = super().subscribe(options)
        self._redis_publisher.register_local_filter(options.filter)
        return sub

    def _close_subscription(self, subscription: Subscription) -> None:
        # Grab filter BEFORE super clears anything on the subscription.
        f = subscription.options.filter
        super()._close_subscription(subscription)
        self._redis_publisher.unregister_local_filter(f)


if TYPE_CHECKING:
    from virtuals.tkv.observer import ObserverProtocol

    _: type[ObserverProtocol[Any]] = RedisObserver
    _p: type[PublisherProtocol] = RedisPublisher
