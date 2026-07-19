"""Redis publisher: routes keys to per-filter-hash pubsub channels.

Owns:
- The publish queue + drain worker (via PublisherBase) and a secondary
  publish thread that batches queue drains into redis pipelines.
- A cache of cluster-wide subscription interest (`_remote_registry`)
  seeded from HGETALL on the registry HASH and refreshed on control-
  channel signals (debounced).
- A tiny listener thread that subscribes ONLY to the control channel;
  every message triggers a debounced HGETALL refresh.

Does NOT own:
- SubscriptionRegistry, `_local_hash_refcount`, cleanup thread, HSET/
  HDEL of the registry HASH, or PUBLISH of the control channel. Those
  are the observer's responsibility now.

Wire format matches the pre-split fused code exactly so a new publisher
can talk to an old observer and vice versa during migration:
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

from virtuals.tkv.filter import Filter, filter_from_dict

from ._base import DEFAULT_QUEUE_MAXSIZE, PublisherBase


try:
    import redis
except ImportError as e:
    raise ImportError(
        "redis is required for RedisPublisher. Install with: pip install virtuals-py[redis]"
    ) from e

try:
    import msgpack
except ImportError as e:
    raise ImportError(
        "msgpack is required for RedisPublisher. Install with: pip install virtuals-py[redis]"
    ) from e


if TYPE_CHECKING:
    from virtuals.tkv.types import Key


logger = getLogger(__name__)


__all__ = [
    "RedisPublisher",
]


# Default publish-pipeline queue capacity. Batches accumulate here between drains.
DEFAULT_PUBLISH_QUEUE_MAXSIZE: int = 10_000

# Default HGETALL refresh debounce window (seconds). Coalesces bursts of
# "changed" signals from many observers subscribing at once.
DEFAULT_REFRESH_DEBOUNCE_SECONDS: float = 0.01

# Pubsub polling timeout (seconds).
_PUBSUB_POLL_TIMEOUT: float = 0.05


class RedisPublisher(PublisherBase):
    """Publisher that routes writes onto per-filter-hash Redis pubsub channels."""

    def __init__(
        self,
        *,
        redis_url: str = "redis://localhost:6379",
        channel_prefix: str = "everyshape",
        queue_maxsize: int = DEFAULT_QUEUE_MAXSIZE,
        publish_queue_maxsize: int = DEFAULT_PUBLISH_QUEUE_MAXSIZE,
        refresh_debounce_seconds: float = DEFAULT_REFRESH_DEBOUNCE_SECONDS,
    ) -> None:
        super().__init__(queue_maxsize=queue_maxsize)
        self._redis_url = redis_url
        self._channel_prefix = channel_prefix
        # Instance id: kept in the wire payload for backwards-compat with the
        # OLD fused RedisObserver (which used it for self-echo suppression).
        # The NEW split RedisObserver ignores it.
        self._instance_id = id(self)
        self._publish_queue_maxsize = publish_queue_maxsize
        self._refresh_debounce = refresh_debounce_seconds

        # Redis connections (created on connect)
        self._redis_pub: Any = None  # publish + HGETALL for cache
        self._redis_sub: Any = None  # dedicated connection for control-channel pubsub
        self._pubsub: Any = None

        # Cluster-wide interest cache: filter_hash -> reconstructed Filter.
        self._remote_registry: dict[str, Filter] = {}
        self._state_lock = threading.Lock()

        # Threads and queues
        self._pub_stop_event = threading.Event()
        self._publish_queue: queue.Queue[tuple[str, bytes] | None] | None = None
        self._publish_thread: threading.Thread | None = None
        self._control_listener_thread: threading.Thread | None = None
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

    # -- Lifecycle hooks ------------------------------------------------------

    def _on_connect(self) -> None:
        """Open Redis connections, seed cache, start pipeline + listener threads."""
        self._redis_pub = redis.from_url(self._redis_url)
        self._redis_sub = redis.from_url(self._redis_url)
        self._redis_pub.ping()

        self._pubsub = self._redis_sub.pubsub()
        # Subscribe to control channel FIRST so we don't miss changes during HGETALL.
        self._pubsub.subscribe(self._control_channel())

        # Seed remote registry from current HASH state.
        self._refresh_remote_registry()

        self._pub_stop_event.clear()
        self._publish_queue = queue.Queue(maxsize=self._publish_queue_maxsize)

        self._publish_thread = threading.Thread(
            target=self._publish_loop,
            name=f"RedisPubPipeline-{self._instance_id}",
            daemon=True,
        )
        self._control_listener_thread = threading.Thread(
            target=self._control_listener_loop,
            name=f"RedisPubListener-{self._instance_id}",
            daemon=True,
        )
        self._publish_thread.start()
        self._control_listener_thread.start()

        logger.info(
            "Redis publisher started at %s (prefix=%s, iid=%d)",
            self._redis_url,
            self._channel_prefix,
            self._instance_id,
        )

    def flush(self, timeout: float = 1.0) -> None:
        """Drain the write queue AND the publish pipeline queue.

        The base flush only guarantees `_worker` processed everything in
        `self._queue` (routing done, items pushed onto `_publish_queue`).
        We then wait for the Redis pipeline thread to drain
        `_publish_queue` so the pipeline execute has actually issued to
        Redis by the time this returns. Callers still need to allow
        pubsub fanout time before checking subscriber callbacks.
        """
        import time as _time

        super().flush(timeout=timeout)
        pq = self._publish_queue
        if pq is None:
            return
        end = _time.monotonic() + timeout
        while _time.monotonic() < end:
            if pq.qsize() == 0:
                break
            _time.sleep(0.01)

    def _on_disconnect(self) -> None:
        """Signal helper threads, cancel timers, close Redis connections."""
        self._pub_stop_event.set()

        with self._refresh_timer_lock:
            if self._refresh_timer is not None:
                self._refresh_timer.cancel()
                self._refresh_timer = None

        if self._publish_queue is not None:
            try:
                self._publish_queue.put_nowait(None)  # stop sentinel
            except queue.Full:
                pass

        for t in (self._publish_thread, self._control_listener_thread):
            if t is not None and t.is_alive():
                t.join(timeout=2.0)
        self._publish_thread = None
        self._control_listener_thread = None

        if self._pubsub is not None:
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
        logger.info("Redis publisher stopped")

    # -- Publish routing ------------------------------------------------------

    def _publish_batch(self, batch: list[Key]) -> None:
        """Bucket keys per matching filter hash and enqueue for the pipeline thread."""
        if not batch or self._publish_queue is None:
            return

        # Snapshot remote registry so routing doesn't hold the state lock.
        with self._state_lock:
            remote = dict(self._remote_registry)

        if not remote:
            return  # nobody in the cluster is interested; publish nothing

        per_hash: dict[str, list[Key]] = {}
        for key in batch:
            for h, f in remote.items():
                if f.matches(key):
                    per_hash.setdefault(h, []).append(key)

        if not per_hash:
            return

        for h, ks in per_hash.items():
            payload = msgpack.packb(
                {
                    # `iid` stays in the payload for backwards-compat with the
                    # OLD fused RedisObserver during migration. The new
                    # RedisObserver ignores it.
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

    # -- Publish pipeline thread ---------------------------------------------

    def _publish_loop(self) -> None:
        """Drain publish queue, batch into a redis pipeline, execute."""
        while not self._pub_stop_event.is_set():
            try:
                item = self._publish_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                return

            batch: list[tuple[str, bytes]] = [item]
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

    # -- Control-channel listener --------------------------------------------

    def _control_listener_loop(self) -> None:
        """Poll the control channel; every message schedules a debounced refresh."""
        control_channel = self._control_channel()
        while not self._pub_stop_event.is_set():
            try:
                msg = self._pubsub.get_message(timeout=_PUBSUB_POLL_TIMEOUT)
            except Exception as e:
                logger.warning("Publisher listener get_message failed: %s", e)
                self._pub_stop_event.wait(0.5)
                continue

            if msg is None or msg.get("type") not in ("message", "pmessage"):
                continue

            channel = msg["channel"]
            if isinstance(channel, bytes):
                channel = channel.decode("utf-8")

            if channel == control_channel:
                self._schedule_refresh()

    # -- Debounced refresh ---------------------------------------------------

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
        if not self._pub_stop_event.is_set():
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


if TYPE_CHECKING:
    from virtuals.tkv.publisher import PublisherProtocol

    _p: type[PublisherProtocol] = RedisPublisher
