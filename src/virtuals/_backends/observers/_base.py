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

Exactly one dispatch worker ever runs callbacks. The worker owns the queue
and stop event it was started with (it never reads them off the observer),
so a worker that outlives a timed-out disconnect() can neither pick up the
next connect()'s queue nor be revived by it; connect() refuses while such a
worker is still inside a callback. disconnect() closes every subscription
before it waits, so the worker runs no receiver past the one it is in: once
disconnect() has returned no callback starts, and the old worker is alive
only for as long as that one callback. The worker holds the observer only
weakly between batches, so an observer dropped without disconnect() is
collected and its worker exits. An explicit disconnect() drops the wakes
still queued and closes every subscription (none survives it), so a reader
that reconnects subscribes again and re-reads; "resync on reconnect" in
the contract is about a transport reconnect inside a live observer.

Memory. A batch is at most `DISPATCH_BATCH_MAX` keys, so what one observer
holds is bounded by its dispatch queue, the batch in hand, and whatever the
transport parks on its own side (the Redis listener's overflow, at most one
more queue). A keyless resync wake is queued at most once per filter hash
until it is delivered, however many publishers or reconnects ask for one.
"""

from __future__ import annotations

import queue
import threading
import time
import weakref
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


def _hash_of(sub: Subscription) -> str:
    """``sub.filter_hash``, or a process-local stand-in for a filter without ``to_dict()``.

    Such a filter cannot be subscribed on redis (its hash is sent at
    subscribe), but on the in-memory transport it delivers keys, so its
    resync wake is keyed by the subscription itself. The stand-in never
    reaches a transport, and one subscription that cannot be hashed never
    costs the others their wake.
    """
    try:
        return sub.filter_hash
    except Exception:  # no to_dict(), or one that does not serialise
        return f"local:{id(sub):x}"


@dataclass(frozen=True, slots=True)
class _TaggedKey:
    """A key tagged with a filter hash for hash-scoped delivery."""

    key: Any  # tkv.Key at runtime; kept loose to avoid a circular import
    filter_hash: str


@dataclass(frozen=True, slots=True)
class _Resync:
    """A keyless resync wake for every local subscription on one filter hash."""

    filter_hash: str


logger = getLogger(__name__)


__all__ = [
    "DEFAULT_DISPATCH_QUEUE_MAXSIZE",
    "DISPATCH_BATCH_MAX",
    "ObserverBase",
]


# Default dispatch queue capacity: buffers incoming batches so transport
# listeners rarely block on slow subscribers. When full, they block.
DEFAULT_DISPATCH_QUEUE_MAXSIZE: int = 100_000

# Sentinel to unblock the dispatch worker on Queue.get() during disconnect.
_STOP_MARKER: object = object()

# How often a producer blocked on a full dispatch queue rechecks for shutdown.
_PUT_POLL_SECONDS: float = 0.1

# How long disconnect() waits for the worker to finish the callback it is in.
_WORKER_JOIN_SECONDS: float = 2.0

# Most keys the worker takes off the queue per delivery round. Bounds what is
# in hand (invisible to `qsize()`) and how long a stop waits between rounds.
DISPATCH_BATCH_MAX: int = 256


def _stop_worker(q: queue.Queue[Any], stop: threading.Event) -> None:
    """Ask a worker to exit. Runs from disconnect() and from the finalizer."""
    stop.set()
    try:
        q.put_nowait(_STOP_MARKER)
    except queue.Full:
        pass


def _deliver(ref: weakref.ref[ObserverBase], batch: list[Any]) -> bool:
    """Deliver one batch through the owner. False once the owner is gone.

    A BaseException out of a callback (SystemExit, KeyboardInterrupt, ...)
    is logged like any other callback failure: it must not end the only
    worker, which would silently stop every delivery.
    """
    if not batch:
        return True
    owner = ref()
    if owner is None:
        return False
    try:
        owner._deliver_batch(batch)
    except BaseException:
        logger.exception("Observer dispatch worker: callback raised; delivery continues")
    return True


def _dispatch_worker(
    ref: weakref.ref[ObserverBase], q: queue.Queue[Any], stop: threading.Event
) -> None:
    """Drain `q`: match each key + call bound callbacks, until `stop` or the marker.

    Module level so the thread holds its owner only through `ref`, and only
    while delivering: an observer dropped without disconnect() is collected
    and this worker exits at its next wake.
    """
    while not stop.is_set():
        try:
            item = q.get(timeout=_PUT_POLL_SECONDS)
        except queue.Empty:
            if ref() is None:
                return
            continue

        batch: list[Any] = []
        while True:
            if item is _STOP_MARKER:
                _deliver(ref, batch)
                return
            if isinstance(item, threading.Event):
                alive = _deliver(ref, batch)
                batch = []
                item.set()
                if not alive:
                    return
            else:
                batch.append(item)
                if len(batch) >= DISPATCH_BATCH_MAX:
                    break
            try:
                item = q.get_nowait()
            except queue.Empty:
                break

        if not _deliver(ref, batch):
            return


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
        self._worker_finalizer: weakref.finalize | None = None
        # Filter hashes with a keyless resync wake queued (or parked by the
        # transport) and not yet delivered: one wake per hash at a time.
        self._resync_queued: set[str] = set()
        self._resync_lock = threading.Lock()

    def _ensure_connected(self) -> None:
        if not self._connected:
            raise ObserverConnectionError("Observer not connected")

    def connect(self) -> None:
        """Init registry + dispatch worker, then call the subclass _on_connect hook.

        Raises ObserverConnectionError while the previous worker is still
        inside a callback (a disconnect() whose join timed out): there is
        never a second worker.
        """
        if self._connected:
            return
        old = self._dispatch_thread
        if old is not None and old.is_alive():
            raise ObserverConnectionError(
                f"Observer dispatch worker {old.name} from the previous connection is still "
                "inside a callback; connect again once it returns"
            )
        try:
            self._registry = SubscriptionRegistry()
            stop = self._stop_event = threading.Event()
            q = self._dispatch_queue = queue.Queue(maxsize=self._dispatch_queue_maxsize)
            with self._resync_lock:
                self._resync_queued = set()

            self._on_connect()

            self._dispatch_thread = threading.Thread(
                target=_dispatch_worker,
                args=(weakref.ref(self), q, stop),
                name=f"Observer-{id(self)}",
                daemon=True,
            )
            self._dispatch_thread.start()
            self._worker_finalizer = weakref.finalize(self, _stop_worker, q, stop)

            self._connected = True
        except Exception as e:
            self._abandon_connect()
            raise ObserverConnectionError(f"Failed to connect: {e}") from e

    def _abandon_connect(self) -> None:
        """Release whatever a failed connect() opened (connections, threads)."""
        try:
            self._on_disconnect()
        except Exception as e:
            logger.warning("Observer cleanup after a failed connect raised: %s", e)
        self._registry = None
        self._connected = False

    def disconnect(self) -> None:
        """Symmetric shutdown. _on_disconnect hook runs after the worker joins.

        Closes every subscription first, so the worker calls no receiver
        past the one it is in: nothing queued or in hand is delivered after
        this returns. Wakes still queued at an explicit disconnect are
        dropped with the subscriptions, which do not survive it (they report
        closed, bind raises); a subscription made after the next connect()
        re-reads, so nothing is missed silently. Then waits a bounded time
        for the worker to finish that callback. If it is still running after
        that, the worker is left to exit on its own (logged at WARNING) and
        the next connect() refuses until it has. Called from inside a
        callback, it does not wait for the worker (that is the caller): the
        worker exits once the callback returns.
        """
        if not self._connected:
            return
        try:
            _stop_worker(self._dispatch_queue, self._stop_event)
            self._close_all_subscriptions()
            if self._worker_finalizer is not None:
                self._worker_finalizer.detach()
                self._worker_finalizer = None

            t = self._dispatch_thread
            if t is not None and t is threading.current_thread():
                # Inside a callback: it exits right after this one returns.
                pass
            elif t is not None and t.is_alive():
                t.join(timeout=_WORKER_JOIN_SECONDS)
                if t.is_alive():
                    logger.warning(
                        "Observer dispatch worker %s is still inside a callback after %.1fs; "
                        "it exits when the callback returns",
                        t.name,
                        _WORKER_JOIN_SECONDS,
                    )
                else:
                    self._dispatch_thread = None
            else:
                self._dispatch_thread = None

            self._on_disconnect()
        finally:
            # Not connected first, then a last close pass: a subscribe() that
            # registered after the first pass sees one or the other.
            self._connected = False
            self._close_all_subscriptions()
            if self._registry is not None:
                self._registry.clear()
                self._registry = None

    def _close_all_subscriptions(self) -> None:
        """Mark every registered subscription closed, without the unsubscribe hooks.

        Not through close(): the hooks would queue unsubscribes on a
        connection that is going down anyway.
        """
        registry = self._registry
        if registry is None:
            return
        for sub in registry.subscriptions():
            sub._closed = True
            sub._receivers.clear()

    def subscribe(self, options: SubscriptionOptions) -> Subscription:
        """Register a subscription on the local registry and call _on_subscribe hook.

        Raises ObserverConnectionError when the observer is not connected,
        or disconnects while the subscription is being set up: the
        subscription then reports closed, never an open one that cannot fire.
        """
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
            sub._closed = True
            raise
        if not self._connected:
            registry.remove(sub)
            sub._closed = True
            sub._receivers.clear()
            raise ObserverConnectionError("Observer disconnected while subscribing")
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
        dropped silently. Backpressure then reaches the transport. If the
        observer isn't connected, or disconnects while blocked, the keys are
        dropped -- that is lifecycle, not load.

        `filter_hash` (optional) narrows matching at delivery time to subs
        whose canonical filter hashes to that value. Used by hash-routed
        transports (Redis pubsub) to prevent duplicate fanout when a key
        matches multiple registered filters.
        """
        if not self._connected:
            return
        for k in keys:
            if not self._enqueue(k if filter_hash is None else _TaggedKey(k, filter_hash)):
                return

    def _dispatch_resync(self, filter_hash: str) -> None:
        """Queue one keyless resync wake for every local sub on `filter_hash`.

        A transport calls this when it may have missed changes for that hash
        (after a reconnect, once the hash is routed again). Each matching
        subscription's receivers get `None` in place of a key, meaning
        "re-read the state you watch". Queued behind keys already received,
        so the wake follows them.
        """
        if self._connected:
            item = self._resync_item(filter_hash)
            if item is not None:
                self._enqueue(item)

    def _dispatch_resync_all(self) -> None:
        """Queue one keyless resync wake per filter hash with a local subscription.

        A transport calls this when a publisher on it dropped keys: nobody
        knows which filters they matched, so every subscription re-reads.
        """
        registry = self._registry
        if not self._connected or registry is None:
            return
        for h in {_hash_of(s) for s in registry.subscriptions()}:
            item = self._resync_item(h)
            if item is not None:
                self._enqueue(item)

    def _resync_item(self, filter_hash: str) -> _Resync | None:
        """A resync wake to queue for `filter_hash`, or None while one is pending.

        One keyless wake per hash is enough until it is delivered: it
        re-reads state as of its delivery, which is later than every drop
        or reconnect that asked for one in the meantime. So W publishers
        dropping in one stall cost each subscription one re-read, not W.
        """
        with self._resync_lock:
            if filter_hash in self._resync_queued:
                return None
            self._resync_queued.add(filter_hash)
        return _Resync(filter_hash)

    def _enqueue_nowait(self, item: object) -> bool:
        """Put `item` on the dispatch queue if there is room. False when full."""
        try:
            self._dispatch_queue.put_nowait(item)
            return True
        except queue.Full:
            return False

    def _enqueue(self, item: object) -> bool:
        """Put `item` on the dispatch queue, blocking while full.

        Returns False when the observer stopped while blocked (item dropped).
        The queue and stop event are those of the connection this call
        started under: a reconnect in the meantime ends the wait.
        """
        q = self._dispatch_queue
        stop = self._stop_event
        try:
            q.put_nowait(item)
            return True
        except queue.Full:
            pass
        while True:
            try:
                q.put(item, timeout=_PUT_POLL_SECONDS)
                return True
            except queue.Full:
                if stop.is_set() or not self._connected:
                    logger.warning("Observer stopping; dropping undelivered keys")
                    return False

    def flush(self, timeout: float = 1.0) -> None:
        """Wait up to `timeout` for everything queued so far to be delivered. Test helper.

        Returns at the timeout whether or not the queue drained, including
        when it is full and the marker cannot even be queued.
        """
        if not self._connected:
            return
        deadline = time.monotonic() + timeout
        done = threading.Event()
        try:
            self._dispatch_queue.put(done, timeout=timeout)
        except queue.Full:
            return
        done.wait(timeout=max(0.0, deadline - time.monotonic()))

    def _deliver_batch(self, batch: list[Key | _TaggedKey | _Resync]) -> None:
        if not batch:
            return
        # Every resync wake in hand is consumed here, delivered or not, so
        # its hash is released first: a later subscription on the same
        # filter gets its own wake. Released before any receiver runs, so a
        # drop recorded from now on queues a new wake.
        with self._resync_lock:
            for item in batch:
                if isinstance(item, _Resync):
                    self._resync_queued.discard(item.filter_hash)
        registry = self._registry
        if registry is None:
            return
        if len(registry) == 0:
            return
        notifications: list[tuple[Key | None, list[Subscription]]] = []
        for item in batch:
            if isinstance(item, _Resync):
                h = item.filter_hash
                woken = [s for s in registry.subscriptions() if _hash_of(s) == h]
                if woken:
                    notifications.append((None, woken))
                continue
            if isinstance(item, _TaggedKey):
                key = item.key
                h = item.filter_hash
                # tagged keys come from redis, where every subscription has a real hash
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
        """Hook called from disconnect() after the dispatch worker joins.

        Also called after a failed connect(), with whatever _on_connect
        managed to set up: it must tolerate a half-initialised state.
        """
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
