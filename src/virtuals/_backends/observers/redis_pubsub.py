"""Redis observer: listens to per-filter-hash pubsub channels.

Owns:
- `_local_hash_refcount`: per-hash refcount so we only SUBSCRIBE once
  regardless of how many local subs share a filter shape.
- Registration of each local filter hash in the server-side routing index
  (`nu1_reg`, see `_backends/notify`), done only after the server confirmed
  the SUBSCRIBE. A publisher routes a key to a hash only if the index has it,
  so a subscription is live (no window) by the time `subscribe()` returns.
- The listener thread that owns `self._pubsub` and hands inbound notif
  messages to the dispatch queue. It sleeps in one `select()` on the pubsub
  socket and a local wake socketpair, so a sub/unsub queued from another
  thread is sent at once, not on a timer.
- A maintenance thread that re-runs `nu1_reg` for every local hash the server
  resubscribed after a reconnect (valkey restart included: the loader then
  reloads the library).

Reconnect window. When the pubsub connection drops, every live hash loses its
readiness until it is resubscribed and registered again: a same-filter
subscribe() in that window waits for the re-registration (or raises on
timeout) instead of returning early. Writes committed while the connection
is down, or before the re-registration lands, are NOT delivered (a restarted
server lost the index; a dropped connection misses the PUBLISH). The observer
logs a WARNING when this starts and an INFO per re-registered hash. Once a
hash is re-registered, every local subscription on it gets one keyless resync
wake (its receivers are called with None), so a level flow re-reads and
recovers the writes it missed. Event flows must treat None as a re-read or
skip it. A publisher that had to drop keys sends the same keyless wake to
every hash on the prefix (an empty-keys payload) once it can publish again.

Backlog. The listener never blocks on the dispatch queue. When the queue is
full it parks inbound keys in its own overflow buffer (bounded by the same
capacity, so at most twice the queue is buffered) and keeps serving
sub/unsub commands and reading confirmations, so a slow subscriber never
stalls a subscribe() from another thread or from inside a callback. Only
once the overflow is full too does it stop reading the socket (backpressure
into valkey); a SUBSCRIBE confirmation sent in that state is read when the
backlog clears, so a subscribe() can time out. Subscribe from another thread
when callbacks can face that much backlog.

Timeouts. Both connections have a socket and connect timeout and at most one
transparent redis-py retry, so a stalled server bounds `nu1_reg` (and with it
subscribe()) at about twice the socket timeout past the subscribe timeout.

Unsubscribe is a fire-and-forget UNSUBSCRIBE. The index entry is removed
lazily by the router: a PUBLISH that reaches 0 subscribers unregisters it.
That is safe because registration always follows a confirmed SUBSCRIBE.

Threads hold the observer only weakly between steps and connections are
closed by a finalizer, so an observer dropped without disconnect() is
collected and leaves no thread or server client behind.

Does NOT own the publish path -- that lives in RedisPublisher.

Self-echo: per the split design, this observer does NOT suppress
messages whose `iid` matches.

Wire format:
- Channel: `{prefix}:notif:{filter_hash}`.
- msgpack payload: `{"iid": pub_id, "keys": [[...], ...]}`. Empty `keys`
  is a keyless resync wake for the hash.
- Routing index: `{prefix}:nx1:*` (owned by the Lua library).
Mixing with observers/publishers from before server-side routing on one
prefix is unsupported.
"""

from __future__ import annotations

import queue
import select
import socket
import threading
import time
import weakref
from collections import deque
from logging import getLogger
from typing import TYPE_CHECKING, Any

from virtuals._backends.notify.keyenc import hint_entries
from virtuals._backends.notify.router import NotifyLua
from virtuals.tkv.filter import Filter, canonicalize, filter_hash
from virtuals.tkv.observer import ObserverSubscriptionError

from ._base import DEFAULT_DISPATCH_QUEUE_MAXSIZE, ObserverBase, _Resync, _TaggedKey


try:
    import redis
    from redis.backoff import NoBackoff
    from redis.retry import Retry
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
    from virtuals._backends.notify.keyenc import HintEntry
    from virtuals.tkv.observer import Subscription


logger = getLogger(__name__)


__all__ = [
    "DEFAULT_SOCKET_TIMEOUT_SECONDS",
    "DEFAULT_SUBSCRIBE_TIMEOUT_SECONDS",
    "RedisObserver",
]


# Longest subscribe() waits for the server's SUBSCRIBE confirmation and the
# routing registration (seconds). On timeout subscribe() raises.
DEFAULT_SUBSCRIBE_TIMEOUT_SECONDS: float = 2.0

# Socket read/write and connect timeout on both connections (seconds). A
# stalled server fails a call after this instead of hanging it.
DEFAULT_SOCKET_TIMEOUT_SECONDS: float = 2.0

# Longest the listener sleeps with nothing to do (seconds). Every real event,
# an inbound message or a queued command, wakes it sooner; this only bounds a
# missed wake.
_LISTENER_IDLE_SECONDS: float = 1.0

# How often the listener retries moving parked keys into a full dispatch queue.
_OVERFLOW_POLL_SECONDS: float = 0.02

# Extra time a same-filter waiter allows past the first registrant's deadline,
# which covers the first registrant's `nu1_reg` after a late confirmation.
_WAITER_MARGIN_SECONDS: float = 1.0

# Re-registration retry backoff after a reconnect (seconds).
_REREG_BACKOFF_MIN: float = 0.05
_REREG_BACKOFF_MAX: float = 1.0

# How long disconnect() waits for the listener and maintenance threads.
_THREAD_JOIN_SECONDS: float = 2.0


class _Ack:
    """One waiter's view of a queued pubsub command."""

    __slots__ = ("error", "event", "gen")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.error: BaseException | None = None
        # Connection generation the confirmation was read on (set by the listener).
        self.gen: int = -1

    def fail(self, error: BaseException) -> None:
        self.error = error
        self.event.set()


class _Ready:
    """Readiness of one filter hash: set once SUBSCRIBE is confirmed and registered.

    `deadline` (monotonic) is the first registrant's own deadline, shared by
    same-filter waiters; None for a readiness reset by a reconnect.
    """

    __slots__ = ("deadline", "error", "event")

    def __init__(self, deadline: float | None = None) -> None:
        self.event = threading.Event()
        self.error: BaseException | None = None
        self.deadline = deadline


def _close_quietly(what: str, obj: object) -> None:
    if obj is None:
        return
    try:
        obj.close()  # type: ignore[attr-defined]
    except Exception as e:
        logger.warning("Error closing %s: %s", what, e)


def _release_connection(
    stop: threading.Event,
    wake_w: socket.socket | None,
    pubsub: object,
    redis_sub: object,
    redis_: object,
    unsubscribe: bool = False,
) -> None:
    """Stop the threads and close everything one connect() opened.

    Module level and free of `self`: it runs from disconnect() and, for an
    observer dropped without disconnect(), from its finalizer. The finalizer
    does no round trip (closing the socket unsubscribes server side); only
    disconnect() sends the UNSUBSCRIBE.
    """
    stop.set()
    if wake_w is not None:
        try:
            wake_w.send(b"\0")
        except (BlockingIOError, OSError):
            pass
    if pubsub is not None and unsubscribe:
        try:
            pubsub.unsubscribe()  # type: ignore[attr-defined]
        except Exception as e:
            logger.warning("Error unsubscribing pubsub: %s", e)
    _close_quietly("pubsub", pubsub)
    _close_quietly("_redis_sub", redis_sub)
    _close_quietly("_redis", redis_)


# What a listener step asks the thread to wait on next: (wake socket, pubsub
# socket or None, timeout). None means loop again at once.
_Wait = "tuple[socket.socket, socket.socket | None, float] | None"


def _wait_for_work(wake: socket.socket, sock: socket.socket | None, timeout: float) -> None:
    """Sleep until `sock` is readable or `wake` rings, at most `timeout` seconds.

    Module level so the listener thread holds no reference to its observer
    while it sleeps, which is nearly all the time.
    """
    watched = [wake] if sock is None else [wake, sock]
    try:
        ready, _, _ = select.select(watched, [], [], timeout)
    except (OSError, ValueError):
        # a socket closed under us (reconnect or shutdown); the next step sees it
        time.sleep(min(timeout, _OVERFLOW_POLL_SECONDS))
        return
    if wake in ready:
        try:
            while wake.recv(4096):
                pass
        except (BlockingIOError, OSError):
            pass


def _listener_main(ref: weakref.ref[RedisObserver], stop: threading.Event) -> None:
    """Listener thread body: one `_listener_step` per iteration, owner held weakly.

    The observer is held only inside the step. The wait between steps runs
    here without it, so an observer dropped without disconnect() can be
    collected while its listener sleeps.
    """
    while not stop.is_set():
        obs = ref()
        if obs is None:
            return
        try:
            wait = obs._listener_step(stop)
        except Exception:
            logger.exception("Redis observer listener step failed; continuing")
            wait = None
            stop.wait(0.5)
        del obs
        if wait is False:
            return
        if wait is not None:
            _wait_for_work(*wait)


def _maint_main(
    ref: weakref.ref[RedisObserver],
    stop: threading.Event,
    rereg_q: queue.Queue[tuple[str, int] | None],
) -> None:
    """Maintenance thread body: re-register hashes the server resubscribed.

    `nu1_reg` is idempotent, and the loader reloads the library when the
    server lost it (restart, FUNCTION FLUSH). Retries until the hash is
    registered, unsubscribed locally, or the observer stops.
    """
    while not stop.is_set():
        try:
            item = rereg_q.get(timeout=_LISTENER_IDLE_SECONDS)
        except queue.Empty:
            if ref() is None:
                return
            continue
        if item is None:
            return
        h, gen = item
        backoff = _REREG_BACKOFF_MIN
        while not stop.is_set():
            obs = ref()
            if obs is None:
                return
            done = obs._reregister_once(h, gen)
            del obs
            if done:
                break
            if stop.wait(backoff):
                return
            backoff = min(backoff * 2, _REREG_BACKOFF_MAX)


class RedisObserver(ObserverBase):
    """Observer that receives per-hash routed notifications over Redis pubsub."""

    def __init__(
        self,
        *,
        redis_url: str = "redis://localhost:6379",
        channel_prefix: str = "nu",
        subscribe_timeout: float = DEFAULT_SUBSCRIBE_TIMEOUT_SECONDS,
        socket_timeout: float = DEFAULT_SOCKET_TIMEOUT_SECONDS,
        dispatch_queue_maxsize: int = DEFAULT_DISPATCH_QUEUE_MAXSIZE,
    ) -> None:
        super().__init__(dispatch_queue_maxsize=dispatch_queue_maxsize)
        self._redis_url = redis_url
        self._channel_prefix = channel_prefix
        self._prefix_b = channel_prefix.encode()
        self._notif_prefix_b = f"{channel_prefix}:notif:".encode()
        self._instance_id = id(self)
        self._subscribe_timeout = subscribe_timeout
        self._socket_timeout = socket_timeout

        self._redis: Any = None  # nu1_reg calls
        self._redis_sub: Any = None  # dedicated pubsub connection
        self._pubsub: Any = None
        self._lua = NotifyLua()
        # NOTE: `self._pubsub`, `self._pending` and `self._overflow` are
        # touched EXCLUSIVELY by the listener thread. Sub/unsub from other
        # threads goes via `_control_cmd_queue`.

        # Guarded by _state_lock.
        self._local_hash_refcount: dict[str, int] = {}
        self._local: dict[str, list[HintEntry]] = {}  # h -> routing entries
        self._ready: dict[str, _Ready] = {}
        # Bumped on every pubsub connection loss and reconnect. A registration
        # only marks a hash ready when no reconnect happened since it started.
        self._conn_gen: int = 0
        self._state_lock = threading.Lock()

        # channel (bytes) -> one group of acks per SUBSCRIBE sent, in send order.
        # The server confirms each SUBSCRIBE once, in order, so a confirmation
        # pops exactly the oldest group.
        self._pending: dict[bytes, deque[list[_Ack]]] = {}
        self._watched_conn: Any = None
        # Inbound items the dispatch queue had no room for (listener-owned).
        self._overflow: deque[_TaggedKey | _Resync] = deque()

        self._obs_stop_event = threading.Event()
        self._listener_thread: threading.Thread | None = None
        self._maint_thread: threading.Thread | None = None
        self._rereg_q: queue.Queue[tuple[str, int] | None] = queue.Queue()
        # (action, channel, ack). action in {'sub', 'unsub', 'shutdown'}.
        self._control_cmd_queue: queue.Queue[tuple[str, str | None, _Ack | None]] | None = None
        # Local socketpair: a byte on `_wake_w` wakes the listener's select().
        self._wake_r: socket.socket | None = None
        self._wake_w: socket.socket | None = None
        self._conn_finalizer: weakref.finalize | None = None

    # -- Channel helpers ------------------------------------------------------

    def _notif_channel(self, h: str) -> str:
        return f"{self._channel_prefix}:notif:{h}"

    def _open_client(self) -> redis.Redis:
        return redis.from_url(
            self._redis_url,
            socket_timeout=self._socket_timeout,
            socket_connect_timeout=self._socket_timeout,
            retry=Retry(NoBackoff(), 1),
        )

    # -- Lifecycle hooks ------------------------------------------------------

    def _on_connect(self) -> None:
        """Open connections + pubsub, load the routing library, start threads."""
        for t in (self._listener_thread, self._maint_thread):
            if t is not None and t.is_alive():
                raise RuntimeError(f"thread {t.name} from the previous connection is still running")
        self._redis = self._open_client()
        self._redis_sub = self._open_client()
        self._redis.ping()
        self._lua.ensure(self._redis)

        self._pubsub = self._redis_sub.pubsub()
        self._pending = {}
        self._overflow = deque()
        self._watched_conn = None

        stop = self._obs_stop_event = threading.Event()
        self._control_cmd_queue = queue.Queue()
        rereg_q = self._rereg_q = queue.Queue()
        self._wake_r, self._wake_w = socket.socketpair()
        self._wake_r.setblocking(False)
        self._wake_w.setblocking(False)
        self._conn_finalizer = weakref.finalize(
            self,
            _release_connection,
            stop,
            self._wake_w,
            self._pubsub,
            self._redis_sub,
            self._redis,
        )

        ref = weakref.ref(self)
        self._listener_thread = threading.Thread(
            target=_listener_main,
            args=(ref, stop),
            name=f"RedisObserverListener-{self._instance_id}",
            daemon=True,
        )
        self._maint_thread = threading.Thread(
            target=_maint_main,
            args=(ref, stop, rereg_q),
            name=f"RedisObserverMaint-{self._instance_id}",
            daemon=True,
        )
        self._listener_thread.start()
        self._maint_thread.start()

        logger.info(
            "Redis observer started at %s (prefix=%s, iid=%d, mode=%s)",
            self._redis_url,
            self._channel_prefix,
            self._instance_id,
            self._lua.mode,
        )

    def _on_disconnect(self) -> None:
        """Signal threads, wait for them to exit, close connections.

        Tolerates a half-initialised state (a failed connect) so nothing it
        did open is leaked.
        """
        self._obs_stop_event.set()

        if self._control_cmd_queue is not None:
            self._control_cmd_queue.put(("shutdown", None, None))
            # Closed subscriptions after this point queue nothing (F12).
            self._control_cmd_queue = None
        self._rereg_q.put(None)
        self._wake_listener()

        for attr in ("_listener_thread", "_maint_thread"):
            t = getattr(self, attr)
            if t is not None and t.is_alive():
                t.join(timeout=_THREAD_JOIN_SECONDS)
                if t.is_alive():
                    # Kept: the next connect() refuses while it is alive.
                    logger.warning(
                        "Redis observer thread %s still running after %.1fs; it exits on its own",
                        t.name,
                        _THREAD_JOIN_SECONDS,
                    )
                    continue
            setattr(self, attr, None)

        if self._conn_finalizer is not None:
            self._conn_finalizer.detach()
            self._conn_finalizer = None
        _release_connection(
            self._obs_stop_event, None, self._pubsub, self._redis_sub, self._redis, unsubscribe=True
        )
        self._pubsub = None
        self._redis_sub = None
        self._redis = None

        for sock in (self._wake_r, self._wake_w):
            if sock is not None:
                sock.close()
        self._wake_r = None
        self._wake_w = None

        if self._overflow:
            logger.warning(
                "Redis observer stopping; dropping %d undelivered keys", len(self._overflow)
            )
            self._overflow.clear()

        with self._state_lock:
            self._local_hash_refcount = {}
            self._local = {}
            for r in self._ready.values():
                if not r.event.is_set():
                    r.error = ObserverSubscriptionError("Observer disconnected")
                    r.event.set()
            self._ready = {}
        logger.info("Redis observer stopped")

    # -- Subscribe / unsubscribe hooks ---------------------------------------

    def _on_subscribe(self, subscription: Subscription) -> None:
        """First local sub for a hash -> confirmed SUBSCRIBE, then nu1_reg."""
        self._register_local_filter(subscription.filter)

    def _on_unsubscribe(self, subscription: Subscription) -> None:
        """Last local sub for a hash -> UNSUBSCRIBE. Index removal is lazy (Lua)."""
        self._unregister_local_filter(subscription.filter)

    def _register_local_filter(self, f: Filter) -> None:
        """Make hash(f) live: return only once routing to this observer is guaranteed.

        The first local registrant of a hash SUBSCRIBEs, waits for the server's
        confirmation, then registers the routing hints. Concurrent registrants
        of the same hash wait for that readiness, up to the first registrant's
        deadline plus a margin (so they never give up while it succeeds). After
        a pubsub reconnect they wait for the re-registration instead. Any
        failure or timeout raises ObserverSubscriptionError in every waiter and
        rolls the hash back.
        """
        h = filter_hash(f)
        timeout = self._subscribe_timeout
        now = time.monotonic()
        with self._state_lock:
            prior = self._local_hash_refcount.get(h, 0)
            self._local_hash_refcount[h] = prior + 1
            if prior == 0:
                ready = self._ready[h] = _Ready(deadline=now + timeout)
            else:
                ready = self._ready[h]

        if prior:
            wait_until = now + timeout
            if ready.deadline is not None:
                wait_until = max(wait_until, ready.deadline + _WAITER_MARGIN_SECONDS)
            if not ready.event.wait(max(0.0, wait_until - time.monotonic())):
                self._release(h, ready)
                raise ObserverSubscriptionError(
                    f"Subscription for hash {h} not ready within {wait_until - now:.2f}s"
                )
            if ready.error is not None:
                # The first registrant already rolled the hash back.
                raise ObserverSubscriptionError(
                    f"Subscription for hash {h} failed: {ready.error}"
                ) from ready.error
            return

        deadline = now + timeout
        subscribed = False
        try:
            entries = hint_entries(canonicalize(f).routing_hints())
            if entries:  # empty: matches nothing, stays local only
                with self._state_lock:
                    self._local[h] = entries
                channel = self._notif_channel(h)
                ack = self._dispatch_pubsub_cmd("sub", channel, wait=True)
                subscribed = True
                if ack is None or not ack.event.wait(max(0.0, deadline - time.monotonic())):
                    raise ObserverSubscriptionError(
                        f"SUBSCRIBE {channel} not confirmed within {timeout}s"
                    )
                if ack.error is not None:
                    raise ObserverSubscriptionError(
                        f"SUBSCRIBE {channel} failed: {ack.error}"
                    ) from ack.error
                self._register_hash(h, entries)
                with self._state_lock:
                    # A loss or reconnect since the confirmation was read?
                    stale = self._conn_gen != ack.gen
                    if not stale:
                        ready.event.set()
                # A reconnect raced the registration: the index may have lost
                # it. The resubscribe confirmation re-registers and sets
                # `ready` (still this object); wait for that.
                if stale and not ready.event.wait(
                    max(0.0, deadline + _WAITER_MARGIN_SECONDS - time.monotonic())
                ):
                    raise ObserverSubscriptionError(
                        f"Subscription for hash {h} not re-registered after a reconnect"
                    )
        except BaseException as e:
            with self._state_lock:
                if self._ready.get(h) is ready:
                    del self._ready[h]
                    self._local_hash_refcount.pop(h, None)
                    self._local.pop(h, None)
                    if subscribed:
                        # Queued under the lock, so a later first registrant's
                        # SUBSCRIBE goes out after this UNSUBSCRIBE.
                        self._dispatch_pubsub_cmd("unsub", self._notif_channel(h), wait=False)
                ready.error = e
                ready.event.set()
            if isinstance(e, ObserverSubscriptionError):
                raise
            if isinstance(e, Exception):
                raise ObserverSubscriptionError(f"Subscription for hash {h} failed: {e}") from e
            raise
        ready.event.set()

    def _release(self, h: str, ready: _Ready) -> None:
        """Drop one waiter's refcount after it gave up waiting."""
        with self._state_lock:
            if self._ready.get(h) is not ready:
                return
            n = self._local_hash_refcount.get(h, 0) - 1
            if n > 0:
                self._local_hash_refcount[h] = n
            # n == 0 cannot happen while the first registrant is in flight.

    def _register_hash(self, h: str, entries: list[HintEntry]) -> None:
        args: list[bytes | str] = [self._prefix_b, h]
        for e in entries:
            args.extend(e)
        self._lua.call(self._redis, "nu1_reg", args)

    def _unregister_local_filter(self, f: Filter) -> None:
        h = filter_hash(f)
        with self._state_lock:
            prior = self._local_hash_refcount.get(h, 0)
            if prior > 1:
                self._local_hash_refcount[h] = prior - 1
                return
            self._local_hash_refcount.pop(h, None)
            self._ready.pop(h, None)
            entries = self._local.pop(h, None)
            if entries:
                # Fire-and-forget; the routing index entry is removed lazily.
                self._dispatch_pubsub_cmd("unsub", self._notif_channel(h), wait=False)

    def _dispatch_pubsub_cmd(self, action: str, channel: str, wait: bool = True) -> _Ack | None:
        """Queue a sub/unsub for the listener thread. Returns the ack when `wait`.

        A "sub" ack is set when the server confirms the SUBSCRIBE (or fails
        with the send error). The caller waits on it. Nothing is queued
        while disconnected.
        """
        q = self._control_cmd_queue
        if q is None:
            return None
        ack = _Ack() if wait else None
        q.put((action, channel, ack))
        self._wake_listener()
        return ack

    def _wake_listener(self) -> None:
        """Ring the listener's wake socket. A full buffer means a wake is already pending."""
        try:
            if self._wake_w is not None:
                self._wake_w.send(b"\0")
        except (BlockingIOError, OSError):
            pass

    # -- Listener thread ------------------------------------------------------

    def _wait_spec(self, watch_pubsub: bool, timeout: float) -> _Wait:
        """What to sleep on after a step found nothing to do.

        Called only once redis-py's own buffer is drained (`get_message`
        returned None), so a readable socket is the only way a message can be
        waiting. With nothing subscribed `get_message` reads nothing, so only
        the wake socket is watched; same when the caller has no room for more
        inbound keys. After a drop there is no socket: loop at once so
        `get_message` gets to reconnect.
        """
        wake = self._wake_r
        if wake is None:
            return None
        sock = None
        conn = self._pubsub.connection
        if watch_pubsub and conn is not None and self._pubsub.subscribed:
            sock = conn._sock  # redis-py has no public accessor
            if sock is None:
                return None
        return (wake, sock, timeout)

    def _watch_reconnects(self) -> None:
        """Hook our reconnect callback onto the pubsub connection once it exists."""
        conn = self._pubsub.connection
        if conn is not None and conn is not self._watched_conn:
            conn.register_connect_callback(self._on_pubsub_reconnect)
            self._watched_conn = conn

    def _on_pubsub_reconnect(self, _connection: object) -> None:
        """Runs on the listener thread, after redis-py resubscribed every channel.

        SUBSCRIBEs sent on the dead connection may never be confirmed; the
        resubscribe confirms each live channel exactly once. Collapse each
        channel's pending groups into one so that confirmation releases them
        all, and drop groups for channels no longer subscribed.
        """
        self._invalidate_ready("reconnected")
        channels = self._pubsub.channels
        for ch in list(self._pending):
            groups = self._pending.pop(ch)
            if ch in channels:
                self._pending[ch] = deque([[a for g in groups for a in g]])

    def _invalidate_ready(self, why: str) -> None:
        """Pubsub connection lost or replaced: no local hash is known to be routed.

        Swaps each registered hash's readiness for a fresh unset one, so a
        same-filter subscribe() waits for the re-registration. A hash whose
        first registrant is still in flight keeps its readiness: that
        registrant notices the generation change and waits itself.
        """
        reset = 0
        with self._state_lock:
            self._conn_gen += 1
            for h in self._local:
                r = self._ready.get(h)
                if r is not None and r.event.is_set() and r.error is None:
                    self._ready[h] = _Ready()
                    reset += 1
        if reset:
            logger.warning(
                "Redis observer pubsub connection %s: %d subscribed filter hash(es) not "
                "routed until re-registered; writes in this window are not delivered, "
                "subscribers get a keyless resync wake once re-registered",
                why,
                reset,
            )

    def _run_cmd(self, action: str, channel: str | None, ack: _Ack | None) -> bool:
        """Run one queued command. Returns True on shutdown."""
        if action == "shutdown":
            return True
        if channel is None:
            return False
        try:
            if action == "sub":
                self._pubsub.subscribe(channel)
                self._watch_reconnects()
                if ack is not None:
                    self._pending.setdefault(channel.encode(), deque()).append([ack])
                    ack = None
            elif action == "unsub":
                self._pubsub.unsubscribe(channel)
            elif action == "resync":
                # Behind every key parked so far, so the wake follows them.
                self._overflow.append(_Resync(channel))
                self._drain_overflow()
        except Exception as e:
            logger.warning("Listener cmd %s(%s) failed: %s", action, channel, e)
            if ack is not None:
                ack.fail(e)
                ack = None
        if ack is not None:
            ack.event.set()
        return False

    def _on_subscribed(self, channel: bytes) -> None:
        """Server confirmed a SUBSCRIBE: release its waiters, or re-register."""
        notif_prefix = self._notif_prefix_b
        if not channel.startswith(notif_prefix):
            return
        h = channel[len(notif_prefix) :].decode()
        groups = self._pending.get(channel)
        if groups:
            with self._state_lock:
                gen = self._conn_gen
            for a in groups.popleft():
                a.gen = gen
                a.event.set()
            if groups:
                return
            del self._pending[channel]
            with self._state_lock:
                live = h in self._local
            # A stale group (its registrant timed out and rolled back, and
            # the rollback UNSUBSCRIBE was lost in a reconnect) ate this
            # confirmation: unsubscribe the orphan channel.
            if not live:
                self._unsubscribe_orphan(channel)
            return
        # Nobody waits: a resubscribe after reconnect.
        with self._state_lock:
            live = h in self._local
            gen = self._conn_gen
        if live:
            self._rereg_q.put((h, gen))
        else:
            # redis-py resubscribed a channel whose UNSUBSCRIBE was lost.
            self._unsubscribe_orphan(channel)

    def _unsubscribe_orphan(self, channel: bytes) -> None:
        """UNSUBSCRIBE a channel with no local hash, unless one is already in flight."""
        if channel not in self._pubsub.channels:
            return
        if channel in self._pubsub.pending_unsubscribe_channels:
            return
        try:
            self._pubsub.unsubscribe(channel)
        except Exception as e:
            logger.warning("Listener unsubscribe(%s) failed: %s", channel, e)

    def _drain_overflow(self, wait: float = 0.0) -> None:
        """Move parked inbound items into the dispatch queue while it has room.

        With `wait`, the first item may wait that long for room (the put
        returns as soon as the worker frees a slot), so a backlog drains at
        the worker's pace, not on a timer.
        """
        overflow = self._overflow
        if not overflow:
            return
        if wait > 0.0:
            if not self._enqueue_nowait(overflow[0]):
                try:
                    self._dispatch_queue.put(overflow[0], timeout=wait)
                except queue.Full:
                    return
            overflow.popleft()
        while overflow:
            if not self._enqueue_nowait(overflow[0]):
                return
            overflow.popleft()

    def _listener_step(self, stop: threading.Event) -> _Wait | bool:
        """One listener iteration: commands, parked keys, one message.

        Returns False on shutdown, None to loop again at once, or what the
        thread should sleep on before the next step.

        Owns `self._pubsub`. Never blocks on the dispatch queue: inbound keys
        it cannot queue are parked in `_overflow`, and the socket is read
        only while the overflow has room, so commands and confirmations keep
        flowing under a backlog.
        """
        # 1. Drain pending sub/unsub commands.
        q = self._control_cmd_queue
        if q is None:
            return False
        while True:
            try:
                action, channel, ack = q.get_nowait()
            except queue.Empty:
                break
            if self._run_cmd(action, channel, ack):
                return False
        if stop.is_set():
            return False

        # 2. Parked keys first, so they stay ahead of anything read now. With
        #    a backlog this is also where the step waits (for queue room),
        #    unless a SUBSCRIBE awaits its confirmation: then the listener
        #    reads back to back, so the confirmation is not paced by the
        #    backlog ahead of it.
        pending = bool(self._pending)
        self._drain_overflow(wait=0.0 if pending else _OVERFLOW_POLL_SECONDS)
        # Past the overflow cap the socket is left unread (backpressure into
        # valkey), unless a SUBSCRIBE awaits its confirmation: that must be
        # read whatever the backlog, so subscribe() never depends on
        # callback queue room. In that state the overflow can absorb what
        # the server buffered for this client, so the real bound is valkey's
        # pubsub output buffer limit; past it valkey disconnects the client,
        # which degrades to a reconnect and a resync wake.
        room = len(self._overflow) < self._dispatch_queue_maxsize or pending

        # 3. Take one message if redis-py has one and there is room for it,
        #    else sleep until there is one, a command is queued, or (with keys
        #    parked) the dispatch queue may have room again.
        msg = None
        if room:
            try:
                msg = self._pubsub.get_message(timeout=0)
            except Exception as e:
                logger.warning("Listener get_message failed: %s", e)
                self._invalidate_ready("lost")
                stop.wait(0.5)
                return None
        if msg is None:
            # With keys parked the drain above already waited (or, with a
            # SUBSCRIBE pending, must not wait long): poll the sockets so
            # commands and confirmations are not delayed.
            if not self._overflow:
                timeout = _LISTENER_IDLE_SECONDS
            else:
                timeout = _OVERFLOW_POLL_SECONDS if pending else 0.0
            return self._wait_spec(watch_pubsub=room, timeout=timeout)

        mtype = msg.get("type")
        channel = msg["channel"]
        if isinstance(channel, str):
            channel = channel.encode()
        if mtype == "subscribe":
            self._on_subscribed(channel)
        elif mtype in ("message", "pmessage") and channel.startswith(self._notif_prefix_b):
            self._handle_notif(msg, channel[len(self._notif_prefix_b) :].decode())
        return None

    def _handle_notif(self, msg: dict, filter_hash_str: str) -> None:
        try:
            payload = msgpack.unpackb(msg["data"], raw=False)
        except Exception as e:
            logger.warning("Bad msgpack payload on %s: %s", msg.get("channel"), e)
            return
        if not self._connected:
            return
        # Hash-scoped dispatch prevents duplicate delivery when a key matches
        # multiple registered filters (arriving on multiple notif channels),
        # and drops the router's over-delivery (hints are a superset).
        raw_keys = payload.get("keys")
        if raw_keys:
            self._overflow.extend(_TaggedKey(tuple(k), filter_hash_str) for k in raw_keys)
        elif raw_keys == []:
            # A publisher that dropped keys asks every hash to re-read.
            self._overflow.append(_Resync(filter_hash_str))
        else:
            logger.warning("Payload without keys on %s ignored", msg.get("channel"))
            return
        self._drain_overflow()

    # -- Maintenance thread ---------------------------------------------------

    def _reregister_once(self, h: str, gen: int) -> bool:
        """One re-registration attempt for `h`. True when done or no longer local."""
        with self._state_lock:
            entries = self._local.get(h)
        if not entries:
            return True
        try:
            self._register_hash(h, entries)
        except Exception as e:
            logger.warning("Re-register of hash %s failed, retrying: %s", h, e)
            return False
        logger.info("Re-registered filter hash %s after reconnect", h)
        self._mark_ready(h, gen)
        # Routed again: wake each local sub once so it re-reads whatever was
        # written while the hash was not routed. Through the listener, so
        # the wake queues behind keys it read before the reconnect.
        self._dispatch_pubsub_cmd("resync", h, wait=False)
        return True

    def _mark_ready(self, h: str, gen: int) -> None:
        """Re-registration landed: release same-filter waiters.

        Skipped when a newer loss or reconnect happened since the confirmation
        that queued it; that one queues its own re-registration.
        """
        with self._state_lock:
            r = self._ready.get(h)
            if r is not None and gen == self._conn_gen and h in self._local and r.error is None:
                r.event.set()
