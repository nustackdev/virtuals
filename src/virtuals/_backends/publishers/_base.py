"""Publisher base: bounded queue + drain worker.

Shared queue/worker/flush machinery for all publishers. Subclasses
implement `_publish_batch(batch)` -- the transport-specific send. This
mirrors today's fused Observer._worker/_deliver_batch minus the local
match half (which now belongs to Observer).

Exactly one worker ever runs sends. The worker owns the queue and stop
event it was started with (never read off the publisher), so a worker that
outlives a timed-out disconnect() cannot pick up the next connect()'s queue;
connect() refuses while it is still sending. The worker holds the publisher
only weakly between batches, so a publisher dropped without disconnect() is
collected and its worker exits.
"""

from __future__ import annotations

import queue
import threading
import time
import weakref
from logging import getLogger
from typing import TYPE_CHECKING, Any

from typing_extensions import Self

from virtuals.tkv.publisher import PublisherConnectionError, PublisherOperationError


if TYPE_CHECKING:
    from collections.abc import Iterable
    from types import TracebackType

    from virtuals.tkv.types import Key


logger = getLogger(__name__)


__all__ = [
    "DEFAULT_NOTIFY_TIMEOUT_SECONDS",
    "DEFAULT_PUBLISH_RETRY_SECONDS",
    "DEFAULT_QUEUE_MAXSIZE",
    "PublisherBase",
]


# Default queue capacity. At 1200 keys/s target throughput this is ~80s of
# buffer -- comfortable headroom for bursts, small enough that a stuck
# publisher surfaces as writer backpressure well before OOM.
DEFAULT_QUEUE_MAXSIZE: int = 100_000

# How long a failing batch is retried before it is dropped and surfaced.
DEFAULT_PUBLISH_RETRY_SECONDS: float = 5.0

# Longest notify() blocks a writer on a full queue before dropping the rest
# of its keys (counted, surfaced, followed by a resync wake).
DEFAULT_NOTIFY_TIMEOUT_SECONDS: float = 1.0

# Retry backoff: first delay, doubling, capped.
_RETRY_BACKOFF_START: float = 0.05
_RETRY_BACKOFF_CAP: float = 1.0

# Sentinel that ends the worker: everything queued before it is published.
_STOP_MARKER: object = object()

# How long disconnect() waits to queue the stop marker and for the worker to
# drain up to it.
_STOP_JOIN_SECONDS: float = 5.0

# Idle poll of the worker; also how often a pending resync is retried at most.
_IDLE_POLL_SECONDS: float = 0.1


def _stop_worker(q: queue.Queue[Any], stop: threading.Event) -> None:
    """Ask a worker to stop. Runs from disconnect() and from the finalizer."""
    stop.set()
    try:
        q.put_nowait(_STOP_MARKER)
    except queue.Full:
        pass


def _worker(ref: weakref.ref[PublisherBase], q: queue.Queue[Any], stop: threading.Event) -> None:
    """Drain `q`, batch, publish. Exits at the stop marker, or on an empty queue after stop.

    Module level so the thread holds its owner only through `ref`, and only
    while publishing: a publisher dropped without disconnect() is collected
    and this worker exits at its next wake.
    """
    while True:
        try:
            item = q.get(timeout=_IDLE_POLL_SECONDS)
        except queue.Empty:
            if stop.is_set():
                return
            owner = ref()
            if owner is None:
                return
            owner._try_resync()
            del owner
            continue

        # Drain what's queued right now into a single batch. Sentinels
        # flush the accumulated batch first, then signal.
        batch: list[Key] = []
        while True:
            if item is _STOP_MARKER:
                if _publish(ref, batch, stop):
                    owner = ref()
                    if owner is not None and not owner._drain_failed:
                        owner._try_resync(force=True)  # a drop just before stop still wakes
                return
            if isinstance(item, threading.Event):
                alive = _publish(ref, batch, stop)
                batch = []
                item.set()
                if not alive:
                    return
            else:
                batch.append(item)  # Key tuple
            try:
                item = q.get_nowait()
            except queue.Empty:
                break

        if not _publish(ref, batch, stop):
            return


def _publish(ref: weakref.ref[PublisherBase], batch: list[Key], stop: threading.Event) -> bool:
    """Publish one batch through the owner. False once the owner is gone."""
    if not batch:
        return True
    owner = ref()
    if owner is None:
        return False
    try:
        owner._safe_publish(batch, stop)
    except BaseException:
        logger.exception("Publisher worker: publish raised; dropping %d keys", len(batch))
        owner._record_drop(len(batch), PublisherOperationError("publish raised"))
    return True


class PublisherBase:
    """Write-side base with queue + worker.

    notify() enqueues keys; a background worker drains, batches, and
    hands the batch to the subclass hook `_publish_batch`, split into
    chunks of at most `MAX_KEYS_PER_CALL` keys when a subclass sets it.

    No silent drops. A failing chunk is retried with backoff for up to
    `publish_retry_seconds`; meanwhile the worker does not drain, the queue
    fills and notify() blocks (backpressure) for up to `notify_timeout`,
    then drops the rest of its keys. Every drop is logged at ERROR, counted
    in `dropped_keys`, raised once by the next flush(), and followed by a
    resync wake: the next time the worker can publish, it asks every
    subscriber on the prefix to re-read (`_send_resync`, transport
    specific), so a level flow recovers what the drop lost.

    disconnect() drains: the worker publishes every key queued before it
    (one attempt per chunk, no retry window; after the first failed chunk
    the rest are dropped without an attempt, so a dead server bounds the
    drain at one attempt). Keys it cannot publish, and keys still queued if
    the worker cannot finish, are counted and surfaced the same way.

    Delivery is at least once per retry: a chunk whose reply is lost (for
    example a connection error after the server ran the call) is sent again,
    so subscribers may see a key twice. Harmless for level flows, which
    re-read state on every wake.

    Subclasses either implement `_publish_batch(batch)`, or split it into
    `_prepare_chunk(chunk)` (encode once) and `_send_chunk(prepared)` (the
    retried transport call).
    """

    MAX_KEYS_PER_CALL: int | None = None

    def __init__(
        self,
        *,
        queue_maxsize: int = DEFAULT_QUEUE_MAXSIZE,
        publish_retry_seconds: float = DEFAULT_PUBLISH_RETRY_SECONDS,
        notify_timeout: float | None = DEFAULT_NOTIFY_TIMEOUT_SECONDS,
    ) -> None:
        self._queue_maxsize = queue_maxsize
        self._publish_retry_seconds = publish_retry_seconds
        self._notify_timeout = notify_timeout
        self._connected: bool = False
        self.dropped_keys: int = 0
        self._last_error: BaseException | None = None
        self._error_lock = threading.Lock()
        # Set by any drop; cleared once a resync wake went out (worker only).
        self._resync_pending: bool = False
        self._next_resync_at: float = 0.0
        self._resync_backoff: float = _RETRY_BACKOFF_START
        # Set by a notify() drop, cleared by the worker's next successful
        # send: while set, writers drop at once instead of each waiting out
        # `notify_timeout` (the loss is covered by the resync wake anyway).
        self._stalled: bool = False
        # Set once a chunk failed after stop: the rest of the drain is dropped.
        self._drain_failed: bool = False

        self._queue: queue.Queue[Any] = queue.Queue(maxsize=queue_maxsize)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._worker_finalizer: weakref.finalize | None = None

    def _ensure_connected(self) -> None:
        if not self._connected:
            raise PublisherConnectionError("Publisher not connected")

    def connect(self) -> None:
        """Connect publisher: fresh queue + worker thread, call _on_connect hook.

        Raises PublisherConnectionError while the previous worker is still
        sending (a disconnect() whose join timed out): there is never a
        second worker.
        """
        if self._connected:
            return
        old = self._thread
        if old is not None and old.is_alive():
            raise PublisherConnectionError(
                f"Publisher worker {old.name} from the previous connection is still sending; "
                "connect again once it has stopped"
            )
        try:
            stop = self._stop_event = threading.Event()
            # Fresh queue on each connect so re-connect doesn't inherit stale items.
            q = self._queue = queue.Queue(maxsize=self._queue_maxsize)
            self._drain_failed = False
            # Per connection: a fresh queue has room, so writers wait for it
            # again. `_resync_pending` survives, the wake is still owed.
            self._stalled = False

            self._on_connect()

            self._thread = threading.Thread(
                target=_worker,
                args=(weakref.ref(self), q, stop),
                name=f"Publisher-{id(self)}",
                daemon=True,
            )
            self._thread.start()
            self._worker_finalizer = weakref.finalize(self, _stop_worker, q, stop)

            self._connected = True
        except Exception as e:
            self._abandon_connect()
            raise PublisherConnectionError(f"Failed to connect: {e}") from e

    def _abandon_connect(self) -> None:
        """Release whatever a failed connect() opened."""
        try:
            self._on_disconnect()
        except Exception as e:
            logger.warning("Publisher cleanup after a failed connect raised: %s", e)
        self._connected = False

    def disconnect(self) -> None:
        """Disconnect publisher: drain the queue, stop worker, call _on_disconnect hook.

        The worker publishes everything queued before the stop marker, one
        attempt per chunk, until the first failure. Whatever it cannot reach
        in time is counted in `dropped_keys` and raised by the next flush().
        If the worker is still inside a send after the join timeout it is
        left to exit on its own (logged at WARNING) and the next connect()
        refuses until it has.
        """
        if not self._connected:
            return
        try:
            # Stop first: from here a failing chunk gets one attempt, so the
            # worker reaches the marker in bounded time.
            self._stop_event.set()
            try:
                self._queue.put(_STOP_MARKER, timeout=_STOP_JOIN_SECONDS)
            except queue.Full:
                pass  # worker stuck; its leftovers are recorded below
            if self._worker_finalizer is not None:
                self._worker_finalizer.detach()
                self._worker_finalizer = None

            t = self._thread
            if t is not None and t.is_alive():
                t.join(timeout=_STOP_JOIN_SECONDS)
                if t.is_alive():
                    logger.warning(
                        "Publisher worker %s still sending after %.1fs; it exits when the "
                        "send returns",
                        t.name,
                        _STOP_JOIN_SECONDS,
                    )
                else:
                    self._thread = None
            else:
                self._thread = None

            if self._thread is None:
                self._drop_leftovers()
            self._on_disconnect()
        finally:
            self._connected = False

    def _drop_leftovers(self) -> None:
        """Record keys left in the queue once the worker is gone."""
        n = 0
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, threading.Event):
                item.set()
            elif item is not _STOP_MARKER:
                n += 1
        if n:
            err = PublisherConnectionError(f"{n} keys still queued at disconnect")
            logger.error("Publisher disconnected with %d unpublished keys", n)
            self._record_drop(n, err)

    def notify(self, keys: Key | Iterable[Key]) -> None:
        """Enqueue keys for publish.

        Fire-and-forget: caller just enqueues; the worker handles batching
        and transport send. Blocks on a full queue for at most
        `notify_timeout` in total, then drops the keys it could not queue
        (logged, counted, surfaced by flush() and by a resync wake): a
        committing writer is never held by a stalled transport. Once a call
        has dropped, later calls on the still-full queue drop at once until
        the worker sends again, so a stall costs the first commit the
        timeout and the rest nothing. With `notify_timeout=None` the call
        blocks until there is room (pure backpressure, never a drop).

        Accepts a single key (tuple) or a batch of keys (set, list, etc.).
        """
        self._ensure_connected()
        q = self._queue
        batch = [keys] if isinstance(keys, tuple) else list(keys)
        timeout = self._notify_timeout
        deadline: float | None = None
        for i, k in enumerate(batch):
            try:
                q.put_nowait(k)
                continue
            except queue.Full:
                pass
            if timeout is None:
                q.put(k)
                continue
            if not self._stalled:
                if deadline is None:
                    deadline = time.monotonic() + timeout
                try:
                    q.put(k, timeout=max(0.0, deadline - time.monotonic()))
                    continue
                except queue.Full:
                    pass
            n = len(batch) - i
            self._stalled = True
            err = PublisherOperationError(f"publish queue full for {timeout}s, {n} keys dropped")
            logger.error("Publisher queue full, dropping %d keys", n)
            self._record_drop(n, err)
            return

    def flush(self, timeout: float = 1.0) -> None:
        """Wait for pending publishes to be sent.

        Places a sentinel in the queue. When the worker reaches it, all
        previously-enqueued keys have been handed to _publish_batch.

        Raises:
            PublisherOperationError: if keys were dropped since the last
                flush (publish kept failing past `publish_retry_seconds`).
                Raised once; the error is cleared.
        """
        if self._connected:
            done = threading.Event()
            self._queue.put(done)  # sentinel after all pending keys
            done.wait(timeout=timeout)
        self._raise_last_error()

    def _record_drop(self, n: int, err: BaseException, *, resync: bool = True) -> None:
        """Count a drop and surface it. `resync=False` for keys no subscriber could match."""
        with self._error_lock:
            self.dropped_keys += n
            self._last_error = err
        if resync:
            self._resync_pending = True

    def _raise_last_error(self) -> None:
        with self._error_lock:
            err, self._last_error = self._last_error, None
        if err is not None:
            raise PublisherOperationError(f"Publish failed, keys dropped: {err}") from err

    def _safe_publish(self, batch: list[Key], stop: threading.Event | None = None) -> None:
        if not batch:
            return
        if stop is None:
            stop = self._stop_event
        n = self.MAX_KEYS_PER_CALL or len(batch)
        for i in range(0, len(batch), n):
            self._publish_with_retry(batch[i : i + n], stop)

    def _publish_with_retry(self, chunk: list[Key], stop: threading.Event) -> None:
        """Publish one chunk, retrying with backoff until the deadline.

        The chunk is encoded once (`_prepare_chunk`); only the send retries,
        so per-key encode errors are recorded once. After stop, a chunk gets
        one attempt, and once one has failed the rest are dropped unsent.
        """
        try:
            prepared, n = self._prepare_chunk(chunk)
        except Exception as e:
            logger.error("Publish failed, dropping %d keys: %s", len(chunk), e)
            self._record_drop(len(chunk), e, resync=False)  # not encodable: unroutable anyway
            return
        if not n:
            return
        if self._drain_failed and stop.is_set():
            logger.error("Publish skipped at disconnect, dropping %d keys", n)
            self._record_drop(n, PublisherConnectionError("disconnecting, transport down"))
            return
        deadline = time.monotonic() + self._publish_retry_seconds
        delay = _RETRY_BACKOFF_START
        while True:
            try:
                self._send_chunk(prepared)
            except Exception as e:
                remaining = deadline - time.monotonic()
                # Disconnect in progress: don't hold the worker past one attempt.
                if remaining <= 0 or stop.is_set():
                    logger.error("Publish failed, dropping %d keys: %s", n, e)
                    self._record_drop(n, e)
                    if stop.is_set():
                        self._drain_failed = True
                    return
                logger.warning("Publish failed (%d keys), retrying: %s", n, e)
                if stop.wait(min(delay, remaining)):
                    continue  # loop once more; the check above records the drop
                delay = min(delay * 2, _RETRY_BACKOFF_CAP)
                continue
            self._stalled = False  # writers may wait for room again
            if self._resync_pending:
                self._try_resync()
            return

    def _try_resync(self, *, force: bool = False) -> None:
        """Send the pending resync wake if due. Worker only.

        Called after every successful send and on every idle poll, so the
        wake goes out as soon as the transport is back even with nothing
        else to publish. Failures back off up to the retry cap (`force`
        ignores the backoff: the last attempt before the worker exits).

        The flag is cleared before the send and set back on failure: a drop
        recorded while the wake is in flight keeps its own wake, since that
        drop's commit may have landed after this wake was published.
        """
        if not self._resync_pending:
            return
        if not force and time.monotonic() < self._next_resync_at:
            return
        self._resync_pending = False
        try:
            self._send_resync()
        except Exception as e:
            logger.warning("Publisher resync wake failed, retrying: %s", e)
            self._resync_pending = True
            self._next_resync_at = time.monotonic() + self._resync_backoff
            self._resync_backoff = min(self._resync_backoff * 2, _RETRY_BACKOFF_CAP)
            return
        self._resync_backoff = _RETRY_BACKOFF_START
        logger.info("Publisher sent a resync wake after dropping keys")

    # -- Subclass hooks -------------------------------------------------------

    def _on_connect(self) -> None:
        """Hook called from connect() before the worker starts. Default: no-op."""
        return None

    def _on_disconnect(self) -> None:
        """Hook called from disconnect() after the worker joins. Default: no-op.

        Also called after a failed connect(), with whatever _on_connect
        managed to set up: it must tolerate a half-initialised state.
        """
        return None

    def _prepare_chunk(self, chunk: list[Key]) -> tuple[Any, int]:
        """Encode a chunk once, before any send attempt.

        Returns (prepared, n): what `_send_chunk` sends, and how many keys it
        carries (0 skips the send). Default: the chunk as is.
        """
        return chunk, len(chunk)

    def _send_chunk(self, prepared: Any) -> None:  # noqa: ANN401 -- subclass-defined
        """Send one prepared chunk; retried on error. Default: `_publish_batch`."""
        self._publish_batch(prepared)

    def _send_resync(self) -> None:
        """Ask every subscriber on this transport to re-read, after a drop.

        Default: nothing (the drop stays counted and surfaced by flush()).
        Transports that can reach their subscribers implement it; it is
        retried until it succeeds.
        """
        return None

    def _publish_batch(self, batch: list[Key]) -> None:
        """Send a batch of keys onto the transport. Subclass responsibility."""
        raise NotImplementedError

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
