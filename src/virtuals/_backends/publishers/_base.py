"""Publisher base: bounded queue + drain worker.

Shared queue/worker/flush machinery for all publishers. Subclasses
implement `_publish_batch(batch)` -- the transport-specific send. This
mirrors today's fused Observer._worker/_deliver_batch minus the local
match half (which now belongs to Observer).
"""

from __future__ import annotations

import queue
import threading
import time
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

# Retry backoff: first delay, doubling, capped.
_RETRY_BACKOFF_START: float = 0.05
_RETRY_BACKOFF_CAP: float = 1.0

# Sentinel used to unblock the worker sleeping on Queue.get() during disconnect.
_STOP_MARKER: object = object()


class PublisherBase:
    """Write-side base with queue + worker.

    notify() enqueues keys; a background worker drains, batches, and
    hands the batch to the subclass hook `_publish_batch`, split into
    chunks of at most `MAX_KEYS_PER_CALL` keys when a subclass sets it.

    No silent drops. A failing chunk is retried with backoff for up to
    `publish_retry_seconds`; meanwhile the worker does not drain, the queue
    fills and notify() blocks (backpressure). After the deadline the keys
    are logged at ERROR, counted in `dropped_keys`, and the next flush()
    raises PublisherOperationError once.
    """

    MAX_KEYS_PER_CALL: int | None = None

    def __init__(
        self,
        *,
        queue_maxsize: int = DEFAULT_QUEUE_MAXSIZE,
        publish_retry_seconds: float = DEFAULT_PUBLISH_RETRY_SECONDS,
    ) -> None:
        self._queue_maxsize = queue_maxsize
        self._publish_retry_seconds = publish_retry_seconds
        self._connected: bool = False
        self.dropped_keys: int = 0
        self._last_error: BaseException | None = None
        self._error_lock = threading.Lock()

        self._queue: queue.Queue[Any] = queue.Queue(maxsize=queue_maxsize)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _ensure_connected(self) -> None:
        if not self._connected:
            raise PublisherConnectionError("Publisher not connected")

    def connect(self) -> None:
        """Connect publisher: fresh queue + worker thread, call _on_connect hook."""
        if self._connected:
            return
        try:
            self._stop_event.clear()
            # Fresh queue on each connect so re-connect doesn't inherit stale items.
            self._queue = queue.Queue(maxsize=self._queue_maxsize)

            self._on_connect()

            self._thread = threading.Thread(
                target=self._worker,
                name=f"Publisher-{id(self)}",
                daemon=True,
            )
            self._thread.start()

            self._connected = True
        except Exception as e:
            raise PublisherConnectionError(f"Failed to connect: {e}") from e

    def disconnect(self) -> None:
        """Disconnect publisher: stop worker, drain, call _on_disconnect hook."""
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

            self._on_disconnect()
        finally:
            self._connected = False

    def notify(self, keys: Key | Iterable[Key]) -> None:
        """Enqueue keys for publish.

        Fire-and-forget: caller just enqueues; the worker handles batching
        and transport send. Blocks briefly when the queue is full.

        Accepts a single key (tuple) or a batch of keys (set, list, etc.).
        """
        self._ensure_connected()
        if isinstance(keys, tuple):
            self._queue.put(keys)
        else:
            for k in keys:
                self._queue.put(k)

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

    def _record_drop(self, n: int, err: BaseException) -> None:
        with self._error_lock:
            self.dropped_keys += n
            self._last_error = err

    def _raise_last_error(self) -> None:
        with self._error_lock:
            err, self._last_error = self._last_error, None
        if err is not None:
            raise PublisherOperationError(f"Publish failed, keys dropped: {err}") from err

    def _worker(self) -> None:
        """Background thread: drain queue, batch, call _publish_batch."""
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
                    self._safe_publish(batch)
                    return
                if isinstance(item, threading.Event):
                    self._safe_publish(batch)
                    batch = []
                    item.set()
                else:
                    batch.append(item)  # Key tuple
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    break

            self._safe_publish(batch)

    def _safe_publish(self, batch: list[Key]) -> None:
        if not batch:
            return
        n = self.MAX_KEYS_PER_CALL or len(batch)
        for i in range(0, len(batch), n):
            self._publish_with_retry(batch[i : i + n])

    def _publish_with_retry(self, chunk: list[Key]) -> None:
        """Publish one chunk, retrying with backoff until the deadline."""
        deadline = time.monotonic() + self._publish_retry_seconds
        delay = _RETRY_BACKOFF_START
        while True:
            try:
                self._publish_batch(chunk)
                return
            except Exception as e:
                remaining = deadline - time.monotonic()
                # Disconnect in progress: don't hold the worker past one attempt.
                if remaining <= 0 or self._stop_event.is_set():
                    logger.error("Publish failed, dropping %d keys: %s", len(chunk), e)
                    self._record_drop(len(chunk), e)
                    return
                logger.warning("Publish failed (%d keys), retrying: %s", len(chunk), e)
                if self._stop_event.wait(min(delay, remaining)):
                    continue  # loop once more; the check above records the drop
                delay = min(delay * 2, _RETRY_BACKOFF_CAP)

    # -- Subclass hooks -------------------------------------------------------

    def _on_connect(self) -> None:
        """Hook called from connect() before the worker starts. Default: no-op."""
        return None

    def _on_disconnect(self) -> None:
        """Hook called from disconnect() after the worker joins. Default: no-op."""
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
