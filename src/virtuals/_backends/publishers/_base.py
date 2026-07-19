"""Publisher base: bounded queue + drain worker.

Shared queue/worker/flush machinery for all publishers. Subclasses
implement `_publish_batch(batch)` -- the transport-specific send. This
mirrors today's fused Observer._worker/_deliver_batch minus the local
match half (which now belongs to Observer).
"""

from __future__ import annotations

import queue
import threading
from logging import getLogger
from typing import TYPE_CHECKING, Any, Self

from virtuals.tkv.publisher import PublisherConnectionError


if TYPE_CHECKING:
    from collections.abc import Iterable
    from types import TracebackType

    from virtuals.tkv.types import Key


logger = getLogger(__name__)


__all__ = [
    "DEFAULT_QUEUE_MAXSIZE",
    "PublisherBase",
]


# Default queue capacity. At 1200 keys/s target throughput this is ~80s of
# buffer -- comfortable headroom for bursts, small enough that a stuck
# publisher surfaces as writer backpressure well before OOM.
DEFAULT_QUEUE_MAXSIZE: int = 100_000

# Sentinel used to unblock the worker sleeping on Queue.get() during disconnect.
_STOP_MARKER: object = object()


class PublisherBase:
    """Write-side base with queue + worker.

    notify() enqueues keys; a background worker drains, batches, and
    hands the batch to the subclass hook `_publish_batch`. Backpressure:
    notify() blocks briefly when the queue is full (default capacity
    DEFAULT_QUEUE_MAXSIZE). No silent drops.
    """

    def __init__(self, *, queue_maxsize: int = DEFAULT_QUEUE_MAXSIZE) -> None:
        self._queue_maxsize = queue_maxsize
        self._connected: bool = False

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
        """
        if not self._connected:
            return
        done = threading.Event()
        self._queue.put(done)  # sentinel after all pending keys
        done.wait(timeout=timeout)

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
        try:
            self._publish_batch(batch)
        except Exception as e:
            logger.error("Publisher _publish_batch failed (%d keys): %s", len(batch), e)

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
