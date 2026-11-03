"""Shared helpers for the storage conformance suite."""

from __future__ import annotations

import os
import threading
import time
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

from virtuals.codecs import BinaryKeyCodec
from virtuals.codecs.pickle import PickleCodec
from virtuals.tkv.codec import Codec
from virtuals.tkv.storage import (
    StorageLockTimeoutError,
    StorageTransactionConflictError,
)


if TYPE_CHECKING:
    from collections.abc import Callable


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


# Iteration knobs. Small defaults keep each case well under a second on a
# correct backend; raise them through the environment for a soak run.
ROUNDS = _env_int("CONFORMANCE_ROUNDS", 5)  # races and interleavings per case
THREADS = _env_int("CONFORMANCE_THREADS", 4)  # writer threads in lost update
ITERS = _env_int("CONFORMANCE_ITERS", 10)  # increments per thread or process
PROCS = _env_int("CONFORMANCE_PROCS", 3)  # writer processes in lost update
CORO_N = _env_int("CONFORMANCE_CORO_N", 10)  # coroutine writers in section 4

# Timing. PAUSE sits between a read and the dependent write so that a race
# happens on every round on a backend that allows it. BARRIER is how long a
# racer waits for its peer; on a serializing backend the peer is blocked at
# begin, the wait times out, and the run continues serially.
PAUSE = 0.002
BARRIER = 0.05
JOIN = 10.0

RETRYABLE = (StorageTransactionConflictError, StorageLockTimeoutError)
ATTEMPTS = 500


class GaveUpError(AssertionError):
    pass


def retrying(fn: Callable[[], Any], attempts: int = ATTEMPTS) -> Any:
    """Run ``fn``, retrying only contract-retryable errors with backoff.

    Any other exception propagates, so a backend that signals contention with
    a non-retryable class fails the case.
    """
    for i in range(attempts):
        try:
            return fn()
        except RETRYABLE:
            time.sleep(min(0.0005 * (i + 1), 0.005))
    raise GaveUpError(f"still conflicting after {attempts} attempts")


async def aretrying(fn: Callable[[], Any], attempts: int = ATTEMPTS) -> Any:
    import asyncio

    for i in range(attempts):
        try:
            return await fn()
        except RETRYABLE:
            await asyncio.sleep(min(0.0005 * (i + 1), 0.005))
    raise GaveUpError(f"still conflicting after {attempts} attempts")


def run_threads(*targets: Callable[[], Any], timeout: float = JOIN) -> list[Any]:
    """Run each target on its own thread, return results, re-raise the first error."""
    results: list[Any] = [None] * len(targets)
    errors: list[BaseException] = []

    def wrap(i: int, fn: Callable[[], Any]) -> None:
        try:
            results[i] = fn()
        except BaseException as e:
            errors.append(e)

    threads = [
        threading.Thread(target=wrap, args=(i, fn), daemon=True) for i, fn in enumerate(targets)
    ]
    for t in threads:
        t.start()
    deadline = time.monotonic() + timeout
    for t in threads:
        t.join(max(0.0, deadline - time.monotonic()))
    if any(t.is_alive() for t in threads):
        raise AssertionError(f"threads still running after {timeout}s (stall)")
    if errors:
        raise errors[0]
    return results


def wait_barrier(barrier: threading.Barrier) -> None:
    try:
        barrier.wait(timeout=BARRIER)
    except threading.BrokenBarrierError:
        pass


def put_committed(storage: Any, items: dict) -> None:
    def body() -> None:
        with storage.transaction() as t:
            for k, v in items.items():
                t.put(k, v)

    retrying(body)


# Cross-process worker. Module level so spawn can import it.
def proc_increment(backend: str, root: str, key: tuple, iters: int, pause: float) -> None:
    from tests.conformance.backends import BACKENDS

    st = BACKENDS[backend].make(root)
    st.open()
    try:
        for _ in range(iters):

            def body() -> None:
                with st.transaction() as t:
                    v = t.get(key)
                    time.sleep(pause)
                    t.put(key, v + 1)

            retrying(body)
    finally:
        st.close()


# Section 7 counter, below the transaction API.
class CountingCodec(Codec):
    """``BinaryCodec`` that counts every key the backend encodes or decodes.

    Every backend turns a key into bytes to address it and decodes a key for
    every entry it walks in a scan, so ``touched()`` counts keys handled
    inside the backend, whatever the transaction wrapper reports. It cannot
    see a walk over raw bytes that never decodes, which the timing check in
    section 7 covers.
    """

    def __init__(self) -> None:
        super().__init__(key_codec_cls=BinaryKeyCodec, value_codec_cls=PickleCodec)
        self.c: Counter = Counter()
        self.encode_key = self._counted("encode_key", self.encode_key)
        self.decode_key = self._counted("decode_key", self.decode_key)

    def _counted(self, name: str, fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
        def counted(x: Any) -> Any:
            self.c[name] += 1
            return fn(x)

        return counted

    def reset(self) -> None:
        self.c.clear()

    def touched(self) -> int:
        return self.c["encode_key"] + self.c["decode_key"]


# Cross-process contention worker for section 3. Module level so spawn can import it.
def proc_contend(
    backend: str, root: str, out: str, lock_timeout: float, hold: float, rounds: int, start: Any
) -> None:
    import json

    from tests.conformance.backends import BACKENDS

    key = ("counter",)
    st = BACKENDS[backend].make(root, lock_timeout=lock_timeout)
    st.open()
    errors: list[list] = []
    commits = 0
    try:
        start.wait(30)
        for _ in range(rounds):
            try:
                with st.transaction() as t:
                    v = t.get(key)
                    t.put(key, v + 1)
                    time.sleep(hold)
                commits += 1
            except Exception as e:
                errors.append([type(e).__name__, isinstance(e, RETRYABLE), str(e)[:80]])
    finally:
        st.close()
    Path(out).write_text(json.dumps({"errors": errors, "commits": commits}))
