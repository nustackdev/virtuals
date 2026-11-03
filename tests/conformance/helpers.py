"""Shared helpers for the storage conformance suite."""

from __future__ import annotations

import os
import threading
import time
from collections import Counter
from dataclasses import replace
from typing import TYPE_CHECKING, Any

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


# Section 7 counting proxy.
class _CountingScan:
    def __init__(self, raw: Any, filt: Any, counts: Counter) -> None:
        self._raw, self._filter, self._c = raw, filt, counts

    def _keep(self, k: Any) -> bool:
        self._c["visited"] += 1
        return self._filter is None or self._filter.matches(k)

    def items(self):
        for k, v in self._raw.items():
            if self._keep(k):
                yield k, v

    def keys(self):
        for k in self._raw.keys():
            if self._keep(k):
                yield k

    def values(self):
        for k, v in self._raw.items():
            if self._keep(k):
                yield v

    def __iter__(self):
        return self.keys()


class CountingTxn:
    """Wrap a transaction and count every key it touches.

    Scans run with the include ``filter`` stripped and applied here, so
    ``visited`` counts every key the backend walks, not only the matches.
    ``break_filter`` and ``limit`` pass through.
    """

    def __init__(self, inner: Any) -> None:
        self._i = inner
        self.c: Counter = Counter()

    def get(self, k):
        self.c["get"] += 1
        return self._i.get(k)

    def exists(self, k):
        self.c["exists"] += 1
        return self._i.exists(k)

    def multiget(self, ks):
        ks = list(ks)
        self.c["multiget"] += len(ks)
        return self._i.multiget(ks)

    def put(self, k, v):
        self.c["put"] += 1
        return self._i.put(k, v)

    def delete(self, k):
        self.c["delete"] += 1
        return self._i.delete(k)

    def scan(self, options):
        self.c["scan"] += 1
        filt = options.filter
        if filt is None or options.limit is not None:
            # a limit applies after the filter, so it cannot be stripped
            return _CountingScan(self._i.scan(options), None, self.c)
        return _CountingScan(self._i.scan(replace(options, filter=None)), filt, self.c)

    def touched(self) -> int:
        return sum(v for k, v in self.c.items() if k != "scan")

    def __getattr__(self, name):
        return getattr(self._i, name)
