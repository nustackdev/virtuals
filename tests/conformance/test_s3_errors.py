"""Contract section 3: only conflict and lock timeout are retryable."""

from __future__ import annotations

import threading
import time

import pytest

from tests.conformance.helpers import RETRYABLE, THREADS, run_threads
from virtuals.tkv.storage import StorageError


K = ("counter",)
HOLD = 0.03  # a writer holds its lock this long
LOCK_TIMEOUT = 0.01  # well under HOLD, so waiting writers give up


@pytest.mark.timeout(20)
def test_s3_contention_raises_only_retryable(backend, tmp_path):
    """Writers aborted by conflict or busy raise only RETRYABLE classes.

    The storage is opened with a lock timeout shorter than the time each
    writer holds its lock, so a backend with a timeout knob must surface
    contention. Backends that queue writers raise nothing, which also passes.
    """
    st = backend.make(tmp_path, lock_timeout=LOCK_TIMEOUT)
    st.open()
    try:
        with st.transaction() as t:
            t.put(K, 0)
        errors: list[BaseException] = []
        lock = threading.Lock()

        def writer() -> None:
            for _ in range(3):
                try:
                    with st.transaction() as t:
                        v = t.get(K)
                        t.put(K, v + 1)
                        time.sleep(HOLD)
                except Exception as e:
                    with lock:
                        errors.append(e)

        run_threads(*[writer] * THREADS)
    finally:
        st.close()
    wrong = sorted(
        {f"{type(e).__name__}: {str(e)[:80]}" for e in errors if not isinstance(e, RETRYABLE)}
    )
    assert not wrong, f"contention raised non-retryable errors: {wrong}"


@pytest.mark.timeout(10)
def test_s3_closed_storage_error_not_retryable(backend, tmp_path):
    """Writing to a closed storage raises a StorageError that is not retryable."""
    st = backend.make(tmp_path)
    st.open()
    st.close()
    with pytest.raises(StorageError) as info:
        with st.transaction() as t:
            t.put(K, 1)
    assert not isinstance(info.value, RETRYABLE), type(info.value).__name__
