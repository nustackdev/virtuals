"""Fixtures for the storage conformance suite.

Every case takes the ``backend`` fixture, so it runs once per backend in
``BACKENDS``. Known failures from ``known_failures.py`` are applied here.
"""

from __future__ import annotations

import pytest

from tests.conformance.backends import BACKENDS, BackendSpec
from tests.conformance.helpers import JOIN, STALLED_THREADS
from tests.conformance.known_failures import KNOWN_FAILURES
from tests.support.valkey import redis_url_or_private


@pytest.fixture(params=list(BACKENDS), ids=list(BACKENDS))
def backend(request) -> BackendSpec:
    return BACKENDS[request.param]


@pytest.fixture(scope="session")
def redis_url(tmp_path_factory: pytest.TempPathFactory):
    """Section 5 over the redis transport: TEST_REDIS_URL, else a private valkey."""
    yield from redis_url_or_private(tmp_path_factory)


@pytest.fixture
def storage(backend: BackendSpec, tmp_path):
    st = backend.make(tmp_path)
    st.open()
    yield st
    st.close()


@pytest.fixture(autouse=True)
def reap_stalled_threads():
    """Join threads a stalled ``run_threads`` left behind, after the storage closed.

    Autouse fixtures are set up first, so this tears down after ``storage``:
    by then close() has freed any writer stuck in the backend. A thread
    still alive after the join window fails here, by name, instead of
    running on into the next case.
    """
    yield
    if not STALLED_THREADS:
        return
    threads, STALLED_THREADS[:] = list(STALLED_THREADS), []
    for t in threads:
        t.join(JOIN)
    alive = [t.name for t in threads if t.is_alive()]
    assert not alive, f"stalled threads survived storage close: {alive}"


@pytest.fixture
def multi_process(backend: BackendSpec) -> BackendSpec:
    if backend.process_model != "multi":
        pytest.skip(f"{backend.name} is a single-process backend (contract section 4)")
    return backend


def pytest_collection_modifyitems(config, items):
    for item in items:
        callspec = getattr(item, "callspec", None)
        if callspec is None or "backend" not in callspec.params:
            continue
        rows = KNOWN_FAILURES.get(item.originalname, {})
        backend = callspec.params["backend"]
        # A row names a backend, or one parametrization of it ("mem-txn").
        keys = [backend, *(f"{backend}-{v}" for k, v in callspec.params.items() if k != "backend")]
        row = next((rows[k] for k in keys if k in rows), None)
        if row is not None:
            strict, reason = row
            item.add_marker(pytest.mark.xfail(strict=strict, reason=reason))
