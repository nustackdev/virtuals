"""Fixtures for the storage conformance suite.

Every case takes the ``backend`` fixture, so it runs once per backend in
``BACKENDS``. Known failures from ``known_failures.py`` are applied here.
"""

from __future__ import annotations

import pytest

from tests.conformance.backends import BACKENDS, BackendSpec
from tests.conformance.known_failures import KNOWN_FAILURES


@pytest.fixture(params=list(BACKENDS), ids=list(BACKENDS))
def backend(request) -> BackendSpec:
    return BACKENDS[request.param]


@pytest.fixture
def storage(backend: BackendSpec, tmp_path):
    st = backend.make(tmp_path)
    st.open()
    yield st
    st.close()


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
        row = KNOWN_FAILURES.get(item.originalname, {}).get(callspec.params["backend"])
        if row is not None:
            strict, reason = row
            item.add_marker(pytest.mark.xfail(strict=strict, reason=reason))
