"""Fixtures for view layer testing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from virtuals._views import EagerDictView
from virtuals.codecs import NoOpCodec
from virtuals.navigator import Navigator
from virtuals.storages.mem import InMemoryStorage


if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from virtuals.tkv.storage import SnapshotProtocol, StorageProtocol, TransactionProtocol
    from virtuals.view import View


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def storage() -> Generator[StorageProtocol, None, None]:
    """Memory storage instance for functional tests."""
    storage = InMemoryStorage(codec=NoOpCodec())
    storage.open()
    try:
        yield storage
    finally:
        storage.close()


@pytest.fixture
def tx(storage: StorageProtocol) -> Generator[TransactionProtocol, None, None]:
    """Read-write transaction context."""
    with storage.transaction() as transaction:
        yield transaction


@pytest.fixture
def snapshot(storage: StorageProtocol) -> Generator[SnapshotProtocol, None, None]:
    """Read-only snapshot context."""
    with storage.snapshot() as snap:
        yield snap


@pytest.fixture
def nav(storage: StorageProtocol) -> Navigator:
    """Navigator instance for testing."""
    return Navigator(storage)


@pytest.fixture
def root_view(tx: TransactionProtocol, nav: Navigator) -> EagerDictView:
    """Create an EagerDictView at root for testing."""
    return nav.root(tx)


# ============================================================================
# View Factories
# ============================================================================


@pytest.fixture
def dict_factory(root_view: EagerDictView) -> Callable[[str, dict[str, Any] | None], View]:
    """Factory for creating EagerDictViews with test data."""

    def _create(address: str, data: dict | None = None) -> View:
        view = root_view.open_child(address, EagerDictView)
        if data is not None:
            view.store(data)
        return view

    return _create


@pytest.fixture
def list_factory(root_view: EagerDictView) -> Callable[[str, list[Any] | None], View]:
    """Factory for creating EagerListViews with test data."""

    def _create(address: str, data: list | None = None) -> View:
        from virtuals._views import EagerListView

        view = root_view.open_child(address, EagerListView)
        if data is not None:
            view.store(data)
        return view

    return _create


@pytest.fixture
def set_factory(root_view: EagerDictView) -> Callable[[str, set[Any] | None], View]:
    """Factory for creating SetViews with test data."""

    def _create(address: str, data: set | None = None) -> View:
        from virtuals._views import SetView

        view = root_view.open_child(address, SetView)
        if data is not None:
            view.store(data)
        return view

    return _create


@pytest.fixture
def tuple_factory(root_view: EagerDictView) -> Callable[[str, tuple[Any, ...] | None], View]:
    """Factory for creating TupleViews with test data."""

    def _create(address: str, data: tuple | None = None) -> View:
        from virtuals._views import TupleView

        view = root_view.open_child(address, TupleView)
        if data is not None:
            view.store(data)
        return view

    return _create


@pytest.fixture
def frozenset_factory(root_view: EagerDictView) -> Callable[[str, frozenset[Any] | None], View]:
    """Factory for creating FrozenSetViews with test data."""

    def _create(address: str, data: frozenset | set | None = None) -> View:
        from virtuals._views import FrozenSetView

        view = root_view.open_child(address, FrozenSetView)
        if data is not None:
            view.store(data)
        return view

    return _create


@pytest.fixture
def bytearray_factory(root_view: EagerDictView) -> Callable[[str, bytearray | None], View]:
    """Factory for creating ByteArrayViews with test data."""

    def _create(address: str, data: bytearray | bytes | None = None) -> View:
        from virtuals._views import ByteArrayView

        view = root_view.open_child(address, ByteArrayView)
        if data is not None:
            view.store(data)
        return view

    return _create


@pytest.fixture
def flat_dict_factory(root_view: EagerDictView) -> Callable[[str, dict[str, Any] | None], View]:
    """Factory for creating FlatDictViews with test data."""

    def _create(address: str, data: dict | None = None) -> View:
        from virtuals._views import FlatDictView

        view = root_view.open_child(address, FlatDictView)
        if data is not None:
            view.store(data)
        return view

    return _create


@pytest.fixture
def flat_list_factory(root_view: EagerDictView) -> Callable[[str, list[Any] | None], View]:
    """Factory for creating FlatListViews with test data."""

    def _create(address: str, data: list | None = None) -> View:
        from virtuals._views import FlatListView

        view = root_view.open_child(address, FlatListView)
        if data is not None:
            view.store(data)
        return view

    return _create


@pytest.fixture
def light_dict_factory(root_view: EagerDictView) -> Callable[[str, dict[str, Any] | None], View]:
    """Factory for creating LightDictViews with test data."""

    def _create(address: str, data: dict | None = None) -> View:
        from virtuals._views import LightDictView

        view = root_view.open_child(address, LightDictView)
        if data is not None:
            view.store(data)
        return view

    return _create


@pytest.fixture
def indexed_dict_factory(
    root_view: EagerDictView,
) -> Callable[[str, dict[str, Any] | None], View]:
    """Factory for creating EagerIndexedDictViews with test data."""

    def _create(address: str, data: dict | None = None) -> View:
        from virtuals._views import EagerIndexedDictView

        view = root_view.open_child(address, EagerIndexedDictView)
        if data is not None:
            view.store(data)
        return view

    return _create
