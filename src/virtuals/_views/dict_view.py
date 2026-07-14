"""DictView - Dict-like view over container.

Provides three classes following the eager/lazy facet pattern:
- DictViewBase: shared mutations, keys, lifecycle
- EagerDictView: reads return extracted Python values
- LazyDictView: reads return child Views for containers, values for primitives
"""

from __future__ import annotations

from collections.abc import ItemsView, KeysView, MutableMapping, ValuesView
from typing import TYPE_CHECKING, ClassVar

from virtuals.container import (
    ContainerNotFoundError,
    ContainerProtocol,
    ContainerStructure,
    NodeType,
)
from virtuals.types import EMPTY, Empty, is_empty
from virtuals.view import (
    ChildNavigationBase,
    ChildNestedGetBase,
    ChildNestedSetBase,
    ChildObservableBase,
    ChildPrimitiveSetBase,
    DescendantsObservableBase,
    LazyChildReadBase,
    MetadataBasedChildrenCountBase,
    ObservableBase,
    PrimitiveOpsBase,
    UnsafePrimitiveOpsBase,
    ViewBase,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from collections.abc import Mapping as PyMapping

    from virtuals.collections import (
        Assignable,
        ChildObservable,
        Clearable,
        Containable,
        Convertible,
        Deletable,
        Initializable,
        Nestable,
        Observable,
        ReactiveMappingProtocol,
        Sizeable,
        Subscriptable,
    )

__all__ = [
    "DictViewBase",
    "EagerDictView",
    "LazyDictView",
]


# =============================================================================
# BASE — shared by eager and lazy facets
# =============================================================================


class DictViewBase(
    ObservableBase,
    ChildObservableBase[str | int],
    DescendantsObservableBase,
    MetadataBasedChildrenCountBase,
    ChildNavigationBase[str | int],
    ChildNestedGetBase,
    ChildNestedSetBase,
    ChildPrimitiveSetBase,
    LazyChildReadBase,
    PrimitiveOpsBase,
    UnsafePrimitiveOpsBase,
    ViewBase,
):
    """Dict-like view base — shared by eager and lazy facets.

    Provides mutations, key iteration, membership, lifecycle. Read operations
    that surface child data (__getitem__, values, items) are defined by the
    EagerDictView and LazyDictView facets.

    Example:
        >>> users = nav.root(ctx)
        >>> users["alice"] = {"name": "Alice", "tags": ["python"]}
        >>> list(users.keys())
    """

    STRUCTURE: ClassVar[ContainerStructure] = ContainerStructure(1)
    PROTOCOL: ClassVar[ContainerProtocol] = ContainerProtocol.MAPPING | ContainerProtocol.MUTABLE

    @classmethod
    def is_address_static(cls, address: object) -> bool:
        """Dict keys are always passthrough — no normalization needed."""
        return True

    def normalize_address(self, address: str | int) -> str | int:
        """No normalization needed for dict keys - passthrough."""
        return address

    # =========================================================================
    # MEMBERSHIP & KEYS (same for both facets)
    # =========================================================================

    def __contains__(self, obj: str | int) -> bool:
        """Check if key exists."""
        return self.container.exists_child(obj)

    def __iter__(self) -> Generator[str | int, None, None]:
        """Iterate over keys."""
        yield from self.container.iter_child_keys(validate=False)

    def keys(self) -> KeysView[str | int]:
        """Get all keys as a set-like view.

        Returns a proper KeysView supporting len, contains, and set
        operations (union, intersection, difference, etc.).
        """
        return KeysView(self)  # type: ignore[arg-type]

    # =========================================================================
    # MUTATIONS (same for both facets)
    # =========================================================================

    def __setitem__(self, address: str | int, value: object) -> None:
        """Set value for key."""
        is_new = not self.container.exists_child(address)
        self._set_child_value(address, value)
        if is_new:
            self._increment_length()

    def __delitem__(self, address: str | int) -> None:
        """Delete key. Raises KeyError if not found."""
        self.ensure_created()
        self.container.delete_child(address)
        self._update_count()

    def get(self, address: str | int, default: object | Empty = EMPTY) -> object | Empty:
        """Get value with default fallback."""
        try:
            return self[address]
        except Exception:
            return default

    def clear(self) -> None:
        """Remove all items."""
        self.ensure_created()
        self.container.clear_children(validate=True)
        self._set_length(0)

    def update(self, other: PyMapping[str | int, object] | None = None, **kwargs: object) -> None:
        """Update from dict or kwargs."""
        if other:
            for key, value in other.items():
                self[key] = value
        for key, value in kwargs.items():
            self[key] = value  # type: ignore[assignment]

    def store(self, value: PyMapping[str | int, object], *, replace: bool = True) -> None:
        """Store dict contents."""
        self.ensure_created()
        if replace:
            current_len = len(self)
            if current_len > 0:
                self.clear()

        count = 0
        for key, val in value.items():
            self._set_child_value(key, val)
            count += 1

        self._set_length(count)

    def set_child_container_as(
        self,
        address: str | int,
        value: object,
        view_class: type,
    ) -> None:
        """Peer of ``__setitem__`` that takes the child's view class explicitly.

        ``__setitem__`` dispatches child layout by Python value type
        (``dict → DictView``) — a convention for callers who don't know or
        care. A Ref that carries ``view_type=view_class`` in its slot payload
        already knows; this is the door for it to say so on the write path,
        so non-default layouts (``Kh57View``, ``IndexedDictView``, …) round-
        trip correctly.
        """
        is_new = not self.container.exists_child(address)
        self.ensure_created()
        self._populate_child_container(address, value, view_class=view_class)
        if is_new:
            self._increment_length()


# =============================================================================
# EAGER FACET — reads return extracted Python values
# =============================================================================


class EagerDictView(DictViewBase):
    """Eager dict view — reads return extracted Python values.

    The default dict experience: __getitem__ returns materialized values,
    values()/items() yield Python objects. Works naturally with the entire
    Python ecosystem (json.dumps, itertools, sorted, pprint, etc.).

    Cross-navigate to lazy facet via .lazy property.
    """

    CONTAINER_CLS: ClassVar[type] = dict

    def __getitem__(self, address: str | int) -> object | Empty:
        """Get value for key — returns extracted Python value."""
        try:
            return self._get_child_value(address)
        except ContainerNotFoundError as e:
            raise KeyError(address) from e

    def values(self) -> ValuesView[object]:
        """Get all values as a collection view.

        Returns a proper ValuesView supporting len and contains.
        Uses efficient single-pass iteration over storage.
        """
        return _EagerValuesView(self)

    def items(self) -> ItemsView[str | int, object]:
        """Get all key-value pairs as a set-like view.

        Returns a proper ItemsView supporting len, contains, and set
        operations (union, intersection, difference, etc.).
        Uses efficient single-pass iteration over storage.
        """
        return _EagerItemsView(self)

    def _iter_values(self) -> Generator[object, None, None]:
        """Efficient single-pass value iteration over storage."""
        for k, v in self.container.iter_children(validate=False):
            if v.node_type == NodeType.PRIMITIVE:
                yield v.primitive_value
            elif v.node_type == NodeType.CONTAINER:
                yield self._get_child_value(k, node_info=v)

    def _iter_items(self) -> Generator[tuple[str | int, object], None, None]:
        """Efficient single-pass items iteration over storage."""
        for k, v in self.container.iter_children(validate=False):
            if v.node_type == NodeType.PRIMITIVE:
                yield k, v.primitive_value
            elif v.node_type == NodeType.CONTAINER:
                yield k, self._get_child_value(k, node_info=v)

    def pop(self, address: str | int, default: object | Empty = EMPTY) -> object | Empty:
        """Remove and return value."""
        try:
            value = self[address]
            del self[address]
            return value
        except KeyError:
            if is_empty(default):
                raise
            return default

    def extract(self) -> dict[str | int, object]:
        """Extract all items as dict."""
        return dict(self.items())

    # =========================================================================
    # FACET NAVIGATION
    # =========================================================================

    @property
    def lazy(self) -> LazyDictView:
        """Switch to lazy facet — reads return child Views."""
        return LazyDictView(container=self.container, registry=self.registry)

    @property
    def eager(self) -> EagerDictView:
        """Identity — already eager."""
        return self

    # =========================================================================
    # FUNCTIONAL OPERATIONS
    # =========================================================================

    def map_values(self, fn: Callable[[object], object]) -> dict[str | int, object]:
        """Apply function to each value, keeping keys unchanged."""
        return {k: fn(v) for k, v in self.items()}

    def map_items(
        self, fn: Callable[[str | int, object], tuple[str | int, object]]
    ) -> dict[str | int, object]:
        """Apply function to each (key, value) pair."""
        return dict(fn(k, v) for k, v in self.items())

    def filter(self, fn: Callable[[str | int, object], bool]) -> dict[str | int, object]:
        """Filter items by predicate."""
        return {k: v for k, v in self.items() if fn(k, v)}

    def reduce(self, fn: Callable[[object, str | int, object], object], initial: object) -> object:
        """Reduce dict to single value."""
        result = initial
        for k, v in self.items():
            result = fn(result, k, v)
        return result

    # =========================================================================
    # SEARCH OPERATIONS
    # =========================================================================

    def find(self, fn: Callable[[object], bool]) -> object:
        """Find first value matching predicate."""
        for v in self.values():
            if fn(v):
                return v
        raise ValueError("No matching value found")

    def find_key(self, fn: Callable[[object], bool]) -> str | int:
        """Find first key whose value matches predicate."""
        for k, v in self.items():
            if fn(v):
                return k
        raise ValueError("No matching value found")

    def find_item(self, fn: Callable[[str | int, object], bool]) -> tuple[str | int, object]:
        """Find first item (key, value) matching predicate."""
        for k, v in self.items():
            if fn(k, v):
                return k, v
        raise ValueError("No matching item found")


# =============================================================================
# LAZY FACET — reads return child Views
# =============================================================================


class LazyDictView(DictViewBase):
    """Lazy dict view — reads return child Views for containers.

    Enables composition without materializing data. __getitem__ returns a child
    View for container children (value as-is for primitives). Works naturally
    with Python tools: islice(lazy_dict.values(), 5) yields Views.

    Cross-navigate to eager facet via .eager property.
    """

    def __getitem__(self, address: str | int) -> object:
        """Get child — returns View for containers, value for primitives."""
        try:
            return self._get_child_view_or_value(address)
        except ContainerNotFoundError as e:
            raise KeyError(address) from e

    def values(self) -> ValuesView[object]:
        """Get all children as a collection view.

        Returns Views for containers, values for primitives.
        """
        return _LazyValuesView(self)

    def items(self) -> ItemsView[str | int, object]:
        """Get all pairs as a set-like view.

        Returns (key, View|value) pairs.
        """
        return _LazyItemsView(self)

    def _iter_values(self) -> Generator[object, None, None]:
        """Efficient single-pass value iteration over storage."""
        for k, v in self.container.iter_children(validate=False):
            if v.node_type == NodeType.PRIMITIVE:
                yield v.primitive_value
            elif v.node_type == NodeType.CONTAINER:
                yield self._get_child_view_or_value(k, node_info=v)

    def _iter_items(self) -> Generator[tuple[str | int, object], None, None]:
        """Efficient single-pass items iteration over storage."""
        for k, v in self.container.iter_children(validate=False):
            if v.node_type == NodeType.PRIMITIVE:
                yield k, v.primitive_value
            elif v.node_type == NodeType.CONTAINER:
                yield k, self._get_child_view_or_value(k, node_info=v)

    # =========================================================================
    # FACET NAVIGATION
    # =========================================================================

    @property
    def eager(self) -> EagerDictView:
        """Switch to eager facet — reads return extracted values."""
        return EagerDictView(container=self.container, registry=self.registry)

    @property
    def lazy(self) -> LazyDictView:
        """Identity — already lazy."""
        return self


# =============================================================================
# VIEW CLASSES — proper KeysView / ValuesView / ItemsView
# =============================================================================


class _EagerValuesView(ValuesView):
    """Efficient ValuesView — single-pass iteration over storage."""

    _mapping: EagerDictView

    def __iter__(self) -> Generator[object, None, None]:  # type: ignore[override]
        yield from self._mapping._iter_values()


class _EagerItemsView(ItemsView):
    """Efficient ItemsView — single-pass iteration over storage."""

    _mapping: EagerDictView

    def __iter__(self) -> Generator[tuple[str | int, object], None, None]:  # type: ignore[override]
        yield from self._mapping._iter_items()


class _LazyValuesView(ValuesView):
    """Efficient ValuesView — single-pass iteration, lazy child access."""

    _mapping: LazyDictView

    def __iter__(self) -> Generator[object, None, None]:  # type: ignore[override]
        yield from self._mapping._iter_values()


class _LazyItemsView(ItemsView):
    """Efficient ItemsView — single-pass iteration, lazy child access."""

    _mapping: LazyDictView

    def __iter__(self) -> Generator[tuple[str | int, object], None, None]:  # type: ignore[override]
        yield from self._mapping._iter_items()


MutableMapping.register(EagerDictView)


if TYPE_CHECKING:
    # Verify protocol implementations
    _subscriptable: type[Subscriptable[str | int, object]] = EagerDictView
    _convertible: type[Convertible[object]] = EagerDictView
    _initializable: type[Initializable[PyMapping[str | int, object]]] = EagerDictView
    _assignable: type[Assignable[str | int, object]] = EagerDictView
    _nestable: type[Nestable[str | int]] = EagerDictView
    _containable: type[Containable[str | int]] = EagerDictView
    _sizeable: type[Sizeable] = EagerDictView
    _deletable: type[Deletable[str | int]] = EagerDictView
    _clearable: type[Clearable] = EagerDictView
    _reactive_mapping: type[ReactiveMappingProtocol[str | int, object]] = EagerDictView
    _Observable: type[Observable] = EagerDictView
    _Observable_children: type[ChildObservable] = EagerDictView
