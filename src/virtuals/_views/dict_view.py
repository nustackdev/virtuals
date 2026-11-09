"""DictView - Dict-like view over container.

Uncounted: no stored length. ``len`` counts live children with the
subtree-skipping child scan, so every write touches a constant number of
keys and no key is shared between writers of different entries.

Provides three classes following the eager/lazy facet pattern:
- DictViewBase: shared mutations, keys, lifecycle
- EagerDictView: reads return extracted Python values
- LazyDictView: reads return child Views for containers, values for primitives
"""

from __future__ import annotations

from collections.abc import ItemsView, KeysView, ValuesView
from typing import TYPE_CHECKING, ClassVar

from virtuals.collections import DictMethods, ReactiveMappingBase
from virtuals.container import (
    ContainerNotFoundError,
    ContainerProtocol,
    ContainerStructure,
    NodeType,
)
from virtuals.view import (
    ChildNavigationBase,
    ChildNestedGetBase,
    ChildNestedSetBase,
    ChildObservableBase,
    ChildPrimitiveSetBase,
    DescendantsObservableBase,
    KeyNotFoundError,
    LazyChildReadBase,
    LiveChildrenCountBase,
    ObservableBase,
    PrimitiveOpsBase,
    UnsafePrimitiveOpsBase,
    ViewBase,
    is_view_key,
)

from ._keys_view import ReversibleKeysView


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
    from virtuals.types import Empty

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
    LiveChildrenCountBase,
    ChildNavigationBase[str | int],
    ChildNestedGetBase,
    ChildNestedSetBase,
    ChildPrimitiveSetBase,
    LazyChildReadBase,
    PrimitiveOpsBase,
    UnsafePrimitiveOpsBase,
    ViewBase,
    ReactiveMappingBase[str | int, object],
    DictMethods[str | int, object],
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
        """Check if key exists. A key the view cannot hold is not in it."""
        return is_view_key(obj) and self.container.exists_child(obj, strict=True)

    def __iter__(self) -> Generator[str | int, None, None]:
        """Iterate over keys."""
        yield from self.container.iter_child_keys(validate=False, strict=True)

    def __reversed__(self) -> Generator[str | int, None, None]:
        """Keys last first: one reverse scan."""
        return self.reversed_keys()

    def keys(
        self, *, after: str | int | None = None, limit: int | None = None
    ) -> KeysView[str | int] | Generator[str | int, None, None]:
        """Keys in key order (ints first, then strs bytewise).

        With no arguments: a set-like KeysView supporting len, contains and
        set operations. With ``after`` or ``limit``: a generator, a page of
        the keys strictly after ``after`` (from the start when None), at
        most ``limit`` of them.

        ``after`` is a position in key order, not a row: it does not have
        to exist, so a page resumes after a key that was deleted since.
        Cost: one seek plus O(limit) rows, whatever the dict's size.

        Raises:
            ContainerNotFoundError: On a never-written view, when nothing
                is read (as iteration does).
        """
        if after is None and limit is None:
            return ReversibleKeysView(self)
        return self._keys_page(cursor=after, limit=limit, reverse=False)

    def reversed_keys(
        self, *, before: str | int | None = None, limit: int | None = None
    ) -> Generator[str | int, None, None]:
        """Keys in reverse key order, strictly before ``before``, at most ``limit``.

        From the end when ``before`` is None. Like ``keys(after=)``, the
        cursor does not have to exist. Cost: one seek plus O(limit) rows.
        """
        return self._keys_page(cursor=before, limit=limit, reverse=True)

    def _keys_page(
        self, *, cursor: str | int | None, limit: int | None, reverse: bool
    ) -> Generator[str | int, None, None]:
        """One page of keys. A bad cursor raises at the call, not on the first read."""
        if cursor is not None and not is_view_key(cursor):
            raise TypeError(f"view keys must be str or int, got {type(cursor).__name__}")
        return self.container.iter_child_keys(
            validate=False, strict=True, reverse=reverse, cursor=cursor, limit=limit
        )

    # =========================================================================
    # MUTATIONS (same for both facets)
    # =========================================================================

    def __setitem__(self, address: str | int, value: object) -> None:
        """Set value for key."""
        self._set_child_value(address, value)

    def __delitem__(self, address: str | int) -> None:
        """Delete key (and its subtree). KeyError if the key is missing, like a dict.

        A delete never materializes the view: on an absent view it raises
        after one existence read, so nothing is written and no parent
        bookkeeping fires.
        """
        if not is_view_key(address) or not self.container.exists_child(address):
            raise KeyNotFoundError(address)
        self.container.delete_child(address)

    def clear(self) -> None:
        """Remove all items."""
        if not self.container.exists():
            return  # clear of nothing is not a write: the view stays absent
        self.ensure_created()
        self.container.clear_children(validate=True)

    def update(self, other: object = (), /, **kwargs: object) -> None:
        """Update as ``dict.update``. The whole batch is checked for EMPTY before the first write."""
        items = self._update_items(other, kwargs)
        self._reject_empty_in(items)
        for key, value in items.items():
            self[key] = value

    def store(self, value: PyMapping[str | int, object], *, replace: bool = True) -> None:
        """Store dict contents."""
        value = self._reject_empty_in(value)  # type: ignore[assignment]
        self.ensure_created()
        if replace:
            self.clear()
        for key, val in value.items():
            self._set_child_value(key, val)

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

        Raises:
            ValueError: If ``value`` holds EMPTY anywhere. Checked before
                the view is materialized, so a rejected write leaves nothing
                behind.
        """
        value = self._reject_empty_at(address, value)
        self.ensure_created()
        self._populate_child_container(address, value, view_class=view_class)

    def _rebuild_bookkeeping(self) -> None:
        """Drop the stored counter older versions kept; length is live now."""
        self.container.delete_metadata("__len__")


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
        if not is_view_key(address):
            raise KeyNotFoundError(address)
        try:
            return self._get_child_value(address)
        except ContainerNotFoundError as e:
            raise KeyNotFoundError(address) from e

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
        for k, v in self.container.iter_children(validate=False, strict=True):
            if v.node_type == NodeType.PRIMITIVE:
                yield v.primitive_value
            elif v.node_type == NodeType.CONTAINER:
                yield self._get_child_value(k, node_info=v)

    def _iter_items(self) -> Generator[tuple[str | int, object], None, None]:
        """Efficient single-pass items iteration over storage."""
        for k, v in self.container.iter_children(validate=False, strict=True):
            if v.node_type == NodeType.PRIMITIVE:
                yield k, v.primitive_value
            elif v.node_type == NodeType.CONTAINER:
                yield k, self._get_child_value(k, node_info=v)

    def _iter_values_reverse(self) -> Generator[object, None, None]:
        """Efficient single-pass reverse value iteration over storage."""
        for k, v in self.container.iter_children(validate=False, strict=True, reverse=True):
            if v.node_type == NodeType.PRIMITIVE:
                yield v.primitive_value
            elif v.node_type == NodeType.CONTAINER:
                yield self._get_child_value(k, node_info=v)

    def _iter_items_reverse(self) -> Generator[tuple[str | int, object], None, None]:
        """Efficient single-pass reverse items iteration over storage."""
        for k, v in self.container.iter_children(validate=False, strict=True, reverse=True):
            if v.node_type == NodeType.PRIMITIVE:
                yield k, v.primitive_value
            elif v.node_type == NodeType.CONTAINER:
                yield k, self._get_child_value(k, node_info=v)

    def extract(self) -> dict[str | int, object]:
        """Extract all items as dict."""
        return dict(self.items())

    # =========================================================================
    # FACET NAVIGATION
    # =========================================================================

    @property
    def lazy(self) -> LazyDictView:
        """Switch to lazy facet — reads return child Views."""
        return LazyDictView(container=self.container, registry=self.registry, parent=self.parent)

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
        if not is_view_key(address):
            raise KeyNotFoundError(address)
        try:
            return self._get_child_view_or_value(address)
        except ContainerNotFoundError as e:
            raise KeyNotFoundError(address) from e

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
        for k, v in self.container.iter_children(validate=False, strict=True):
            if v.node_type == NodeType.PRIMITIVE:
                yield v.primitive_value
            elif v.node_type == NodeType.CONTAINER:
                yield self._get_child_view_or_value(k, node_info=v)

    def _iter_items(self) -> Generator[tuple[str | int, object], None, None]:
        """Efficient single-pass items iteration over storage."""
        for k, v in self.container.iter_children(validate=False, strict=True):
            if v.node_type == NodeType.PRIMITIVE:
                yield k, v.primitive_value
            elif v.node_type == NodeType.CONTAINER:
                yield k, self._get_child_view_or_value(k, node_info=v)

    def _iter_values_reverse(self) -> Generator[object, None, None]:
        """Efficient single-pass reverse value iteration over storage."""
        for k, v in self.container.iter_children(validate=False, strict=True, reverse=True):
            if v.node_type == NodeType.PRIMITIVE:
                yield v.primitive_value
            elif v.node_type == NodeType.CONTAINER:
                yield self._get_child_view_or_value(k, node_info=v)

    def _iter_items_reverse(self) -> Generator[tuple[str | int, object], None, None]:
        """Efficient single-pass reverse items iteration over storage."""
        for k, v in self.container.iter_children(validate=False, strict=True, reverse=True):
            if v.node_type == NodeType.PRIMITIVE:
                yield k, v.primitive_value
            elif v.node_type == NodeType.CONTAINER:
                yield k, self._get_child_view_or_value(k, node_info=v)

    # =========================================================================
    # FACET NAVIGATION
    # =========================================================================

    def extract(self) -> dict[str | int, object]:
        """The whole subtree as plain Python values, read through the eager facet."""
        return self.eager.extract()

    @property
    def eager(self) -> EagerDictView:
        """Switch to eager facet — reads return extracted values."""
        return EagerDictView(container=self.container, registry=self.registry, parent=self.parent)

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

    def __reversed__(self) -> Generator[object, None, None]:
        return _EagerReversedValuesView(self._mapping).__iter__()


class _EagerItemsView(ItemsView):
    """Efficient ItemsView — single-pass iteration over storage."""

    _mapping: EagerDictView

    def __iter__(self) -> Generator[tuple[str | int, object], None, None]:  # type: ignore[override]
        yield from self._mapping._iter_items()

    def __reversed__(self) -> Generator[tuple[str | int, object], None, None]:
        return _EagerReversedItemsView(self._mapping).__iter__()


class _LazyValuesView(ValuesView):
    """Efficient ValuesView — single-pass iteration, lazy child access."""

    _mapping: LazyDictView

    def __iter__(self) -> Generator[object, None, None]:  # type: ignore[override]
        yield from self._mapping._iter_values()

    def __reversed__(self) -> Generator[object, None, None]:
        return _LazyReversedValuesView(self._mapping).__iter__()


class _LazyItemsView(ItemsView):
    """Efficient ItemsView — single-pass iteration, lazy child access."""

    _mapping: LazyDictView

    def __iter__(self) -> Generator[tuple[str | int, object], None, None]:  # type: ignore[override]
        yield from self._mapping._iter_items()

    def __reversed__(self) -> Generator[tuple[str | int, object], None, None]:
        return _LazyReversedItemsView(self._mapping).__iter__()


class _EagerReversedValuesView(ValuesView):
    """Reverse-order ValuesView — single-pass reverse iteration over storage."""

    _mapping: EagerDictView

    def __iter__(self) -> Generator[object, None, None]:  # type: ignore[override]
        yield from self._mapping._iter_values_reverse()


class _EagerReversedItemsView(ItemsView):
    """Reverse-order ItemsView — single-pass reverse iteration over storage."""

    _mapping: EagerDictView

    def __iter__(self) -> Generator[tuple[str | int, object], None, None]:  # type: ignore[override]
        yield from self._mapping._iter_items_reverse()


class _LazyReversedValuesView(ValuesView):
    """Reverse-order ValuesView — single-pass reverse iteration, lazy child access."""

    _mapping: LazyDictView

    def __iter__(self) -> Generator[object, None, None]:  # type: ignore[override]
        yield from self._mapping._iter_values_reverse()


class _LazyReversedItemsView(ItemsView):
    """Reverse-order ItemsView — single-pass reverse iteration, lazy child access."""

    _mapping: LazyDictView

    def __iter__(self) -> Generator[tuple[str | int, object], None, None]:  # type: ignore[override]
        yield from self._mapping._iter_items_reverse()


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
