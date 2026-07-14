"""Kh57View — sparse int-keyed map backed by kh57 range reservoir sampling.

Layout under the view's container::

    site/
    ├── __len__ (metadata)
    ├── <kh57(int_key_1)>: value
    ├── <kh57(int_key_2)>: value
    └── ...

Items live directly as int-keyed children where the child key is
``kh57(int_key)``. Storage sorts them by kh57 output: sparsest level
first, then denser levels; within a level, original int order is
preserved. This lays items out so ``kh57.sample`` can walk a query
range level by level with low read amplification.

Semantically a sparse ``Mapping[int, V]``:
    view[42] = value        # put by original int key
    view[42]                # get
    del view[42]            # delete
    iter(view)              # yields keys in original int order
    view.sample(n, ...)     # kh57 range reservoir sample
    view.range(begin, end)  # ordered iterator over a sub-range

Provides three classes following the eager/lazy facet pattern:
- Kh57ViewBase: shared mutations, lifecycle, __len__, __contains__
- EagerKh57View: reads return extracted Python values
- LazyKh57View: reads return child Views for containers, values for primitives
"""

from __future__ import annotations

import heapq
from collections.abc import ItemsView, KeysView, ValuesView
from typing import TYPE_CHECKING, ClassVar, cast

from kh57 import kh57
from kh57 import sample as kh57_sample

from virtuals.container import Container, ContainerProtocol, ContainerStructure, NodeType, node_ops
from virtuals.tkv.filter import LengthFilter, PrefixFilter
from virtuals.tkv.storage import StorageScanOptions
from virtuals.types import EMPTY, Empty, Value
from virtuals.view import (
    ChildNavigationBase,
    ChildNestedGetBase,
    ChildNestedSetBase,
    ChildObservableBase,
    ChildPrimitiveSetBase,
    LazyChildReadBase,
    MetadataBasedChildrenCountBase,
    ObservableBase,
    PrimitiveOpsBase,
    UnsafePrimitiveOpsBase,
    ViewBase,
)


if TYPE_CHECKING:
    import random
    from collections.abc import Callable, Generator, Iterator
    from collections.abc import Mapping as PyMapping

    from virtuals.container.types import NodeInfo
    from virtuals.view import View


__all__ = [
    "EagerKh57View",
    "Kh57ViewBase",
    "LazyKh57View",
]


_LEVEL_BITS = 57
_KEY_MASK = (1 << _LEVEL_BITS) - 1
# kh57 encodes level as bit_length(hash) - 1 of a 64-bit siphash output, so
# in practice the top bit stays 0 and only levels 0..63 hold data. We walk
# up to level 63 for iteration; kh57.sample() itself scans up to level 127
# via the backend, which safely returns empty for out-of-range slices.
_LEVELS = 64
_MAX_INT64_PLUS_1 = 1 << 63


# =============================================================================
# BACKEND ADAPTER — kh57.Backend over a virtuals container child
# =============================================================================


class _Kh57ContainerBackend:
    """Adapts a virtuals container to the kh57 Backend protocol.

    Keys are 8-byte big-endian kh57 encoded ints, stored as int child
    segments (order-preserving via the binary key codec). Values pass
    through unchanged — kh57's algorithm never inspects them, so the
    adapter never bytes-encodes user values.

    Reads produced by ``range_scan`` come out already resolved for the
    caller's facet: eager views get extracted Python values, lazy views
    get child Views for containers and raw values for primitives. That
    resolution happens through ``value_reader``, supplied by the view.
    """

    def __init__(
        self,
        view: Kh57ViewBase,
        value_reader: Callable[[int, NodeInfo], object],
    ) -> None:
        """Wire adapter to a view and its facet-specific value reader."""
        self._view = view
        self._read_value = value_reader

    def get(self, key: bytes) -> object | None:
        """Return the value for `key`, or None if absent."""
        key_int = int.from_bytes(key, "big")
        child_site = (*self._view.container.site, key_int)
        info = node_ops.get_node_info(child_site, self._view.container.ctx)
        if not info.exists or info.node_type == NodeType.NOT_FOUND:
            return None
        if info.node_type == NodeType.PRIMITIVE:
            return info.primitive_value
        return self._read_value(key_int, info)

    def put(self, key: bytes, value: object) -> None:
        """Insert or overwrite `key` with `value`."""
        key_int = int.from_bytes(key, "big")
        self._view._put_encoded(key_int, value)

    def delete(self, key: bytes) -> None:
        """Remove `key` if present."""
        key_int = int.from_bytes(key, "big")
        self._view._delete_encoded(key_int)

    def range_scan(
        self,
        lo: bytes,
        hi: bytes,
    ) -> Iterator[tuple[bytes, object]]:
        """Yield (key, value) pairs with lo <= key < hi, ascending by key."""
        lo_int = int.from_bytes(lo, "big")
        hi_int = int.from_bytes(hi, "big")
        for encoded_int, info in self._view._scan_encoded_range(lo_int, hi_int):
            encoded_bytes = encoded_int.to_bytes(8, "big")
            if info.node_type == NodeType.PRIMITIVE:
                yield encoded_bytes, info.primitive_value
            else:
                yield encoded_bytes, self._read_value(encoded_int, info)


# =============================================================================
# BASE — shared by eager and lazy facets
# =============================================================================


class Kh57ViewBase(
    ObservableBase,
    ChildObservableBase[int],
    MetadataBasedChildrenCountBase,
    ChildNavigationBase[int],
    ChildNestedGetBase,
    ChildNestedSetBase,
    ChildPrimitiveSetBase,
    LazyChildReadBase,
    PrimitiveOpsBase,
    UnsafePrimitiveOpsBase,
    ViewBase,
):
    """Sparse int-keyed map with kh57-encoded layout for range sampling.

    Items are stored under int child segments equal to ``kh57(int_key)``,
    so storage order matches kh57's level-first layout. Length tracked
    via metadata. Shared by eager and lazy facets.

    Example::

        >>> view = nav.root(ctx).open_child("events", EagerKh57View)
        >>> view[100] = {"ts": 1}
        >>> view[42] = {"ts": 2}
        >>> list(view)              # original int order
        [42, 100]
        >>> view.sample(1)          # deterministic-ish reservoir sample
        [(42, {'ts': 2})]
    """

    STRUCTURE: ClassVar[ContainerStructure] = ContainerStructure(16)
    PROTOCOL: ClassVar[ContainerProtocol] = ContainerProtocol.MAPPING | ContainerProtocol.MUTABLE

    @classmethod
    def is_address_static(cls, address: object) -> bool:
        """Non-negative ints pass through; nothing else is a valid view key."""
        return isinstance(address, int) and address >= 0

    def normalize_address(self, address: int) -> int:
        """Encode an original int key to its kh57 child segment."""
        if not isinstance(address, int):
            raise TypeError(f"Kh57View keys must be int, got {type(address).__name__}")
        return kh57(address)

    # -- Internal encoded-key ops -----------------------------------------

    def _put_encoded(
        self,
        encoded: int,
        value: object,
        view_class: type | None = None,
    ) -> None:
        """Write `value` at the kh57-encoded child segment `encoded`.

        If `view_class` is given, force the container child to that layout;
        otherwise fall back to the registry's type-based dispatch. Callers
        that already know the child layout (Refs via
        :meth:`set_child_container_as`) pass it explicitly.
        """
        self.ensure_created()
        is_new = not self.container.exists_child(encoded)
        if self.registry.is_container_type(value):
            self._populate_child_container(encoded, value, view_class=view_class)
        else:
            self.container.put_child_primitive(encoded, cast("Value", value))
        if is_new:
            self._increment_length()

    def set_child_container_as(
        self,
        address: int,
        value: object,
        view_class: type,
    ) -> None:
        """Peer of ``__setitem__`` with an explicit child view class.

        Same as ``__setitem__`` — encode address via ``kh57`` and populate
        — but takes ``view_class`` explicitly so nested Refs (e.g. a
        ``ShapeRef`` inside a ``Kh57ShapesRef``) get their declared layout
        instead of the registry's default for the Python value's type.
        """
        self._put_encoded(self.normalize_address(address), value, view_class=view_class)

    def _delete_encoded(self, encoded: int) -> None:
        """Remove the child at `encoded` if present. Silent on miss."""
        self.ensure_created()
        if not self.container.exists_child(encoded):
            return
        self.container.delete_child(encoded)
        self._decrement_length()

    def _scan_encoded_range(
        self,
        lo_int: int,
        hi_int: int,
    ) -> Generator[tuple[int, NodeInfo], None, None]:
        """Yield (encoded_int, NodeInfo) for children with encoded in [lo_int, hi_int).

        Uses a bounded scan starting at `lo_int` and breaks when past the
        container's key prefix. Non-int children are skipped, and we stop
        as soon as an encoded key crosses `hi_int`.
        """
        if lo_int >= hi_int:
            return
        # kh57 encoded values fit in int64 (top bit is 0). Bounds beyond
        # int64 come from kh57.sample's high-level probes and hold no data.
        if lo_int >= _MAX_INT64_PLUS_1:
            return
        if hi_int > _MAX_INT64_PLUS_1:
            hi_int = _MAX_INT64_PLUS_1
        site = self.container.site
        prefix = PrefixFilter(prefix=site)
        child_len = LengthFilter(length=len(site) + 1)
        opts = StorageScanOptions(
            start=(*site, lo_int),
            break_filter=prefix,
            filter=prefix & child_len,
        )
        from virtuals.container.context import require_read_context

        rctx = require_read_context(self.container.ctx)
        for k, v in rctx.scan(opts).items():
            seg = k[-1]
            if not isinstance(seg, int):
                continue
            if seg < lo_int:
                # start is inclusive but the codec may seek slightly early
                continue
            if seg >= hi_int:
                return
            yield seg, node_ops.get_node_info(k, self.container.ctx, raw_value=v)

    def _extract_container_value(self, encoded: int, info: NodeInfo) -> object:
        """Eager reader: extract a container child by encoded segment."""
        from virtuals.collections import Convertible

        child_site = (*self.container.site, encoded)
        child_container = Container(ctx=self.container.ctx, site=child_site)
        if info.structure is None:
            raise ValueError(f"Child container at {encoded} has no structure ID")
        view_class = self.registry.get_view_for_structure(info.structure)
        child_view = view_class(container=child_container, registry=self.registry)
        if not isinstance(child_view, Convertible):
            raise TypeError(f"Child view {view_class.__name__} does not support extraction")
        return child_view.extract()

    def _view_container_value(self, encoded: int, info: NodeInfo) -> object:
        """Lazy reader: return a child View for a container child."""
        child_site = (*self.container.site, encoded)
        child_container = Container(ctx=self.container.ctx, site=child_site)
        if info.structure is None:
            raise ValueError(f"Child container at {encoded} has no structure ID")
        view_class = self.registry.get_view_for_structure(info.structure)
        return view_class(container=child_container, registry=self.registry)

    # -- Membership -------------------------------------------------------

    def __contains__(self, obj: int) -> bool:
        """True if an item is stored under original int key `obj`."""
        if not isinstance(obj, int):
            return False
        return self.container.exists_child(kh57(obj))

    # -- Mutations --------------------------------------------------------

    def __setitem__(self, address: int, value: object) -> None:
        """Store `value` under original int key `address`."""
        self._put_encoded(self.normalize_address(address), value)

    def __delitem__(self, address: int) -> None:
        """Delete item at original int key `address`."""
        encoded = self.normalize_address(address)
        if not self.container.exists_child(encoded):
            raise KeyError(address)
        self.container.delete_child(encoded)
        self._decrement_length()

    def clear(self) -> None:
        """Remove all items."""
        self.ensure_created()
        self.container.clear_children(validate=False)
        self._set_length(0)

    def update(
        self,
        other: PyMapping[int, object] | None = None,
        **kwargs: object,
    ) -> None:
        """Update from mapping or kwargs — int keys only."""
        if other:
            for key, value in other.items():
                self[key] = value
        for key, value in kwargs.items():
            self[key] = value  # type: ignore[assignment,index]

    def store(self, value: PyMapping[int, object], *, replace: bool = True) -> None:
        """Bulk-load mapping contents under kh57-encoded segments."""
        self.ensure_created()
        if replace and len(self) > 0:
            self.clear()
        count = len(self)
        for key, val in value.items():
            if not isinstance(key, int) or key < 0:
                raise TypeError(f"Kh57View keys must be non-negative int, got {key!r}")
            encoded = kh57(key)
            is_new = not self.container.exists_child(encoded)
            self._set_child_value(encoded, val)
            if is_new:
                count += 1
        self._set_length(count)

    # -- Iteration in original int order ---------------------------------

    def _iter_level_keys(self, level: int) -> Generator[int, None, None]:
        """Yield original int keys stored on `level`, in original int order."""
        base = level << _LEVEL_BITS
        hi_int = base + (1 << _LEVEL_BITS)
        for encoded, _info in self._scan_encoded_range(base, hi_int):
            yield encoded & _KEY_MASK

    def __iter__(self) -> Generator[int, None, None]:
        """Yield keys in original int order via k-way merge across levels."""
        level_iters = [self._iter_level_keys(level) for level in range(_LEVELS)]
        yield from heapq.merge(*level_iters)

    # -- Range ------------------------------------------------------------

    def _iter_level_items(
        self,
        level: int,
        begin: int,
        end: int,
    ) -> Generator[tuple[int, NodeInfo], None, None]:
        """Yield (original_key, NodeInfo) for `level` restricted to [begin, end)."""
        base = level << _LEVEL_BITS
        lo_int = base + begin
        hi_int = base + end
        for encoded, info in self._scan_encoded_range(lo_int, hi_int):
            yield encoded & _KEY_MASK, info

    def range(
        self,
        begin: int,
        end: int,
    ) -> Generator[tuple[int, object], None, None]:
        """Yield (int_key, value) pairs with begin <= int_key < end, ascending."""
        if begin < 0:
            raise ValueError("begin must be non-negative")
        if end > (1 << _LEVEL_BITS):
            raise ValueError(f"end must be <= 2**{_LEVEL_BITS}")
        if begin >= end:
            return
        level_iters = [self._iter_level_items(level, begin, end) for level in range(_LEVELS)]
        for key, info in heapq.merge(*level_iters, key=lambda kv: kv[0]):
            if info.node_type == NodeType.PRIMITIVE:
                yield key, info.primitive_value
            else:
                yield key, self._read_container_value(kh57(key), info)

    # Subclasses override this — eager extracts, lazy views. Default eager.
    def _read_container_value(self, encoded: int, info: NodeInfo) -> object:
        """Facet hook: turn a container NodeInfo into a user-facing value."""
        return self._extract_container_value(encoded, info)

    # -- Backend / sample -------------------------------------------------

    def _make_backend(self) -> _Kh57ContainerBackend:
        """Build a kh57.Backend adapter using this facet's read semantics."""
        return _Kh57ContainerBackend(self, self._read_container_value)

    def sample(
        self,
        n: int,
        begin: int | None = None,
        end: int | None = None,
        *,
        rng: random.Random | None = None,
    ) -> list[tuple[int, object]]:
        """Return up to `n` uniform (int_key, value) samples from [begin, end).

        Deterministic given a seeded `rng`. Stable under appends outside
        the queried range. Delegates to ``kh57.sample`` via the container
        adapter.
        """
        return kh57_sample(self._make_backend(), n, begin, end, rng=rng)

    def get(self, address: int, default: object | Empty = EMPTY) -> object | Empty:
        """Return item at `address` or `default` if absent."""
        try:
            return self[address]  # type: ignore[index]
        except KeyError:
            return default

    # -- Mapping surface --------------------------------------------------

    def keys(self) -> KeysView:
        """Return an iterable of int keys in original int order."""
        return KeysView(self)

    def values(self) -> ValuesView:
        """Return an iterable of values in original-key order."""
        return ValuesView(self)

    def items(self) -> ItemsView:
        """Return an iterable of (int_key, value) pairs in original-key order."""
        return ItemsView(self)

    # -- Navigation -------------------------------------------------------

    def open_child(self, address: int, view: type) -> View:  # type: ignore[override]
        """Open a child view under the kh57-encoded segment of `address`."""
        encoded = kh57(address)
        child_site = (*self.container.site, encoded)
        child_container = Container(ctx=self.container.ctx, site=child_site)
        return view(child_container, self.registry)


# =============================================================================
# EAGER FACET — reads return extracted Python values
# =============================================================================


class EagerKh57View(Kh57ViewBase):
    """Eager kh57-encoded map — reads return extracted Python values.

    Cross-navigate to lazy facet via .lazy property.
    """

    CONTAINER_CLS: ClassVar[type | None] = None

    def _read_container_value(self, encoded: int, info: NodeInfo) -> object:
        """Eager: extract child container contents to a Python value."""
        return self._extract_container_value(encoded, info)

    def __getitem__(self, address: int) -> object:
        """Get extracted Python value at original int key `address`."""
        encoded = kh57(address)
        child_site = (*self.container.site, encoded)
        info = node_ops.get_node_info(child_site, self.container.ctx)
        if not info.exists or info.node_type == NodeType.NOT_FOUND:
            raise KeyError(address)
        if info.node_type == NodeType.PRIMITIVE:
            from virtuals.types import is_empty

            if is_empty(info.primitive_value):
                raise KeyError(address)
            return info.primitive_value
        return self._extract_container_value(encoded, info)

    def items(self) -> Generator[tuple[int, object], None, None]:
        """Yield (int_key, value) pairs in original int order."""
        level_iters = [
            self._iter_level_items(level, 0, 1 << _LEVEL_BITS) for level in range(_LEVELS)
        ]
        for key, info in heapq.merge(*level_iters, key=lambda kv: kv[0]):
            if info.node_type == NodeType.PRIMITIVE:
                yield key, info.primitive_value
            else:
                yield key, self._extract_container_value(kh57(key), info)

    def values(self) -> Generator[object, None, None]:
        """Yield values in original int-key order."""
        for _k, v in self.items():
            yield v

    def extract(self) -> dict[int, object]:
        """Materialize as a plain dict keyed by original int keys."""
        return dict(self.items())

    def pop(self, address: int, default: object | Empty = EMPTY) -> object | Empty:
        """Remove item at ``address`` and return its extracted value."""
        from virtuals.types import is_empty

        try:
            value = self[address]
            del self[address]
            return value
        except KeyError:
            if is_empty(default):
                raise
            return default

    # -- Facet navigation ------------------------------------------------

    @property
    def lazy(self) -> LazyKh57View:
        """Switch to lazy facet — reads return child Views."""
        return LazyKh57View(container=self.container, registry=self.registry)

    @property
    def eager(self) -> EagerKh57View:
        """Identity — already eager."""
        return self


# =============================================================================
# LAZY FACET — reads return child Views
# =============================================================================


class LazyKh57View(Kh57ViewBase):
    """Lazy kh57-encoded map — reads return child Views for containers.

    Cross-navigate to eager facet via .eager property.
    """

    def _read_container_value(self, encoded: int, info: NodeInfo) -> object:
        """Lazy: return a child View for the container child, no extraction."""
        return self._view_container_value(encoded, info)

    def __getitem__(self, address: int) -> object:
        """Get child — View for containers, raw value for primitives."""
        encoded = kh57(address)
        child_site = (*self.container.site, encoded)
        info = node_ops.get_node_info(child_site, self.container.ctx)
        if not info.exists or info.node_type == NodeType.NOT_FOUND:
            raise KeyError(address)
        if info.node_type == NodeType.PRIMITIVE:
            from virtuals.types import is_empty

            if is_empty(info.primitive_value):
                raise KeyError(address)
            return info.primitive_value
        return self._view_container_value(encoded, info)

    def items(self) -> Generator[tuple[int, object], None, None]:
        """Yield (int_key, value_or_view) pairs in original int order."""
        level_iters = [
            self._iter_level_items(level, 0, 1 << _LEVEL_BITS) for level in range(_LEVELS)
        ]
        for key, info in heapq.merge(*level_iters, key=lambda kv: kv[0]):
            if info.node_type == NodeType.PRIMITIVE:
                yield key, info.primitive_value
            else:
                yield key, self._view_container_value(kh57(key), info)

    # -- Facet navigation ------------------------------------------------

    @property
    def eager(self) -> EagerKh57View:
        """Switch to eager facet — reads return extracted values."""
        return EagerKh57View(container=self.container, registry=self.registry)

    @property
    def lazy(self) -> LazyKh57View:
        """Identity — already lazy."""
        return self
