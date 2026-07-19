"""LogIndexedDictView — dict view with append-only log key index.

Layout under the view's container::

    site/
    ├── __keys__/    # append-only log (timestamp+uuid entries)
    │   ├── 1711446123456-a1b2c3: "mint1"
    │   ├── 1711446123457-d4e5f6: "mint2"
    │   └── ...
    └── __data__/    # actual dict children (nested, with controlled granularity)
        ├── mint1/
        │   ├── field1: value
        │   └── field2: {compound}   ← primitive if declared so
        └── mint2/
            └── ...

Multi-writer safe: each writer generates unique log keys independently
(timestamp_ms + uuid fragment). No shared state, no append contention.

No ``__len__`` metadata. Length is O(n) via ``__keys__/`` scan.

Ordered enumeration via ``__keys__/`` prefix scan (lexicographic =
chronological). Direct lookup via ``__data__.[key]`` (skips ``__keys__``).

Cursor-friendly: ``keys(after=cursor)`` seeks into ``__keys__/`` log.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import ItemsView, KeysView, MutableMapping, ValuesView
from typing import TYPE_CHECKING, ClassVar, cast

from virtuals.container import Container, ContainerProtocol, ContainerStructure, NodeType
from virtuals.tkv.filter import LengthFilter, PrefixFilter, WildcardFilter
from virtuals.tkv.observer import SubscriptionOptions
from virtuals.tkv.storage import StorageScanOptions
from virtuals.types import EMPTY, Empty, Value
from virtuals.view import (
    ChildNavigationBase,
    ChildNestedGetBase,
    ChildNestedSetBase,
    ChildObservableBase,
    ChildPrimitiveSetBase,
    DescendantsObservableBase,
    LazyChildReadBase,
    ObservableBase,
    PrimitiveOpsBase,
    UnsafePrimitiveOpsBase,
    ViewBase,
)


if TYPE_CHECKING:
    from collections.abc import Generator
    from collections.abc import Mapping as PyMapping

    from virtuals.view import View


__all__ = [
    "EagerLogIndexedDictView",
    "LazyLogIndexedDictView",
    "LogIndexedDictViewBase",
]

_KEYS = "__keys__"
_DATA = "__data__"


_log_key_lock = threading.Lock()
_log_key_counter = 0


def _generate_log_key() -> str:
    """Generate a unique, monotonically increasing log key.

    Format: {timestamp_ms}-{counter:06d}-{uuid_fragment}

    - Timestamp provides cross-writer chronological ordering.
    - Counter (per-process monotonic) guarantees ordering within a writer
      even when multiple writes happen in the same millisecond.
    - UUID fragment breaks ties across independent writers/processes.
    """
    global _log_key_counter
    ts = int(time.time() * 1000)
    with _log_key_lock:
        _log_key_counter += 1
        counter = _log_key_counter
    uid = uuid.uuid4().hex[:6]
    return f"{ts:013d}-{counter:06d}-{uid}"


# =============================================================================
# BASE — shared by eager and lazy facets
# =============================================================================


class LogIndexedDictViewBase(
    ObservableBase,
    ChildObservableBase[str | int],
    DescendantsObservableBase,
    ChildNavigationBase[str | int],
    ChildNestedGetBase,
    ChildNestedSetBase,
    ChildPrimitiveSetBase,
    LazyChildReadBase,
    PrimitiveOpsBase,
    UnsafePrimitiveOpsBase,
    ViewBase,
):
    """Dict view with append-only log key index for multi-writer safety.

    Uses ``__keys__/`` as a timestamp+uuid append-only log and ``__data__/``
    for nested data with controlled granularity. No ``__len__`` metadata.

    Multi-writer safe: each writer generates unique log keys independently.
    Cursor-friendly: ``keys(after=cursor)`` for seek-based iteration.
    """

    STRUCTURE: ClassVar[ContainerStructure] = ContainerStructure(15)
    PROTOCOL: ClassVar[ContainerProtocol] = ContainerProtocol.MAPPING | ContainerProtocol.MUTABLE

    @classmethod
    def is_address_static(cls, address: object) -> bool:
        return False

    def normalize_address(self, address: str | int) -> str | int:
        return address

    # -- Internal containers -----------------------------------------------

    def _data_container(self) -> Container:
        """Container for actual dict data children."""
        return Container(
            ctx=self.container.ctx,
            site=(*self.container.site, _DATA),
        )

    def _keys_container(self) -> Container:
        """Container for the append-only log key index."""
        return Container(
            ctx=self.container.ctx,
            site=(*self.container.site, _KEYS),
        )

    def _ensure_internal_layout(self) -> None:
        """Materialize ``__data__/`` and ``__keys__/`` sub-containers.

        Called from ``ensure_created`` (via the base hook) after this view's
        own marker is stamped. Idempotent.
        """
        self.container.create_child_container(
            _DATA,
            structure=ContainerStructure(1),
            protocol=ContainerProtocol.MAPPING | ContainerProtocol.MUTABLE,
            validate=False,
        )
        self.container.create_child_container(
            _KEYS,
            structure=ContainerStructure(1),
            protocol=ContainerProtocol.MAPPING | ContainerProtocol.MUTABLE,
            validate=False,
        )

    # -- Log key operations ------------------------------------------------

    def _append_log_key(self, actual_key: str | int) -> str:
        """Append a key to the log index. Returns the generated log key."""
        log_key = _generate_log_key()
        kc = self._keys_container()
        kc.put_child_primitive(log_key, actual_key)
        return log_key

    # -- Observability overrides -------------------------------------------
    # LogIndexedDictView stores data under `__data__/` and log keys under
    # `__keys__/`, not directly under `container.site`. The base observable
    # methods (ChildObservableBase, ObservableBase) build filters at
    # `(container.site, ...)`, so no notifs would ever match real writes.
    # These overrides rewrite the filter sites so callers can watch the
    # view the way they'd watch a plain dict.

    def on_change(self) -> SubscriptionOptions:
        """Any data-side write. Scoped to __data__/ so log-key appends
        (bookkeeping noise) don't fire the callback."""
        return SubscriptionOptions(PrefixFilter(prefix=(*self.container.site, _DATA)))

    def on_child_change(self, address: str | int) -> SubscriptionOptions:
        """Any write at or under __data__/<address> -- matches primitive
        set/replace on this child AND nested field writes for compound
        (shape) children."""
        normalized = self.normalize_address(address)
        child_data_site = (*self.container.site, _DATA, normalized)
        return SubscriptionOptions(PrefixFilter(prefix=child_data_site))

    def on_children_change(self) -> SubscriptionOptions:
        """Fires once per new key ever added. Watches __keys__/ appends,
        which happen exactly once when `__setitem__` sees `is_new`. New
        mints appear here; replacements of an existing key don't."""
        keys_site = (*self.container.site, _KEYS)
        return SubscriptionOptions(WildcardFilter(pattern=(*keys_site, "*")))

    def on_descendants_change(
        self,
        address: object,
        *addresses: object,
    ) -> SubscriptionOptions:
        """Wildcard match under __data__/. Callers pattern relative to the
        view's dict (e.g. ("*", "total_txs")); the __data__/ prefix is
        prepended so real writes match. Name matches nu.core.reactive's
        OnDescendantsChangeQuery (spelled with 'a')."""
        pattern = (address, *addresses)
        wildcard_site = (*self.container.site, _DATA, *pattern)
        return SubscriptionOptions(WildcardFilter(pattern=wildcard_site))

    def _scan_log_keys(
        self,
        after: str | None = None,
        *,
        reverse: bool = False,
        before: str | None = None,
        limit: int | None = None,
    ) -> Generator[tuple[str, str | int], None, None]:
        """Scan __keys__/ log entries.

        Args:
            after: forward-scan only -- start just past this cursor.
            reverse: scan newest-first instead of oldest-first.
            before: reverse-scan only -- start just below this cursor.
                If omitted for a reverse scan, uses a max-sentinel that
                sorts after any real log key.
            limit: optional cap on yielded rows (before-filter counting;
                honored by the storage layer).

        Yields:
            (log_key, actual_key) tuples in chosen chronological direction.
        """
        kc = self._keys_container()
        site = kc.site

        if reverse:
            # Reverse scans need an upper bound within our prefix range,
            # else seek_to_last drops us past our prefix and break_filter
            # trips on the first row. "\x7f" (DEL) sorts after every ASCII
            # char that shows up in log keys (digits, dash, hex), so it
            # bounds any real log key from above.
            cursor = before if before is not None else "\x7f"
            start = (*site, cursor)
        elif after is not None:
            # Forward: start just past the cursor (we skip the cursor row).
            start = (*site, after)
        else:
            start = site

        prefix = PrefixFilter(prefix=site)
        child_len = LengthFilter(length=len(site) + 1)
        opts = StorageScanOptions(
            start=start,
            reverse=reverse,
            limit=limit,
            break_filter=prefix,
            filter=prefix & child_len,
        )

        from virtuals.container.context import require_read_context

        rctx = require_read_context(kc.ctx)
        for key, value in rctx.scan(opts).items():
            log_key = key[-1]
            # Skip the exact-cursor row on either direction.
            if (after is not None and log_key == after) or (
                before is not None and log_key == before
            ):
                continue
            yield log_key, cast("Value", value)

    # -- Membership & keys -------------------------------------------------

    def __len__(self) -> int:
        return sum(1 for _ in self._scan_log_keys())

    def __contains__(self, obj: str | int) -> bool:
        return self._data_container().exists_child(obj)

    def __iter__(self) -> Generator[str | int, None, None]:
        for _log_key, actual_key in self._scan_log_keys():
            yield actual_key

    def __reversed__(self) -> Generator[str | int, None, None]:
        """Yield actual keys in reverse insertion order (newest first)."""
        for _log_key, actual_key in self._scan_log_keys(reverse=True):
            yield actual_key

    def keys(
        self,
        *,
        after: str | None = None,
        limit: int | None = None,
    ) -> KeysView[str | int] | Generator[str | int, None, None]:
        """Get keys in insertion order.

        Args:
            after: If provided, start iteration past this log-key cursor.
            limit: Optional cap on the number of keys yielded.
                Requires ``after`` or triggers a generator (KeysView has
                no natural cap).
        """
        if after is not None or limit is not None:
            return self._keys_scan(after=after, limit=limit)
        return KeysView(self)  # type: ignore[arg-type]

    def keys_reverse(
        self,
        *,
        before: str | None = None,
        limit: int | None = None,
    ) -> Generator[str | int, None, None]:
        """Yield actual keys in reverse insertion order (newest first).

        Args:
            before: If provided, start just below this log-key cursor
                (older-than semantics).
            limit: Optional cap on the number of keys yielded.

        Cost: O(limit) via a reverse rocksdb range scan over ``__keys__/``,
        so ``keys_reverse(limit=n)`` is safe against arbitrarily large
        streams -- the natural fit for tail-N reads on this view.
        """
        for _log_key, actual_key in self._scan_log_keys(
            reverse=True, before=before, limit=limit,
        ):
            yield actual_key

    def _keys_scan(
        self, *, after: str | None, limit: int | None,
    ) -> Generator[str | int, None, None]:
        """Yield actual keys with forward cursor + optional cap."""
        for _log_key, actual_key in self._scan_log_keys(after=after, limit=limit):
            yield actual_key

    def keys_with_log_keys(
        self, *, after: str | None = None
    ) -> Generator[tuple[str, str | int], None, None]:
        """Yield (log_key, actual_key) pairs for cursor tracking.

        This is the primary method for Stream consumers that need to
        advance a cursor.
        """
        yield from self._scan_log_keys(after=after)

    def next_key_after(self, cursor: str | None = None) -> tuple[str, str | int] | None:
        """Single seek + read: return the next (log_key, actual_key) after cursor.

        Returns None if no more items. This is the micro-snapshot primitive
        for cursor-based streaming - one call per snapshot, no long-lived
        iterators.
        """
        for entry in self._scan_log_keys(after=cursor):
            return entry  # first result only
        return None

    def get(self, address: str | int, default: object | Empty = EMPTY) -> object | Empty:
        try:
            return self[address]
        except (KeyError, Exception):
            return default

    # -- Mutations ---------------------------------------------------------

    def __setitem__(self, address: str | int, value: object) -> None:
        self.ensure_created()
        dc = self._data_container()
        is_new = not dc.exists_child(address)
        # Write to __data__
        if self.registry.is_container_type(value):
            from virtuals.collections import Initializable

            value_type = value.__class__
            view_class = self.registry.get_view_for_type(value_type)
            structure_id = view_class.get_structure()
            protocol_hints = view_class.get_protocol()
            child_container = dc.create_child_container(
                address,
                structure=ContainerStructure(structure_id),
                protocol=protocol_hints,
            )
            child_view = view_class(container=child_container, registry=self.registry)
            if not isinstance(child_view, Initializable):
                raise TypeError(f"Child view {view_class.__name__} does not support initialization")
            child_view.store(value)
        else:
            dc.put_child_primitive(address, cast("Value", value))
        if is_new:
            self._append_log_key(address)

    def set_primitive(self, address: str | int, value: object) -> None:
        """Store a compound value as a single primitive blob.

        Bypasses decomposition - stores the value as-is regardless of whether
        it would normally be decomposed into sub-keys. Use for controlled
        granularity (e.g., storing a list or dict as a single key).
        """
        self.ensure_created()
        dc = self._data_container()
        is_new = not dc.exists_child(address)
        dc.put_child_primitive(address, cast("Value", value))
        if is_new:
            self._append_log_key(address)

    def __delitem__(self, address: str | int) -> None:
        # No ensure_created here: deletes must not materialize the view as
        # a side effect. If the underlying __data__/ container doesn't
        # exist yet, exists_child returns False against raw storage and we
        # raise KeyError -- matching Python dict semantics and letting ref
        # callers no-op via their KeyError/IndexError catch.
        dc = self._data_container()
        if not dc.exists_child(address):
            raise KeyError(address)
        dc.delete_child(address)
        # Remove from __keys__ log: scan and delete the matching entry
        for log_key, actual_key in self._scan_log_keys():
            if actual_key == address:
                self._keys_container().delete_child(log_key)
                break

    def clear(self) -> None:
        self.ensure_created()
        self._data_container().clear_children(validate=False)
        self._keys_container().clear_children(validate=False)

    def set_child_container_as(
        self,
        address: str | int,
        value: object,
        view_class: type,
    ) -> None:
        """Peer of ``__setitem__`` with an explicit child view class.

        Same shape as ``__setitem__`` -- write into ``__data__/<address>``
        and log the key into ``__keys__/`` if new -- but takes ``view_class``
        explicitly instead of dispatching by Python value type. Used by Refs
        that carry ``view_type=view_class`` in their slot payload and know
        the child layout up front.
        """
        from virtuals.collections import Initializable

        self.ensure_created()
        dc = self._data_container()
        is_new = not dc.exists_child(address)
        structure_id = view_class.get_structure()
        protocol_hints = view_class.get_protocol()
        child_container = dc.create_child_container(
            address,
            structure=ContainerStructure(structure_id),
            protocol=protocol_hints,
        )
        child_view = view_class(container=child_container, registry=self.registry)
        if not isinstance(child_view, Initializable):
            raise TypeError(f"Child view {view_class.__name__} does not support initialization")
        child_view.store(value)
        if is_new:
            self._append_log_key(address)

    def append(self, value: object) -> str:
        """Append value under a freshly generated unique log key.

        Convenience for callers that treat the view as an append-only log
        without a natural key. The generated log key is used as both the
        actual dict key and the entry in ``__keys__/``; each writer picks
        an independently unique key so parallel appends never touch the
        same rocksdb row. Returns the generated key.
        """
        key = _generate_log_key()
        self[key] = value
        return key

    def update(self, other: PyMapping[str | int, object] | None = None, **kwargs: object) -> None:
        if other:
            for key, value in other.items():
                self[key] = value
        for key, value in kwargs.items():
            self[key] = value  # type: ignore[assignment]

    def store(self, value: PyMapping[str | int, object], *, replace: bool = True) -> None:
        self.ensure_created()
        if replace and any(True for _ in self._scan_log_keys()):
            self.clear()

        dc = self._data_container()
        for key, val in value.items():
            if self.registry.is_container_type(val):
                from virtuals.collections import Initializable

                value_type = val.__class__
                view_class = self.registry.get_view_for_type(value_type)
                structure_id = view_class.get_structure()
                protocol_hints = view_class.get_protocol()
                child_container = dc.create_child_container(
                    key,
                    structure=ContainerStructure(structure_id),
                    protocol=protocol_hints,
                )
                child_view = view_class(container=child_container, registry=self.registry)
                if not isinstance(child_view, Initializable):
                    raise TypeError(
                        f"Child view {view_class.__name__} does not support initialization"
                    )
                child_view.store(val)
            else:
                dc.put_child_primitive(key, cast("Value", val))
            self._append_log_key(key)

    # -- Navigation --------------------------------------------------------

    def open_child(self, address: str | int, view: type) -> View:  # type: ignore[override]
        """Open child view under __data__/."""
        normalized = self.normalize_address(address)
        dc = self._data_container()
        child_site = (*dc.site, normalized)
        child_container = Container(ctx=dc.ctx, site=child_site)
        return view(child_container, self.registry)

    # -- Internal read helpers ---------------------------------------------

    def _read_child_from_data(self, address: str | int) -> tuple[object, bool]:
        dc = self._data_container()
        child_site = (*dc.site, address)
        from virtuals.container import node_ops

        node_info = node_ops.get_node_info(child_site, dc.ctx)
        if not node_info.exists or node_info.node_type == NodeType.NOT_FOUND:
            raise KeyError(address)
        return node_info, node_info.node_type == NodeType.CONTAINER

    def _extract_data_child(self, address: str | int, node_info: object) -> object:
        from virtuals.collections import Convertible

        dc = self._data_container()
        child_site = (*dc.site, address)
        child_container = Container(ctx=dc.ctx, site=child_site)
        if node_info.structure is None:  # type: ignore[union-attr]
            raise ValueError(f"Child container '{address}' has no structure ID")
        view_class = self.registry.get_view_for_structure(node_info.structure)  # type: ignore[union-attr]
        child_view = view_class(container=child_container, registry=self.registry)
        if not isinstance(child_view, Convertible):
            raise TypeError(f"Child view {view_class.__name__} does not support extraction")
        return child_view.extract()

    def _view_data_child(self, address: str | int, node_info: object) -> object:
        dc = self._data_container()
        child_site = (*dc.site, address)
        child_container = Container(ctx=dc.ctx, site=child_site)
        if node_info.structure is None:  # type: ignore[union-attr]
            raise ValueError(f"Child container '{address}' has no structure ID")
        view_class = self.registry.get_view_for_structure(node_info.structure)  # type: ignore[union-attr]
        return view_class(container=child_container, registry=self.registry)


# =============================================================================
# EAGER FACET
# =============================================================================


class EagerLogIndexedDictView(LogIndexedDictViewBase):
    """Eager log-indexed dict view — reads return extracted Python values."""

    CONTAINER_CLS: ClassVar[type] = dict

    def __getitem__(self, address: str | int) -> object | Empty:
        node_info, is_container = self._read_child_from_data(address)
        if not is_container:
            from virtuals.types import is_empty

            if is_empty(node_info.primitive_value):  # type: ignore[union-attr]
                raise KeyError(address)
            return node_info.primitive_value  # type: ignore[union-attr]
        return self._extract_data_child(address, node_info)

    def values(self) -> ValuesView[object]:
        return ValuesView(self)  # type: ignore[arg-type]

    def items(self) -> ItemsView[str | int, object]:
        return ItemsView(self)  # type: ignore[arg-type]

    def pop(self, address: str | int, default: object | Empty = EMPTY) -> object | Empty:
        try:
            value = self[address]
            del self[address]
            return value
        except KeyError:
            from virtuals.types import is_empty

            if is_empty(default):
                raise
            return default

    def extract(self) -> dict[str | int, object]:
        return {k: self[k] for k in self}

    @property
    def lazy(self) -> LazyLogIndexedDictView:
        return LazyLogIndexedDictView(container=self.container, registry=self.registry)

    @property
    def eager(self) -> EagerLogIndexedDictView:
        return self


# =============================================================================
# LAZY FACET
# =============================================================================


class LazyLogIndexedDictView(LogIndexedDictViewBase):
    """Lazy log-indexed dict view — reads return child Views for containers."""

    def __getitem__(self, address: str | int) -> object:
        node_info, is_container = self._read_child_from_data(address)
        if not is_container:
            from virtuals.types import is_empty

            if is_empty(node_info.primitive_value):  # type: ignore[union-attr]
                raise KeyError(address)
            return node_info.primitive_value  # type: ignore[union-attr]
        return self._view_data_child(address, node_info)

    def values(self) -> ValuesView[object]:
        return ValuesView(self)  # type: ignore[arg-type]

    def items(self) -> ItemsView[str | int, object]:
        return ItemsView(self)  # type: ignore[arg-type]

    def extract(self) -> dict[str | int, object]:
        """Extract all items as a native dict.

        Delegates to the eager facet so `_extract_child_container` on a parent
        view works when this class is registered as the log-indexed variant
        (e.g., via `virtuals.views.LogIndexedDictView`).
        """
        return self.eager.extract()

    @property
    def eager(self) -> EagerLogIndexedDictView:
        return EagerLogIndexedDictView(container=self.container, registry=self.registry)

    @property
    def lazy(self) -> LazyLogIndexedDictView:
        return self


MutableMapping.register(EagerLogIndexedDictView)
