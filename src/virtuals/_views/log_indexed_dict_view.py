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

    def _ensure_layout(self) -> None:
        """Ensure both child containers exist."""
        self.ensure_created()
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

    def _scan_log_keys(
        self,
        after: str | None = None,
    ) -> Generator[tuple[str, str | int], None, None]:
        """Scan __keys__/ log entries, optionally starting after a cursor.

        Yields:
            (log_key, actual_key) tuples in chronological order.
        """
        kc = self._keys_container()
        site = kc.site

        if after is not None:
            # Seek past the cursor: start from (site..., after) and skip it
            start = (*site, after)
        else:
            start = site

        prefix = PrefixFilter(prefix=site)
        child_len = LengthFilter(length=len(site) + 1)
        opts = StorageScanOptions(
            start=start,
            break_filter=prefix,
            filter=prefix & child_len,
        )

        from virtuals.container.context import require_read_context

        rctx = require_read_context(kc.ctx)
        for key, value in rctx.scan(opts).items():
            log_key = key[-1]
            # Skip the cursor key itself when using after=
            if after is not None and log_key == after:
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

    def keys(
        self, *, after: str | None = None
    ) -> KeysView[str | int] | Generator[str | int, None, None]:
        """Get keys in insertion order.

        Args:
            after: If provided, return a generator starting after this cursor
                   (log key). Otherwise return a standard KeysView.
        """
        if after is not None:
            return self._keys_after(after)
        return KeysView(self)  # type: ignore[arg-type]

    def _keys_after(self, cursor: str) -> Generator[str | int, None, None]:
        """Yield actual keys starting after the given cursor."""
        for _log_key, actual_key in self._scan_log_keys(after=cursor):
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
        self._ensure_layout()
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
        self._ensure_layout()
        dc = self._data_container()
        is_new = not dc.exists_child(address)
        dc.put_child_primitive(address, cast("Value", value))
        if is_new:
            self._append_log_key(address)

    def __delitem__(self, address: str | int) -> None:
        self._ensure_layout()
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
        self._ensure_layout()
        self._data_container().clear_children(validate=False)
        self._keys_container().clear_children(validate=False)

    def update(self, other: PyMapping[str | int, object] | None = None, **kwargs: object) -> None:
        if other:
            for key, value in other.items():
                self[key] = value
        for key, value in kwargs.items():
            self[key] = value  # type: ignore[assignment]

    def store(self, value: PyMapping[str | int, object], *, replace: bool = True) -> None:
        self._ensure_layout()
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
