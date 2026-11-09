"""LogIndexedDictView — dict view with append-only log key index.

Layout under the view's container::

    site/
    ├── __keys__/    # append-only log (timestamp+uuid entries)
    │   ├── 1711446123456-a1b2c3: "mint1"
    │   ├── 1711446123457-d4e5f6: "mint2"
    │   └── ...
    ├── __log__/     # reverse index: actual key -> its log key (point delete)
    │   ├── mint1: 1711446123456-a1b2c3
    │   └── mint2: 1711446123457-d4e5f6
    └── __data__/    # actual dict children (nested, with controlled granularity)
        ├── mint1/
        │   ├── field1: value
        │   └── field2: {compound}   ← primitive if declared so
        └── mint2/
            └── ...

Multi-writer safe: each writer generates unique log keys independently
(timestamp_ms + uuid fragment). No shared state, no append contention.

No ``__len__`` metadata. Length is O(n) via ``__keys__/`` scan (rows are
primitives, so the scan is tight); ``bool()`` is one seek. Delete is a
point op: the reverse index gives the log key in one get. Data written
before ``__log__/`` existed has no reverse row and falls back to a log
scan on delete; ``rebuild_bookkeeping()`` fills the index.

Ordered enumeration via ``__keys__/`` prefix scan (lexicographic =
chronological). Direct lookup via ``__data__.[key]`` (skips ``__keys__``).

Cursors. ``keys(after=id, limit=)`` and ``keys_reverse(before=id, limit=)``
page by an item's own id: the reverse index gives its log key in one get,
then one seek into ``__keys__/``. An id that is gone raises KeyError, like
``FractionalIndexedDictView``: its log slot went with it. Streams that must
resume past deleted rows hold the log key instead
(``keys_with_log_keys(after=)``, ``next_key_after``).
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import ItemsView, KeysView, ValuesView
from typing import TYPE_CHECKING, ClassVar, cast

from virtuals.collections import DictMethods, ReactiveMappingBase
from virtuals.container import Container, ContainerProtocol, ContainerStructure, NodeType
from virtuals.tkv.filter import LengthFilter, PrefixFilter, WildcardFilter
from virtuals.tkv.observer import SubscriptionOptions
from virtuals.tkv.storage import StorageScanOptions
from virtuals.types import EMPTY, Empty, Value, is_empty
from virtuals.view import (
    ChildNavigationBase,
    ChildNestedGetBase,
    ChildNestedSetBase,
    ChildObservableBase,
    ChildPrimitiveSetBase,
    DescendantsObservableBase,
    KeyNotFoundError,
    LazyChildReadBase,
    ObservableBase,
    PrimitiveOpsBase,
    UnsafePrimitiveOpsBase,
    ViewBase,
    is_view_key,
)

from ._keys_view import ReversibleKeysView


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
_LOG = "__log__"
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
    ReactiveMappingBase[str | int, object],
    DictMethods[str | int, object],
):
    """Dict view with append-only log key index for multi-writer safety.

    Uses ``__keys__/`` as a timestamp+uuid append-only log and ``__data__/``
    for nested data with controlled granularity. No ``__len__`` metadata.

    Multi-writer safe: each writer generates unique log keys independently.
    Pages by item id: ``keys(after=id, limit=)``, ``keys_reverse(before=id, limit=)``.
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

    def _children_container(self) -> Container:
        """Children live under ``__data__/``."""
        return self._data_container()

    def _keys_container(self) -> Container:
        """Container for the append-only log key index."""
        return Container(
            ctx=self.container.ctx,
            site=(*self.container.site, _KEYS),
        )

    def _log_container(self) -> Container:
        """Container for the reverse index (actual key -> log key)."""
        return Container(
            ctx=self.container.ctx,
            site=(*self.container.site, _LOG),
        )

    def _ensure_internal_layout(self) -> None:
        """Materialize ``__data__/``, ``__keys__/`` and ``__log__/`` sub-containers.

        Called from ``ensure_created`` (via the base hook) after this view's
        own marker is stamped. Idempotent.
        """
        for name in (_DATA, _KEYS, _LOG):
            self.container.create_child_container(
                name,
                structure=ContainerStructure(1),
                protocol=ContainerProtocol.MAPPING | ContainerProtocol.MUTABLE,
                validate=False,
            )

    # -- Log key operations ------------------------------------------------

    def _append_log_key(self, actual_key: str | int) -> str:
        """Append a key to the log index. Returns the generated log key.

        One ``__keys__`` put and one ``__log__`` put (the reverse row that
        makes delete a point op).
        """
        log_key = _generate_log_key()
        self._keys_container().put_child_primitive(
            log_key, actual_key, validate=False, validate_parent=False
        )
        self._log_container().put_child_primitive(
            actual_key, log_key, validate=False, validate_parent=False
        )
        return log_key

    def _log_key_or_empty(self, actual_key: str | int) -> str | Empty:
        """Log key of ``actual_key`` from the reverse index, EMPTY if unrecorded. One get."""
        return cast("str | Empty", self._log_container().get_child_primitive_unsafe(actual_key))

    def _find_log_key(self, actual_key: str | int) -> str | Empty:
        """Log key of a key that holds data, EMPTY if the log never recorded it.

        One reverse-index get, one get to confirm the log row still names
        this key. A missing reverse row (data logged before ``__log__/``
        existed) or a stale one (a writer that does not maintain the index
        deleted and re-set the key) falls back to a log scan.
        """
        log_key = self._log_key_or_empty(actual_key)
        if not is_empty(log_key):
            held = self._keys_container().get_child_primitive_unsafe(cast("str", log_key))
            if not is_empty(held) and held == actual_key:
                return log_key
        for candidate, logged in self._scan_log_keys():
            if logged == actual_key:
                return candidate
        return EMPTY

    def _require_log_key(self, actual_key: str | int) -> str:
        """Log key of ``actual_key`` as a cursor. KeyError if the id holds no data."""
        if actual_key not in self:
            raise KeyNotFoundError(actual_key)
        log_key = self._find_log_key(actual_key)
        if is_empty(log_key):
            raise KeyNotFoundError(actual_key)
        return cast("str", log_key)

    # -- Bookkeeping -------------------------------------------------------

    def _on_child_created(self, address: str | int) -> None:
        """Log every new child's key into ``__keys__/``, exactly once."""
        self._append_log_key(address)

    def _rebuild_bookkeeping(self) -> None:
        """Rebuild ``__keys__/`` and ``__log__/`` from the children in ``__data__/``.

        Keeps the first log entry of every key that still holds data, drops
        stale and repeated entries, realigns the reverse index, then logs
        keys the index never recorded in storage scan order.
        """
        self._ensure_internal_layout()
        data_keys = list(self._data_container().iter_child_keys(validate=False))
        present = set(data_keys)
        kc = self._keys_container()
        lc = self._log_container()
        seen: dict[str | int, str] = {}
        for log_key, actual_key in list(self._scan_log_keys()):
            if actual_key in present and actual_key not in seen:
                seen[actual_key] = log_key
            else:
                kc.delete_child_primitive_unsafe(log_key)
        indexed: set[str | int] = set()
        for actual_key, info in list(lc.iter_children(validate=False)):
            want = seen.get(actual_key)
            if want is None:
                lc.delete_child_primitive_unsafe(actual_key)
                continue
            indexed.add(actual_key)
            if info.primitive_value != want:
                lc.put_child_primitive(actual_key, want, validate=False, validate_parent=False)
        for actual_key, log_key in seen.items():
            if actual_key not in indexed:
                lc.put_child_primitive(actual_key, log_key, validate=False, validate_parent=False)
        for key in data_keys:
            if key not in seen:
                self._append_log_key(key)  # type: ignore[arg-type]

    # -- Observability overrides -------------------------------------------
    # LogIndexedDictView stores data under `__data__/` and log keys under
    # `__keys__/`, not directly under `container.site`. The base observable
    # methods (ChildObservableBase, ObservableBase) build filters at
    # `(container.site, ...)`, so no notifs would ever match real writes.
    # These overrides rewrite the filter sites so callers can watch the
    # view the way they'd watch a plain dict.

    def on_change(self) -> SubscriptionOptions:
        """Any data-side write.

        Scoped to __data__/ so log-key appends (bookkeeping noise) don't
        fire the callback.
        """
        return SubscriptionOptions(PrefixFilter(prefix=(*self.container.site, _DATA)))

    def on_child_change(self, address: str | int) -> SubscriptionOptions:
        """Any write at or under __data__/<address>.

        Matches primitive set/replace on this child AND nested field
        writes for compound (shape) children.
        """
        normalized = self.normalize_address(address)
        child_data_site = (*self.container.site, _DATA, normalized)
        return SubscriptionOptions(PrefixFilter(prefix=child_data_site))

    def on_children_change(self) -> SubscriptionOptions:
        """Fires once per new key ever added.

        Watches __keys__/ appends, which happen exactly once when
        `__setitem__` sees `is_new`. New mints appear here; replacements
        of an existing key don't.
        """
        keys_site = (*self.container.site, _KEYS)
        return SubscriptionOptions(WildcardFilter(pattern=(*keys_site, "*")))

    def on_descendants_change(
        self,
        address: object,
        *addresses: object,
    ) -> SubscriptionOptions:
        """Wildcard match under __data__/.

        Callers pattern relative to the view's dict (e.g.
        ("*", "total_txs")); the __data__/ prefix is prepended so real
        writes match. Name matches nu.core.reactive's
        OnDescendantsChangeQuery (spelled with 'a').
        """
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
            limit: cap on yielded rows, counted after the cursor row is
                dropped. The storage scan is asked for one more row when a
                cursor is given, since the cursor row (if it still exists)
                is read and skipped here.

        Yields:
            (log_key, actual_key) tuples in chosen chronological direction.
        """
        kc = self._keys_container()
        site = kc.site

        prefix = PrefixFilter(prefix=site)
        child_len = LengthFilter(length=len(site) + 1)
        cursor = before if reverse else after
        storage_limit = None if limit is None else limit + (1 if cursor is not None else 0)

        if reverse and before is None:
            # Reverse scans need an upper bound within our prefix range;
            # encoded `site` sorts *below* every child, so it would yield
            # nothing on a reverse seek. The codec's `upper_bound_of_prefix`
            # gives a sentinel that sorts strictly above every real log key
            # under this site — exactly what a reverse scan needs.
            codec = kc.ctx._storage.codec  # type: ignore[attr-defined]
            opts = StorageScanOptions(
                start_encoded=codec.upper_bound_of_prefix(site),
                reverse=True,
                limit=storage_limit,
                break_filter=prefix,
                filter=prefix & child_len,
            )
        else:
            # With a cursor, seek to its row and skip it below; the storage
            # limit already allows for that row.
            start = site if cursor is None else (*site, cursor)
            opts = StorageScanOptions(
                start=start,
                reverse=reverse,
                limit=storage_limit,
                break_filter=prefix,
                filter=prefix & child_len,
            )

        from virtuals.container import scan_or_absent
        from virtuals.container.context import require_read_context

        rctx = require_read_context(kc.ctx)
        rows = scan_or_absent(rctx.scan(opts).items(), self.container.site, kc.ctx)
        yielded = 0
        for key, value in rows:
            log_key = key[-1]
            if log_key == cursor:
                continue
            yield log_key, cast("Value", value)
            yielded += 1
            if limit is not None and yielded >= limit:
                return

    # -- Membership & keys -------------------------------------------------

    def __len__(self) -> int:
        """Live count of ``__keys__/`` rows. O(n) over primitive rows, no stored length."""
        return sum(1 for _ in self._scan_log_keys())

    def __bool__(self) -> bool:
        """True if any key is logged. One seek, so ``if view:`` never counts."""
        return self.next_key_after(None) is not None

    def __contains__(self, obj: str | int) -> bool:
        """Check if key exists. A key the view cannot hold is not in it."""
        return is_view_key(obj) and self._data_container().exists_child(
            obj, strict=True, owner=self.container.site
        )

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
        after: str | int | None = None,
        limit: int | None = None,
    ) -> KeysView[str | int] | Generator[str | int, None, None]:
        """Keys in insertion order.

        With no arguments: a set-like KeysView. With ``after`` or ``limit``:
        a generator, a page of the keys logged after the item ``after``
        (from the oldest when None), at most ``limit`` of them.

        Args:
            after: An item id held by the view. KeyError if it holds no
                data. Cost: about three gets to find its log slot; an id
                whose reverse ``__log__/`` row is missing or stale (data
                logged before that index, or by a writer that skips it)
                falls back to a ``__keys__/`` scan, O(n).
            limit: Cap on the number of keys yielded.

        Cost: O(limit) rows from one seek into ``__keys__/``.
        """
        if after is None and limit is None:
            return ReversibleKeysView(self)
        return self._keys_page(cursor=after, limit=limit, reverse=False)

    def keys_reverse(
        self,
        *,
        before: str | int | None = None,
        limit: int | None = None,
    ) -> Generator[str | int, None, None]:
        """Keys newest first, logged before the item ``before``, at most ``limit``.

        From the newest when ``before`` is None. ``before`` is an item id,
        KeyError if it holds no data, resolved as in ``keys(after=)`` (the
        same O(n) fallback for a missing or stale ``__log__/`` row). Cost:
        O(limit) rows from one reverse seek, so a tail-N read stays cheap on
        any size of log.
        """
        return self._keys_page(cursor=before, limit=limit, reverse=True)

    def _keys_page(
        self, *, cursor: str | int | None, limit: int | None, reverse: bool
    ) -> Generator[str | int, None, None]:
        """One page of actual keys. The cursor id resolves to its log key eagerly."""
        if cursor is not None and not is_view_key(cursor):
            raise TypeError(f"view keys must be str or int, got {type(cursor).__name__}")
        log_key = None if cursor is None else self._require_log_key(cursor)
        return self._iter_page(log_key, limit=limit, reverse=reverse)

    def _iter_page(
        self, log_key: str | None, *, limit: int | None, reverse: bool
    ) -> Generator[str | int, None, None]:
        if reverse:
            rows = self._scan_log_keys(reverse=True, before=log_key, limit=limit)
        else:
            rows = self._scan_log_keys(after=log_key, limit=limit)
        for _log_key, actual_key in rows:
            yield actual_key

    def first(self) -> str | int:
        """Oldest key. One seek. IndexError when empty, as ``seq[0]``."""
        for _log_key, actual_key in self._scan_log_keys(limit=1):
            return actual_key
        raise IndexError("first() of an empty collection")

    def last(self) -> str | int:
        """Newest key. One reverse seek. IndexError when empty, as ``seq[-1]``."""
        for _log_key, actual_key in self._scan_log_keys(reverse=True, limit=1):
            return actual_key
        raise IndexError("last() of an empty collection")

    def keys_with_log_keys(
        self, *, after: str | None = None
    ) -> Generator[tuple[str, str | int], None, None]:
        """Yield (log_key, actual_key) pairs, past the log key ``after``.

        The log-key cursor: unlike an item id, a log key still marks a
        position after its item is deleted, so a stream resumes past it.
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

    # -- Mutations ---------------------------------------------------------

    def __setitem__(self, address: str | int, value: object) -> None:
        """Set value under ``__data__/``. A new key is logged via ``_on_child_created``."""
        self._set_child_value(address, value)

    def set_primitive(self, address: str | int, value: object) -> None:
        """Store a compound value as a single primitive blob.

        Bypasses decomposition - stores the value as-is regardless of whether
        it would normally be decomposed into sub-keys. Use for controlled
        granularity (e.g., storing a list or dict as a single key).
        """
        self._reject_empty(address, value)
        self.ensure_created()
        self._put_child_primitive(address, value)

    def __delitem__(self, address: str | int) -> None:
        # No ensure_created here: deletes must not materialize the view as
        # a side effect. If the underlying __data__/ container doesn't
        # exist yet, exists_child returns False against raw storage and we
        # raise KeyNotFoundError -- a KeyError, matching Python dict
        # semantics, and the one error ref callers no-op on.
        dc = self._data_container()
        if not is_view_key(address) or not dc.exists_child(address):
            raise KeyNotFoundError(address)
        dc.delete_child(address)
        # Drop the log entry: found in two gets (a scan only for data
        # logged before the reverse index), then two point deletes.
        log_key = self._find_log_key(address)
        if not is_empty(log_key):
            self._keys_container().delete_child_primitive_unsafe(cast("str", log_key))
            self._log_container().delete_child_primitive_unsafe(address)

    def clear(self) -> None:
        if not self.container.exists():
            return  # clear of nothing is not a write: the view stays absent
        self.ensure_created()
        self._data_container().clear_children(validate=False)
        self._keys_container().clear_children(validate=False)
        self._log_container().clear_children(validate=False)

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

        A value holding EMPTY anywhere raises ``ValueError`` before the view,
        the log rows or the child are written.
        """
        value = self._reject_empty_at(address, value)
        self.ensure_created()
        self._populate_child_container(address, value, view_class=view_class)

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

    def update(self, other: object = (), /, **kwargs: object) -> None:
        """Update as ``dict.update``. The whole batch is checked for EMPTY before the first write."""
        items = self._update_items(other, kwargs)
        self._reject_empty_in(items)
        for key, value in items.items():
            self[key] = value

    def store(self, value: PyMapping[str | int, object], *, replace: bool = True) -> None:
        value = self._reject_empty_in(value)  # type: ignore[assignment]
        self.ensure_created()
        if replace and self.next_key_after(None) is not None:
            self.clear()

        # Each new key is logged via _on_child_created, in insertion order
        for key, val in value.items():
            self._set_child_value(key, val)

    # -- Navigation --------------------------------------------------------

    def open_child(self, address: str | int, view: type) -> View:  # type: ignore[override]
        """Open child view under __data__/."""
        normalized = self.normalize_address(address)
        dc = self._data_container()
        child_site = (*dc.site, normalized)
        child_container = Container(ctx=dc.ctx, site=child_site)
        return view(child_container, self.registry, parent=self)

    # -- Internal read helpers ---------------------------------------------

    def _read_child_from_data(self, address: str | int) -> tuple[object, bool]:
        if not is_view_key(address):
            raise KeyNotFoundError(address)
        dc = self._data_container()
        child_site = (*dc.site, address)
        from virtuals.container import node_ops

        node_info = node_ops.get_node_info(child_site, dc.ctx)
        if not node_info.exists or node_info.node_type == NodeType.NOT_FOUND:
            raise KeyNotFoundError(address)
        return node_info, node_info.node_type == NodeType.CONTAINER

    def _extract_data_child(self, address: str | int, node_info: object) -> object:
        from virtuals.collections import Convertible

        dc = self._data_container()
        child_site = (*dc.site, address)
        child_container = Container(ctx=dc.ctx, site=child_site)
        if node_info.structure is None:  # type: ignore[union-attr]
            raise ValueError(f"Child container '{address}' has no structure ID")
        view_class = self.registry.get_view_for_structure(node_info.structure)  # type: ignore[union-attr]
        child_view = view_class(container=child_container, registry=self.registry, parent=self)
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
        return view_class(container=child_container, registry=self.registry, parent=self)


# =============================================================================
# EAGER FACET
# =============================================================================


class EagerLogIndexedDictView(LogIndexedDictViewBase):
    """Eager log-indexed dict view — reads return extracted Python values."""

    CONTAINER_CLS: ClassVar[type] = dict

    def __getitem__(self, address: str | int) -> object | Empty:
        node_info, is_container = self._read_child_from_data(address)
        if not is_container:
            if is_empty(node_info.primitive_value):  # type: ignore[union-attr]
                raise KeyNotFoundError(address)
            return node_info.primitive_value  # type: ignore[union-attr]
        return self._extract_data_child(address, node_info)

    def values(self) -> ValuesView[object]:
        """Get all values as a collection view.

        Iterates in `__keys__/`-log order (insertion order), reading each
        value from `__data__/` per key.
        """
        return _EagerLogIndexedValuesView(self)

    def items(self) -> ItemsView[str | int, object]:
        """Get all items as a set-like view.

        Iterates in `__keys__/`-log order (insertion order), reading each
        value from `__data__/` per key.
        """
        return _EagerLogIndexedItemsView(self)

    def _iter_values(self) -> Generator[object, None, None]:
        """Single-pass value iteration in log (insertion) order."""
        for _log_key, actual_key in self._scan_log_keys():
            node_info, is_container = self._read_child_from_data(actual_key)
            if not is_container:
                yield node_info.primitive_value  # type: ignore[union-attr]
            else:
                yield self._extract_data_child(actual_key, node_info)

    def _iter_items(self) -> Generator[tuple[str | int, object], None, None]:
        """Single-pass items iteration in log (insertion) order."""
        for _log_key, actual_key in self._scan_log_keys():
            node_info, is_container = self._read_child_from_data(actual_key)
            if not is_container:
                yield actual_key, node_info.primitive_value  # type: ignore[union-attr]
            else:
                yield actual_key, self._extract_data_child(actual_key, node_info)

    def _iter_values_reverse(self) -> Generator[object, None, None]:
        """Single-pass reverse value iteration (newest first)."""
        for _log_key, actual_key in self._scan_log_keys(reverse=True):
            node_info, is_container = self._read_child_from_data(actual_key)
            if not is_container:
                yield node_info.primitive_value  # type: ignore[union-attr]
            else:
                yield self._extract_data_child(actual_key, node_info)

    def _iter_items_reverse(self) -> Generator[tuple[str | int, object], None, None]:
        """Single-pass reverse items iteration (newest first)."""
        for _log_key, actual_key in self._scan_log_keys(reverse=True):
            node_info, is_container = self._read_child_from_data(actual_key)
            if not is_container:
                yield actual_key, node_info.primitive_value  # type: ignore[union-attr]
            else:
                yield actual_key, self._extract_data_child(actual_key, node_info)

    def extract(self) -> dict[str | int, object]:
        return {k: self[k] for k in self}

    @property
    def lazy(self) -> LazyLogIndexedDictView:
        return LazyLogIndexedDictView(
            container=self.container, registry=self.registry, parent=self.parent
        )

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
            if is_empty(node_info.primitive_value):  # type: ignore[union-attr]
                raise KeyNotFoundError(address)
            return node_info.primitive_value  # type: ignore[union-attr]
        return self._view_data_child(address, node_info)

    def values(self) -> ValuesView[object]:
        """Get all values as a collection view.

        Returns Views for containers, values for primitives, in log order.
        """
        return _LazyLogIndexedValuesView(self)

    def items(self) -> ItemsView[str | int, object]:
        """Get all items as a set-like view.

        Returns (key, View|value) pairs in log order.
        """
        return _LazyLogIndexedItemsView(self)

    def _iter_values(self) -> Generator[object, None, None]:
        """Single-pass value iteration in log (insertion) order."""
        for _log_key, actual_key in self._scan_log_keys():
            node_info, is_container = self._read_child_from_data(actual_key)
            if not is_container:
                yield node_info.primitive_value  # type: ignore[union-attr]
            else:
                yield self._view_data_child(actual_key, node_info)

    def _iter_items(self) -> Generator[tuple[str | int, object], None, None]:
        """Single-pass items iteration in log (insertion) order."""
        for _log_key, actual_key in self._scan_log_keys():
            node_info, is_container = self._read_child_from_data(actual_key)
            if not is_container:
                yield actual_key, node_info.primitive_value  # type: ignore[union-attr]
            else:
                yield actual_key, self._view_data_child(actual_key, node_info)

    def _iter_values_reverse(self) -> Generator[object, None, None]:
        """Single-pass reverse value iteration (newest first)."""
        for _log_key, actual_key in self._scan_log_keys(reverse=True):
            node_info, is_container = self._read_child_from_data(actual_key)
            if not is_container:
                yield node_info.primitive_value  # type: ignore[union-attr]
            else:
                yield self._view_data_child(actual_key, node_info)

    def _iter_items_reverse(self) -> Generator[tuple[str | int, object], None, None]:
        """Single-pass reverse items iteration (newest first)."""
        for _log_key, actual_key in self._scan_log_keys(reverse=True):
            node_info, is_container = self._read_child_from_data(actual_key)
            if not is_container:
                yield actual_key, node_info.primitive_value  # type: ignore[union-attr]
            else:
                yield actual_key, self._view_data_child(actual_key, node_info)

    def extract(self) -> dict[str | int, object]:
        """Extract all items as a native dict.

        Delegates to the eager facet so `_extract_child_container` on a parent
        view works when this class is registered as the log-indexed variant
        (e.g., via `virtuals.views.LogIndexedDictView`).
        """
        return self.eager.extract()

    @property
    def eager(self) -> EagerLogIndexedDictView:
        return EagerLogIndexedDictView(
            container=self.container, registry=self.registry, parent=self.parent
        )

    @property
    def lazy(self) -> LazyLogIndexedDictView:
        return self


# =============================================================================
# VIEW CLASSES — proper ValuesView / ItemsView with single-pass iteration
# =============================================================================


class _EagerLogIndexedValuesView(ValuesView):
    """Efficient ValuesView — single-pass iteration in log (insertion) order."""

    _mapping: EagerLogIndexedDictView

    def __iter__(self) -> Generator[object, None, None]:  # type: ignore[override]
        yield from self._mapping._iter_values()

    def __reversed__(self) -> Generator[object, None, None]:
        return _EagerLogIndexedReversedValuesView(self._mapping).__iter__()


class _EagerLogIndexedItemsView(ItemsView):
    """Efficient ItemsView — single-pass iteration in log (insertion) order."""

    _mapping: EagerLogIndexedDictView

    def __iter__(self) -> Generator[tuple[str | int, object], None, None]:  # type: ignore[override]
        yield from self._mapping._iter_items()

    def __reversed__(self) -> Generator[tuple[str | int, object], None, None]:
        return _EagerLogIndexedReversedItemsView(self._mapping).__iter__()


class _LazyLogIndexedValuesView(ValuesView):
    """Efficient ValuesView — single-pass iteration, lazy child access."""

    _mapping: LazyLogIndexedDictView

    def __iter__(self) -> Generator[object, None, None]:  # type: ignore[override]
        yield from self._mapping._iter_values()

    def __reversed__(self) -> Generator[object, None, None]:
        return _LazyLogIndexedReversedValuesView(self._mapping).__iter__()


class _LazyLogIndexedItemsView(ItemsView):
    """Efficient ItemsView — single-pass iteration, lazy child access."""

    _mapping: LazyLogIndexedDictView

    def __iter__(self) -> Generator[tuple[str | int, object], None, None]:  # type: ignore[override]
        yield from self._mapping._iter_items()

    def __reversed__(self) -> Generator[tuple[str | int, object], None, None]:
        return _LazyLogIndexedReversedItemsView(self._mapping).__iter__()


class _EagerLogIndexedReversedValuesView(ValuesView):
    """Reverse-order ValuesView — single-pass reverse iteration."""

    _mapping: EagerLogIndexedDictView

    def __iter__(self) -> Generator[object, None, None]:  # type: ignore[override]
        yield from self._mapping._iter_values_reverse()


class _EagerLogIndexedReversedItemsView(ItemsView):
    """Reverse-order ItemsView — single-pass reverse iteration."""

    _mapping: EagerLogIndexedDictView

    def __iter__(self) -> Generator[tuple[str | int, object], None, None]:  # type: ignore[override]
        yield from self._mapping._iter_items_reverse()


class _LazyLogIndexedReversedValuesView(ValuesView):
    """Reverse-order ValuesView — single-pass reverse iteration, lazy child access."""

    _mapping: LazyLogIndexedDictView

    def __iter__(self) -> Generator[object, None, None]:  # type: ignore[override]
        yield from self._mapping._iter_values_reverse()


class _LazyLogIndexedReversedItemsView(ItemsView):
    """Reverse-order ItemsView — single-pass reverse iteration, lazy child access."""

    _mapping: LazyLogIndexedDictView

    def __iter__(self) -> Generator[tuple[str | int, object], None, None]:  # type: ignore[override]
        yield from self._mapping._iter_items_reverse()
