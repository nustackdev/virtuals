"""FractionalIndexedDictView: an ordered collection keyed by id.

Layout under the view's container::

    site/
    ├── __data__/<id>       value (primitive or compound), as in LogIndexedDictView
    ├── __order__/<okey>    -> id      (range-scanned for order)
    └── __pos__/<id>        -> okey    (reverse index: O(1) delete and move)

``okey`` is a fractional index (see ``fractional_index``): a base62 string
whose byte order is the collection order, generated between its two
neighbours so an insert or a move never rewrites other rows. Every key
carries a random tail, so two writers inserting between the same
neighbours get distinct keys that both land in the gap.

Every operation touches a constant number of storage keys: a neighbour is
found with one bounded seek on ``__order__``, never a scan. There is no
stored head, tail or length: ``first``/``last`` are one seek each and
``len`` counts ``__pos__`` rows live.

Placement is relative (``before=``/``after=`` an id), never by index.

Key length. Inserting again and again at one gap grows the key about one
char per four inserts, and LMDB rejects storage keys above 511 bytes
(the encoded site plus the order key). So a fresh key longer than
``MAX_KEY_LENGTH`` (128) triggers a window respace: up to
``RESPACE_WINDOW`` (16) rows on each side of the gap get fresh, evenly
spread keys between the rows just outside the window. That is a bounded
rewrite (about 100 key touches at 16) and keeps the op constant-cost. If
the outer rows leave no room for short keys the window widens once (x4),
and as a last resort the whole collection is respaced (``respace()``,
O(n)); in practice that needs long keys crammed on both sides of the gap.
A respace never changes the relative order.

Observability: ``on_change`` covers ``__data__`` and ``__order__`` (a
reorder is a change); ``on_children_change`` watches ``__order__/*`` and
fires on add, delete and move. A respacing insert also fires both for
every rewritten row (one delete and one put per row) as if those ids had
moved, though their relative order is unchanged.

Partial states. An insert writes ``__pos__``, then the value, then
``__order__``; if the value write raises, the position and whatever part
of the value was written are deleted again before the error propagates,
so the id is absent (not ``in``, not counted, not iterated, not readable)
and a retry inserts it as if new. A position without an order row can
still appear through bypass writes; such an id is ``in`` the view and
counted by ``len`` but not iterated, and a plain set does not repair it
(the id has data and a position, so set keeps it as is): delete it and
insert again, or call ``rebuild_bookkeeping()``. ``items``, ``values``
and ``extract`` skip order rows whose data is missing; ``iter`` and
``keys`` do not check (one scan, no gets). Setting an id that has data
but no position appends it, through ``__setitem__``, ``set_primitive``
and ``set_child_container_as`` alike. ``rebuild_bookkeeping()`` repairs
everything in O(n).

Concurrency. Two writers inserting at the same gap get distinct keys
(random tail) and both land in the gap. That holds as long as nobody
respaces: a window respace (only when a fresh key passes
``MAX_KEY_LENGTH``) deletes and rewrites the order rows of up to
``RESPACE_WINDOW`` neighbours on each side of its gap, and a concurrent
insert or move whose neighbour lies inside that window is placed
relative to a key the respace removes. Both commit (the plain insert
touches none of the rewritten rows, so no backend sees a conflict); the
item stays present exactly once, but its position can shift into the
rewritten range, next to the respacing gap instead of next to its
neighbour. This is accepted: it needs a key at the cap and a concurrent
write inside the window, and the alternative (writing the neighbour's
order row on every insert to force a conflict) would serialize every
same-gap insert. The mem backend has no write conflicts, so two
transactions inserting the same id there both commit and the second
``__pos__`` write wins (last-writer-wins, two order rows for one id
until ``rebuild_bookkeeping()``); sqlite and rocksdb reject the second
writer.

Order key collision. Two writers that read the same neighbours get the
same plain key and differ only in the random tail (``SUFFIX_LENGTH``
base62 chars, one collision in about 14.8 million same-gap pairs at 4).
Before writing, ``_allot`` reads the order row of its fresh key and
draws again if the row exists, so a collision with any row this
transaction can see is impossible. A collision with a row committed by
a concurrent transaction after this one's snapshot is not caught on
backends that do not track reads (rocksdb, mem): the later ``__order__``
put overwrites the earlier one, and the losing id keeps its ``__pos__``
row (``in`` and counted) but is not iterated until
``rebuild_bookkeeping()``. sqlite and lmdb serialise writers, so their
second writer sees the first row and never collides.
"""

from __future__ import annotations

import contextlib
from collections.abc import ItemsView, KeysView, MutableMapping, ValuesView
from typing import TYPE_CHECKING, ClassVar, cast

from virtuals.container import (
    Container,
    ContainerProtocol,
    ContainerStructure,
    NodeType,
    scan_or_absent,
)
from virtuals.container.context import require_read_context
from virtuals.tkv.filter import LengthFilter, PrefixFilter, WildcardFilter
from virtuals.tkv.observer import SubscriptionOptions
from virtuals.tkv.storage import StorageScanOptions
from virtuals.types import EMPTY, Empty, is_empty
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
    is_view_key,
)

from .fractional_index import (
    DEFAULT_SUFFIX_LENGTH,
    generate_n_suffixed_keys_between,
    generate_suffixed_key_between,
)


if TYPE_CHECKING:
    from collections.abc import Generator, Iterable
    from collections.abc import Mapping as PyMapping

    from virtuals.view import View


__all__ = [
    "EagerFractionalIndexedDictView",
    "FractionalIndexedDictViewBase",
    "LazyFractionalIndexedDictView",
]

_DATA = "__data__"
_ORDER = "__order__"
_POS = "__pos__"

Address = str | int


# =============================================================================
# BASE: shared by eager and lazy facets
# =============================================================================


class FractionalIndexedDictViewBase(
    ObservableBase,
    ChildObservableBase[Address],
    DescendantsObservableBase,
    ChildNavigationBase[Address],
    ChildNestedGetBase,
    ChildNestedSetBase,
    ChildPrimitiveSetBase,
    LazyChildReadBase,
    PrimitiveOpsBase,
    UnsafePrimitiveOpsBase,
    ViewBase,
):
    """Ordered dict view with fractional index order keys.

    Mapping semantics on ``__data__/`` (``view[id]``, ``view[id] = v``,
    ``del view[id]``), order kept in ``__order__/`` and ``__pos__/``.
    Setting a new id appends it; setting an existing id keeps its place.
    """

    STRUCTURE: ClassVar[ContainerStructure] = ContainerStructure(17)
    PROTOCOL: ClassVar[ContainerProtocol] = ContainerProtocol.MAPPING | ContainerProtocol.MUTABLE
    SUFFIX_LENGTH: ClassVar[int] = DEFAULT_SUFFIX_LENGTH
    MAX_KEY_LENGTH: ClassVar[int] = 128
    """A fresh order key longer than this triggers a window respace."""
    RESPACE_WINDOW: ClassVar[int] = 16
    """Rows rewritten on each side of the gap by a window respace."""
    RESPACE_TARGET: ClassVar[int] = 64
    """Longest key a window respace may produce before it widens or falls back."""

    @classmethod
    def is_address_static(cls, address: object) -> bool:
        return False

    def normalize_address(self, address: Address) -> Address:
        return address

    # -- Internal containers -----------------------------------------------

    def _data_container(self) -> Container:
        return Container(ctx=self.container.ctx, site=(*self.container.site, _DATA))

    def _children_container(self) -> Container:
        """Children live under ``__data__/``."""
        return self._data_container()

    def _order_container(self) -> Container:
        return Container(ctx=self.container.ctx, site=(*self.container.site, _ORDER))

    def _pos_container(self) -> Container:
        return Container(ctx=self.container.ctx, site=(*self.container.site, _POS))

    def _ensure_internal_layout(self) -> None:
        """Materialize the three sub-containers. Idempotent."""
        for name in (_DATA, _ORDER, _POS):
            self.container.create_child_container(
                name,
                structure=ContainerStructure(1),
                protocol=ContainerProtocol.MAPPING | ContainerProtocol.MUTABLE,
                validate=False,
            )

    # -- Order index primitives --------------------------------------------

    def _position_or_empty(self, address: Address) -> str | Empty:
        """Order key of ``address``, EMPTY if it has none. One get.

        A key the view cannot hold has no position, so ``in`` and every
        position read treat it as absent.
        """
        if not is_view_key(address):
            return EMPTY
        return cast("str | Empty", self._pos_container().get_child_primitive_unsafe(address))

    def _require_position(self, address: Address) -> str:
        okey = self._position_or_empty(address)
        if is_empty(okey):
            raise KeyError(address)
        return cast("str", okey)

    def _scan_order(
        self,
        *,
        start: str | None = None,
        reverse: bool = False,
        limit: int | None = None,
        skip: tuple[str, ...] = (),
        strict: bool = True,
    ) -> Generator[tuple[str, Address], None, None]:
        """Scan ``__order__/`` rows as ``(okey, id)``.

        Args:
            start: order key to start from (inclusive at the storage level).
                None means the first row (or the last one when ``reverse``).
            reverse: scan descending.
            limit: cap on yielded rows, counted after ``skip`` is applied.
            skip: order keys to drop (the cursor row, a moving row).
            strict: reads: an empty scan on a never-written view raises
                ``ContainerNotFoundError``; any row found proves the view
                exists. Writes, which created the view already, pass False.
        """
        oc = self._order_container()
        site = oc.site
        prefix = PrefixFilter(prefix=site)
        child_len = LengthFilter(length=len(site) + 1)
        storage_limit = None if limit is None else limit + len(skip)
        if reverse and start is None:
            # The encoded site sorts below every child; a reverse seek needs
            # a sentinel strictly above all of them.
            codec = oc.ctx.storage.codec  # type: ignore[attr-defined]
            opts = StorageScanOptions(
                start_encoded=codec.upper_bound_of_prefix(site),
                reverse=True,
                limit=storage_limit,
                break_filter=prefix,
                filter=prefix & child_len,
            )
        else:
            opts = StorageScanOptions(
                start=site if start is None else (*site, start),
                reverse=reverse,
                limit=storage_limit,
                break_filter=prefix,
                filter=prefix & child_len,
            )
        rctx = require_read_context(oc.ctx)
        rows = rctx.scan(opts).items()
        if strict:
            rows = scan_or_absent(rows, self.container.site, oc.ctx)
        yielded = 0
        for key, value in rows:
            okey = key[-1]
            if okey in skip:
                continue
            yield okey, cast("Address", value)
            yielded += 1
            if limit is not None and yielded >= limit:
                return

    def _edge(self, *, last: bool, strict: bool = True) -> tuple[str, Address] | None:
        """First or last ``(okey, id)``. One seek."""
        for row in self._scan_order(reverse=last, limit=1, strict=strict):
            return row
        return None

    def _neighbour(
        self, okey: str, *, before: bool, skip: tuple[str, ...] = ()
    ) -> tuple[str, Address] | None:
        """Row directly before or after ``okey``. One seek."""
        rows = self._scan_order(
            start=okey, reverse=before, limit=1, skip=(okey, *skip), strict=False
        )
        for row in rows:
            return row
        return None

    ALLOT_ATTEMPTS: ClassVar[int] = 8
    """Fresh keys tried before an allot gives up on a gap whose keys are all taken."""

    def _allot(self, prev: str | None, nxt: str | None) -> str:
        """Fresh order key for the gap between ``prev`` and ``nxt``, not yet in ``__order__``.

        One ``__order__`` get checks the key is free (a row this transaction
        can see, written by another id with the same random tail, would be
        overwritten by ``_index``); on a hit a new tail is drawn. If the
        key passes ``MAX_KEY_LENGTH`` the rows around the gap are respaced
        first (``_respace_around``) and the key comes from the respaced
        window.
        """
        oc = self._order_container()
        for _ in range(self.ALLOT_ATTEMPTS):
            okey = generate_suffixed_key_between(prev, nxt, suffix_length=self.SUFFIX_LENGTH)
            if len(okey) > self.MAX_KEY_LENGTH:
                return self._respace_around(prev, nxt)
            if is_empty(oc.get_child_primitive_unsafe(okey)):
                return okey
        raise RuntimeError(
            f"No free order key between {prev!r} and {nxt!r} after {self.ALLOT_ATTEMPTS} draws"
        )

    def _respace_around(self, prev: str | None, nxt: str | None) -> str:
        """Rewrite a bounded window of rows around the gap, return the gap's key.

        Window: ``prev`` and up to ``RESPACE_WINDOW - 1`` rows below it, and
        ``nxt`` and up to ``RESPACE_WINDOW - 1`` rows above it (two seeks).
        The rows just outside the window bound the fresh keys. If the fresh
        keys would still pass ``RESPACE_TARGET`` the window widens four
        times, then the whole collection is respaced. Relative order never
        changes; every rewritten row is one order delete, one order put and
        one pos put.

        A concurrent transaction that inserts or moves next to a row inside
        the window computed its key against an order key this rewrite
        removes. Nothing conflicts (it writes none of the rewritten rows),
        so both commit and its item ends up inside the rewritten range,
        next to this gap, rather than next to its neighbour. Documented in
        the module docstring; accepted rather than forced into a conflict.
        """
        window = self.RESPACE_WINDOW
        for k in (window, 4 * window, None):
            limit = None if k is None else k + 1
            below = (
                []
                if prev is None
                else list(self._scan_order(start=prev, reverse=True, limit=limit, strict=False))
            )
            above = (
                [] if nxt is None else list(self._scan_order(start=nxt, limit=limit, strict=False))
            )
            lo = hi = None
            if k is not None and len(below) > k:
                lo, below = below[k][0], below[:k]
            if k is not None and len(above) > k:
                hi, above = above[k][0], above[:k]
            below.reverse()
            keys = generate_n_suffixed_keys_between(
                lo, hi, len(below) + 1 + len(above), suffix_length=self.SUFFIX_LENGTH
            )
            if k is None or max(len(key) for key in keys) <= self.RESPACE_TARGET:
                break
        oc = self._order_container()
        for okey, _address in (*below, *above):
            oc.delete_child_primitive_unsafe(okey)
        fresh = iter(keys)
        for _okey, address in below:
            self._index(address, next(fresh))
        gap_key = next(fresh)
        for _okey, address in above:
            self._index(address, next(fresh))
        return gap_key

    def _index(self, address: Address, okey: str) -> None:
        """Record ``address`` at ``okey``: one ``__pos__`` put, one ``__order__`` put."""
        self._pos_container().put_child_primitive(
            address, okey, validate=False, validate_parent=False
        )
        self._order_container().put_child_primitive(
            okey, address, validate=False, validate_parent=False
        )

    def _place(self, address: Address, prev: str | None, nxt: str | None) -> str:
        """Give ``address`` (whose data exists) a fresh key between ``prev`` and ``nxt``."""
        okey = self._allot(prev, nxt)
        self._index(address, okey)
        return okey

    def _relocate(self, address: Address, okey: str, prev: str | None, nxt: str | None) -> str:
        """Move ``address`` from ``okey`` to a fresh key between ``prev`` and ``nxt``."""
        self._order_container().delete_child_primitive_unsafe(okey)
        return self._place(address, prev, nxt)

    def _write_value(self, address: Address, value: object) -> None:
        """Write ``__data__/<address>``; the caller has ensured this view exists."""
        if self.registry.is_container_type(value):
            self._populate_child_container(address, value)
        else:
            self._put_child_primitive(address, value)

    # -- Bookkeeping -------------------------------------------------------

    def _on_child_created(self, address: Address) -> None:
        """A new child with no position yet is appended.

        Inserts write ``__pos__`` before the data, so the hook sees the
        position and leaves it alone.
        """
        if not is_empty(self._position_or_empty(address)):
            return
        last = self._edge(last=True, strict=False)
        self._place(address, None if last is None else last[0], None)

    def _rebuild_bookkeeping(self) -> None:
        """Rebuild ``__order__/`` and ``__pos__/`` from ``__data__/``.

        Keeps the first order row of every id that still holds data, drops
        stale and repeated rows, realigns ``__pos__``, then appends ids
        the index never recorded in storage scan order.
        """
        self._ensure_internal_layout()
        oc = self._order_container()
        pc = self._pos_container()
        data_keys = list(self._data_container().iter_child_keys(validate=False))
        present = set(data_keys)
        seen: dict[Address, str] = {}
        for okey, address in list(self._scan_order(strict=False)):
            if address in present and address not in seen:
                seen[address] = okey
            else:
                oc.delete_child_primitive_unsafe(okey)
        for address, info in list(pc.iter_children(validate=False)):
            want = seen.get(address)
            if want is None:
                pc.delete_child_primitive_unsafe(address)
            elif info.primitive_value != want:
                pc.put_child_primitive(address, want, validate=False, validate_parent=False)
        for address in data_keys:
            if address not in seen:
                last = self._edge(last=True, strict=False)
                self._place(address, None if last is None else last[0], None)

    def respace(self) -> None:
        """Rewrite every order key short and evenly spaced. O(n), order kept."""
        rows = list(self._scan_order(strict=False))
        if not rows:
            return
        oc = self._order_container()
        for okey, _address in rows:
            oc.delete_child_primitive_unsafe(okey)
        keys = generate_n_suffixed_keys_between(
            None, None, len(rows), suffix_length=self.SUFFIX_LENGTH
        )
        for (_okey, address), okey in zip(rows, keys, strict=True):
            self._index(address, okey)

    # -- Observability overrides -------------------------------------------
    # Data lives under `__data__/` and order under `__order__/`, so the
    # base filters at `container.site` would never match a real write.

    def on_change(self) -> SubscriptionOptions:
        """Any data write or any reorder (``__data__`` or ``__order__``)."""
        site = self.container.site
        return SubscriptionOptions(
            PrefixFilter(prefix=(*site, _DATA)) | PrefixFilter(prefix=(*site, _ORDER))
        )

    def on_child_change(self, address: Address) -> SubscriptionOptions:
        """Any write at or under ``__data__/<address>``."""
        normalized = self.normalize_address(address)
        return SubscriptionOptions(PrefixFilter(prefix=(*self.container.site, _DATA, normalized)))

    def on_children_change(self) -> SubscriptionOptions:
        """Membership or order changed: an id was added, removed or moved."""
        return SubscriptionOptions(WildcardFilter(pattern=(*self.container.site, _ORDER, "*")))

    def on_descendants_change(self, address: object, *addresses: object) -> SubscriptionOptions:
        """Wildcard match under ``__data__/``, patterned relative to the dict."""
        pattern = (address, *addresses)
        return SubscriptionOptions(WildcardFilter(pattern=(*self.container.site, _DATA, *pattern)))

    # -- Membership & order ------------------------------------------------

    def __len__(self) -> int:
        """Live count of ``__pos__`` rows. O(n) over primitive rows, no stored length."""
        return self._pos_container().count_children(strict=True, owner=self.container.site)

    def __bool__(self) -> bool:
        """True if any id is ordered. One seek, so ``if view:`` never counts."""
        return self._edge(last=False) is not None

    def __contains__(self, obj: Address) -> bool:
        if not is_empty(self._position_or_empty(obj)):
            return True
        self.container.require_exists()  # a miss: is the view itself there?
        return False

    def __iter__(self) -> Generator[Address, None, None]:
        for _okey, address in self._scan_order():
            yield address

    def __reversed__(self) -> Generator[Address, None, None]:
        for _okey, address in self._scan_order(reverse=True):
            yield address

    def keys(
        self, *, after: Address | None = None, limit: int | None = None
    ) -> KeysView[Address] | Generator[Address, None, None]:
        """Ids in order. ``after`` starts just past that id, ``limit`` caps the count."""
        if after is None and limit is None:
            return KeysView(self)  # type: ignore[arg-type]
        return self._keys_scan(after=after, limit=limit, reverse=False)

    def keys_reverse(
        self, *, before: Address | None = None, limit: int | None = None
    ) -> Generator[Address, None, None]:
        """Ids in reverse order. ``before`` starts just below that id."""
        return self._keys_scan(after=before, limit=limit, reverse=True)

    def _keys_scan(
        self, *, after: Address | None, limit: int | None, reverse: bool
    ) -> Generator[Address, None, None]:
        if after is None:
            rows = self._scan_order(reverse=reverse, limit=limit)
        else:
            cursor = self._require_position(after)
            rows = self._scan_order(start=cursor, reverse=reverse, limit=limit, skip=(cursor,))
        for _okey, address in rows:
            yield address

    def first(self) -> Address | None:
        """First id, or None when empty. One seek."""
        row = self._edge(last=False)
        return None if row is None else row[1]

    def last(self) -> Address | None:
        """Last id, or None when empty. One seek."""
        row = self._edge(last=True)
        return None if row is None else row[1]

    def next_after(self, address: Address) -> Address | None:
        """Id directly after ``address``, or None at the end."""
        row = self._neighbour(self._require_position(address), before=False)
        return None if row is None else row[1]

    def prev_before(self, address: Address) -> Address | None:
        """Id directly before ``address``, or None at the start."""
        row = self._neighbour(self._require_position(address), before=True)
        return None if row is None else row[1]

    def position(self, address: Address) -> str:
        """Order key of ``address``. Opaque, stable until the id moves."""
        return self._require_position(address)

    def positions(self, *, reverse: bool = False) -> Generator[tuple[str, Address], None, None]:
        """``(order key, id)`` rows in order."""
        return self._scan_order(reverse=reverse)

    def get(self, address: Address, default: object | Empty = EMPTY) -> object | Empty:
        """Value at ``address``, or ``default`` (EMPTY) when the id is absent."""
        try:
            return self[address]  # type: ignore[index]
        except KeyError:
            return default

    # -- Mutations ---------------------------------------------------------

    def _ensure_positioned(self, address: Address) -> None:
        """Append ``address`` if its data exists without a position. One pos get.

        A plain set of an id whose position is missing (a bypass write, or
        a delete that stopped half way) must land it in the order again;
        ``_on_child_created`` only fires for a new data child.
        """
        if is_empty(self._position_or_empty(address)):
            last = self._edge(last=True, strict=False)
            self._place(address, None if last is None else last[0], None)

    def __setitem__(self, address: Address, value: object) -> None:
        """Set the value. A new id is appended, an existing id keeps its place.

        An id that has data but no position (a bypass write, or a delete
        that stopped half way) is appended too: one extra pos get.
        """
        self._set_child_value(address, value)
        self._ensure_positioned(address)

    def set_primitive(self, address: Address, value: object) -> None:
        """Store a compound value as one primitive blob (no decomposition).

        Same placement rule as ``__setitem__``: a new id is appended, an
        existing id keeps its place, an id with data but no position is
        appended.
        """
        self._reject_empty(address, value)
        self.ensure_created()
        self._put_child_primitive(address, value)
        self._ensure_positioned(address)

    def set_child_container_as(self, address: Address, value: object, view_class: type) -> None:
        """Peer of ``__setitem__`` with an explicit child view class.

        Same placement rule as ``__setitem__``. A value holding EMPTY
        anywhere raises ``ValueError`` before the view, the position rows or
        the child are written.
        """
        value = self._reject_empty_at(address, value)
        self.ensure_created()
        self._populate_child_container(address, value, view_class=view_class)
        self._ensure_positioned(address)

    def _insert(self, address: Address, value: object, prev: str | None, nxt: str | None) -> None:
        """Position, then value, then order row; a failed value write undoes the position.

        A position whose order row is missing (a bypass write, or a delete
        that stopped half way) is stale and gets overwritten rather than
        refused.
        """
        stale = self._position_or_empty(address)
        if not is_empty(stale) and not is_empty(
            self._order_container().get_child_primitive_unsafe(cast("str", stale))
        ):
            raise ValueError(f"{address!r} is already in the collection")
        okey = self._allot(prev, nxt)
        pc = self._pos_container()
        pc.put_child_primitive(address, okey, validate=False, validate_parent=False)
        try:
            self._write_value(address, value)
        except BaseException:
            # Leave nothing behind: the id must not be `in`, counted, or
            # readable with a partial value. The cleanup must not mask the
            # original error if storage itself is failing.
            with contextlib.suppress(Exception):
                pc.delete_child_primitive_unsafe(address)
            with contextlib.suppress(Exception):
                self._data_container().delete_child(address)
            raise
        self._order_container().put_child_primitive(
            okey, address, validate=False, validate_parent=False
        )

    def append(self, address: Address, value: object = True) -> None:
        """Add ``address`` at the end. Raises ValueError if it is present."""
        self._reject_empty(address, value)
        self.ensure_created()
        last = self._edge(last=True, strict=False)
        self._insert(address, value, None if last is None else last[0], None)

    def prepend(self, address: Address, value: object = True) -> None:
        """Add ``address`` at the start. Raises ValueError if it is present."""
        self._reject_empty(address, value)
        self.ensure_created()
        first = self._edge(last=False, strict=False)
        self._insert(address, value, None, None if first is None else first[0])

    def insert_before(self, ref: Address, address: Address, value: object = True) -> None:
        """Add ``address`` directly before ``ref``. KeyError if ``ref`` is absent."""
        ref_okey = self._require_position(ref)
        self._reject_empty(address, value)
        self.ensure_created()
        prev = self._neighbour(ref_okey, before=True)
        self._insert(address, value, None if prev is None else prev[0], ref_okey)

    def insert_after(self, ref: Address, address: Address, value: object = True) -> None:
        """Add ``address`` directly after ``ref``. KeyError if ``ref`` is absent."""
        ref_okey = self._require_position(ref)
        self._reject_empty(address, value)
        self.ensure_created()
        nxt = self._neighbour(ref_okey, before=False)
        self._insert(address, value, ref_okey, None if nxt is None else nxt[0])

    def move_before(self, address: Address, ref: Address) -> None:
        """Move ``address`` directly before ``ref``. Value untouched."""
        okey = self._require_position(address)  # raises on an absent id, self move too
        if address == ref:
            return
        ref_okey = self._require_position(ref)
        prev = self._neighbour(ref_okey, before=True)
        if prev is not None and prev[0] == okey:
            return
        self._relocate(address, okey, None if prev is None else prev[0], ref_okey)

    def move_after(self, address: Address, ref: Address) -> None:
        """Move ``address`` directly after ``ref``. Value untouched."""
        okey = self._require_position(address)  # raises on an absent id, self move too
        if address == ref:
            return
        ref_okey = self._require_position(ref)
        nxt = self._neighbour(ref_okey, before=False)
        if nxt is not None and nxt[0] == okey:
            return
        self._relocate(address, okey, ref_okey, None if nxt is None else nxt[0])

    def move_first(self, address: Address) -> None:
        """Move ``address`` to the start."""
        okey = self._require_position(address)
        first = self._edge(last=False, strict=False)
        if first is None or first[0] == okey:
            return
        self._relocate(address, okey, None, first[0])

    def move_last(self, address: Address) -> None:
        """Move ``address`` to the end."""
        okey = self._require_position(address)
        last = self._edge(last=True, strict=False)
        if last is None or last[0] == okey:
            return
        self._relocate(address, okey, last[0], None)

    def __delitem__(self, address: Address) -> None:
        # No ensure_created: a delete must not materialize the view.
        if not is_view_key(address):
            raise KeyError(address)
        dc = self._data_container()
        okey = self._position_or_empty(address)
        if is_empty(okey):
            if not dc.exists_child(address):
                raise KeyError(address)
        else:
            self._order_container().delete_child_primitive_unsafe(cast("str", okey))
            self._pos_container().delete_child_primitive_unsafe(address)
        dc.delete_child(address)

    def pop(self, address: Address, default: object | Empty = EMPTY) -> object | Empty:
        try:
            value = self[address]  # type: ignore[index]
        except KeyError:
            if is_empty(default):
                raise
            return default
        del self[address]
        return value

    def clear(self) -> None:
        if not self.container.exists():
            return  # clear of nothing is not a write: the view stays absent
        self.ensure_created()
        self._data_container().clear_children(validate=False)
        self._order_container().clear_children(validate=False)
        self._pos_container().clear_children(validate=False)

    def update(self, other: PyMapping[Address, object] | None = None, **kwargs: object) -> None:
        self._reject_empty_in(other or {})
        self._reject_empty_in(kwargs)
        if other:
            for key, value in other.items():
                self[key] = value
        for key, value in kwargs.items():
            self[key] = value

    def store(
        self, value: PyMapping[Address, object] | Iterable[Address], *, replace: bool = True
    ) -> None:
        """Fill from a mapping (id -> value) or an iterable of ids (value True), in order."""
        value = self._reject_empty_in(value)  # type: ignore[assignment]
        self.ensure_created()
        if replace and self._edge(last=False, strict=False) is not None:
            self.clear()
        items = value.items() if hasattr(value, "items") else ((k, True) for k in value)
        for key, val in items:
            self[key] = val

    # -- Navigation --------------------------------------------------------

    def open_child(self, address: Address, view: type) -> View:  # type: ignore[override]
        """Open a child view under ``__data__/``."""
        dc = self._data_container()
        child_container = Container(ctx=dc.ctx, site=(*dc.site, self.normalize_address(address)))
        return view(child_container, self.registry, parent=self)

    # -- Reads -------------------------------------------------------------

    def _read_child(self, address: Address) -> tuple[object, bool]:
        from virtuals.container import node_ops

        if not is_view_key(address):
            raise KeyError(address)
        dc = self._data_container()
        node_info = node_ops.get_node_info((*dc.site, address), dc.ctx)
        if not node_info.exists or node_info.node_type == NodeType.NOT_FOUND:
            raise KeyError(address)
        return node_info, node_info.node_type == NodeType.CONTAINER

    def _child_view(self, address: Address, node_info: object) -> View:
        dc = self._data_container()
        if node_info.structure is None:  # type: ignore[union-attr]
            raise ValueError(f"Child container '{address}' has no structure ID")
        view_class = self.registry.get_view_for_structure(node_info.structure)  # type: ignore[union-attr]
        child_container = Container(ctx=dc.ctx, site=(*dc.site, address))
        return view_class(container=child_container, registry=self.registry, parent=self)

    def _child_result(self, address: Address, node_info: object) -> object:
        """Facet hook: what a container child reads as (value or view)."""
        raise NotImplementedError

    def __getitem__(self, address: Address) -> object:
        node_info, is_container = self._read_child(address)
        if not is_container:
            if is_empty(node_info.primitive_value):  # type: ignore[attr-defined]
                raise KeyError(address)
            return node_info.primitive_value  # type: ignore[attr-defined]
        return self._child_result(address, node_info)

    def _iter_items(
        self, *, reverse: bool = False
    ) -> Generator[tuple[Address, object], None, None]:
        """Single-pass ``(id, value)`` iteration in collection order.

        Order rows whose data is missing (a bypass delete) are skipped.
        """
        for _okey, address in self._scan_order(reverse=reverse):
            try:
                node_info, is_container = self._read_child(address)
            except KeyError:
                continue
            if not is_container:
                yield address, node_info.primitive_value  # type: ignore[attr-defined]
            else:
                yield address, self._child_result(address, node_info)

    def _iter_values(self, *, reverse: bool = False) -> Generator[object, None, None]:
        for _address, value in self._iter_items(reverse=reverse):
            yield value

    def values(self) -> ValuesView[object]:
        """Values in collection order (single pass)."""
        return _FractionalValuesView(self)

    def items(self) -> ItemsView[Address, object]:
        """``(id, value)`` pairs in collection order (single pass)."""
        return _FractionalItemsView(self)

    def extract(self) -> dict[Address, object]:
        return dict(self.eager._iter_items())  # type: ignore[attr-defined]


# =============================================================================
# FACETS
# =============================================================================


class EagerFractionalIndexedDictView(FractionalIndexedDictViewBase):
    """Eager facet: container children read as extracted Python values."""

    CONTAINER_CLS: ClassVar[type] = dict

    def _child_result(self, address: Address, node_info: object) -> object:
        from virtuals.collections import Convertible

        child_view = self._child_view(address, node_info)
        if not isinstance(child_view, Convertible):
            raise TypeError(f"Child view {type(child_view).__name__} does not support extraction")
        return child_view.extract()

    @property
    def lazy(self) -> LazyFractionalIndexedDictView:
        return LazyFractionalIndexedDictView(
            container=self.container, registry=self.registry, parent=self.parent
        )

    @property
    def eager(self) -> EagerFractionalIndexedDictView:
        return self


class LazyFractionalIndexedDictView(FractionalIndexedDictViewBase):
    """Lazy facet: container children read as child Views."""

    def _child_result(self, address: Address, node_info: object) -> object:
        return self._child_view(address, node_info)

    @property
    def eager(self) -> EagerFractionalIndexedDictView:
        return EagerFractionalIndexedDictView(
            container=self.container, registry=self.registry, parent=self.parent
        )

    @property
    def lazy(self) -> LazyFractionalIndexedDictView:
        return self


# =============================================================================
# VALUES / ITEMS VIEWS: single-pass, both directions
# =============================================================================


class _FractionalValuesView(ValuesView):
    _mapping: FractionalIndexedDictViewBase

    def __init__(self, mapping: FractionalIndexedDictViewBase, *, reverse: bool = False) -> None:
        super().__init__(mapping)
        self._reverse = reverse

    def __iter__(self) -> Generator[object, None, None]:  # type: ignore[override]
        yield from self._mapping._iter_values(reverse=self._reverse)

    def __reversed__(self) -> Generator[object, None, None]:
        yield from self._mapping._iter_values(reverse=not self._reverse)


class _FractionalItemsView(ItemsView):
    _mapping: FractionalIndexedDictViewBase

    def __init__(self, mapping: FractionalIndexedDictViewBase, *, reverse: bool = False) -> None:
        super().__init__(mapping)
        self._reverse = reverse

    def __iter__(self) -> Generator[tuple[Address, object], None, None]:  # type: ignore[override]
        yield from self._mapping._iter_items(reverse=self._reverse)

    def __reversed__(self) -> Generator[tuple[Address, object], None, None]:
        yield from self._mapping._iter_items(reverse=not self._reverse)


MutableMapping.register(FractionalIndexedDictViewBase)
