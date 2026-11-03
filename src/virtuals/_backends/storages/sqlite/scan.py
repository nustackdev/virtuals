"""Scan iterator for SQLite storage."""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING, cast

from virtuals.tkv.filter import PrefixFilter
from virtuals.tkv.storage import (
    ScanProtocol,
    StorageOperationError,
    StorageScanOptions,
)


if TYPE_CHECKING:
    from collections.abc import Generator

    from virtuals.tkv.types import Key, Value

    from .context import ContextBase
    from .storage import SQLiteStorage


__all__ = ["SQLiteScan"]


# Rows fetched per page. Starts small so a short prefix scan reads little
# past its range, doubles up to the cap so a long scan pays few re-seeks.
_FIRST_PAGE = 32
_MAX_PAGE = 1024
# Page after a `max_depth` skip: the next child is normally the first row.
_SEEK_PAGE = 4


class _IteratorType(Enum):
    """Type of scan iteration."""

    KEYS = auto()
    VALUES = auto()
    ITEMS = auto()


class SQLiteScan(ScanProtocol):
    """Scan iterator for SQLite storage.

    Keys live in a ``WITHOUT ROWID`` table under a BLOB primary key, so
    SQLite orders them by memcmp, the same byte order LMDB and RocksDB use.

    Iteration is lazy and paged by key: each page is its own
    ``SELECT ... WHERE k > last ORDER BY k LIMIT n`` rather than one open
    statement held across yields. That keeps no statement live between
    pages, so the caller may write through the same transaction while
    iterating (clearing a subtree, say) without SQLite's undefined
    behaviour for a table modified under a running SELECT. Writes made
    ahead of the cursor are seen, as with an LMDB cursor.

    A top-level ``PrefixFilter`` break filter is pushed into SQL as a range
    bound, so a prefix scan stops reading at the end of the prefix instead
    of pulling a page of rows past it. Filter/break_filter/limit semantics
    are otherwise identical to the other adapters.
    """

    def __init__(self, context: ContextBase, options: StorageScanOptions) -> None:
        """Initialize scan iterator."""
        self._context = context
        self._storage = cast("SQLiteStorage", context._storage)
        self._options = options

    def _encoded_bounds(self) -> tuple[object | None, object | None]:
        """Return the (start, far) encoded bounds for the scan.

        ``start`` is inclusive and is where the scan begins (the low end
        forward, the high end reverse). ``far`` is the pushed-down prefix
        bound on the other side: exclusive high end forward, inclusive low
        end reverse. Either may be None.
        """
        codec = self._storage.codec
        options = self._options

        start: object | None = None
        if options.start_encoded is not None:
            # Raw pre-encoded bound (e.g. codec.upper_bound_of_prefix output).
            start = options.start_encoded
        elif options.start is not None:
            try:
                start = codec.encode_key(options.start)
            except Exception as e:
                raise StorageOperationError(f"Failed to encode start key: {e}") from e

        far: object | None = None
        brk = options.break_filter
        if isinstance(brk, PrefixFilter) and brk.prefix:
            # Every key matching the prefix encodes as encode(prefix) plus a
            # suffix, so it sits in [encode(prefix), upper_bound_of_prefix).
            # Iteration breaks at the first key outside the prefix anyway, so
            # clipping the SQL range there yields exactly the same rows.
            try:
                if options.reverse:
                    far = codec.encode_key(brk.prefix)
                else:
                    far = codec.upper_bound_of_prefix(brk.prefix)
            except Exception:
                far = None
            if not isinstance(far, bytes):
                far = None
        return start, far

    def _fetch_page(
        self,
        cols: str,
        order: str,
        bound: object | None,
        op: str,
        far: object | None,
        far_op: str,
        page: int,
    ) -> list[tuple]:
        """Fetch one page of raw ``(k,)`` or ``(k, v)`` rows in scan order."""
        conn = self._context._require_active()
        where: list[str] = []
        params: list[object] = []
        if bound is not None:
            where.append(f"k {op} ?")
            params.append(bound)
        if far is not None:
            where.append(f"k {far_op} ?")
            params.append(far)
        sql = f"SELECT {cols} FROM kv"  # noqa: S608 (fixed fragments, values bound)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += f" ORDER BY k {order} LIMIT ?"
        params.append(page)

        try:
            return conn.execute(sql, params).fetchall()
        except Exception as e:
            raise StorageOperationError(f"Failed to scan: {e}") from e

    def _iterate_impl(self, iterator_type: _IteratorType) -> Generator[object, None, None]:
        """Core iteration implementation.

        Pages by key. A ``max_depth`` skip drops the rest of the current
        page and re-seeks with an inclusive bound at the skip target, on a
        small page since the next useful row is usually right there.
        """
        self._context._require_active()
        codec = self._storage.codec
        options = self._options
        need_values = iterator_type != _IteratorType.KEYS
        max_depth = options.max_depth

        cols = "k, v" if need_values else "k"
        order = "DESC" if options.reverse else "ASC"
        # First page is inclusive of `start`; later pages exclude the last key.
        first_op, next_op, far_op = ("<=", "<", ">=") if options.reverse else (">=", ">", "<")

        bound, far = self._encoded_bounds()
        op = first_op
        page = _FIRST_PAGE
        if options.limit is not None and options.filter is None:
            page = max(1, min(page, options.limit))

        count = 0
        while True:
            rows = self._fetch_page(cols, order, bound, op, far, far_op, page)
            if not rows:
                return

            skipped = False
            for row in rows:
                try:
                    key = codec.decode_key(row[0])
                except Exception as e:
                    raise StorageOperationError(f"Failed to decode key: {e}") from e

                if options.break_filter is not None and not options.break_filter.matches(key):
                    return

                if max_depth is not None and len(key) > max_depth:
                    # Below the walked depth: re-seek past the whole subtree.
                    bound = options.skip_bound(codec, key)
                    op = first_op
                    page = _SEEK_PAGE
                    skipped = True
                    break

                if options.filter is not None and not options.filter.matches(key):
                    continue

                if options.limit is not None and count >= options.limit:
                    return

                value = None
                if need_values:
                    try:
                        value = codec.decode_value(row[1])
                    except Exception as e:
                        raise StorageOperationError(f"Failed to decode value: {e}") from e

                if iterator_type == _IteratorType.KEYS:
                    yield key
                elif iterator_type == _IteratorType.VALUES:
                    yield value
                else:
                    yield (key, value)

                count += 1

            if skipped:
                continue
            if len(rows) < page:
                return
            bound = rows[-1][0]
            op = next_op
            page = min(page * 2, _MAX_PAGE)

    def keys(self) -> Generator[Key, None, None]:
        """Iterate over keys only."""
        return cast(
            "Generator[Key, None, None]", self._iterate_impl(iterator_type=_IteratorType.KEYS)
        )

    def values(self) -> Generator[Value, None, None]:
        """Iterate over values only."""
        return cast(
            "Generator[Value, None, None]", self._iterate_impl(iterator_type=_IteratorType.VALUES)
        )

    def items(self) -> Generator[tuple[Key, Value], None, None]:
        """Iterate over (key, value) tuples."""
        return cast(
            "Generator[tuple[Key, Value], None, None]",
            self._iterate_impl(iterator_type=_IteratorType.ITEMS),
        )
