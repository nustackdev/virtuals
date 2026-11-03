"""Scan iterator for the in-memory storage.

Pages through the context's view by re-seeking from the last key of the
previous page, so no cursor is held over the index while other threads
commit or while the caller writes through the same transaction. A write
made through the transaction during a scan is seen once the scan seeks
past the page it has already fetched; rows of that page are yielded as
fetched. Callers should not rely on either outcome. Pages start small
and double, so a short prefix scan reads little past its end and a long
one pays few seeks. A top-level ``PrefixFilter`` break filter
is pushed down as a range stop, exactly as the SQLite adapter does.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING, Any, cast

from virtuals.tkv.filter import PrefixFilter
from virtuals.tkv.storage import (
    ScanProtocol,
    StorageOperationError,
    StorageScanOptions,
)


if TYPE_CHECKING:
    from collections.abc import Generator

    from virtuals.tkv.types import Key, Value

    from .context import ReadOperationsMixin
    from .storage import InMemoryStorage


__all__ = ["InMemoryScan"]


_FIRST_PAGE = 32
_MAX_PAGE = 1024


class _IteratorType(Enum):
    """What the scan yields."""

    KEYS = auto()
    VALUES = auto()
    ITEMS = auto()


class InMemoryScan(ScanProtocol):
    """Lazy, paged scan over a snapshot's or transaction's view.

    ``start`` (or ``start_encoded``, which wins) is inclusive and is where
    the scan begins: the low end forward, the high end in reverse.
    ``break_filter`` stops the scan at the first key it rejects, ``filter``
    skips keys, and ``limit`` counts the rows that survive both.
    """

    def __init__(self, context: ReadOperationsMixin, options: StorageScanOptions) -> None:
        """Bind the scan to its context and options; nothing is read until iterated."""
        self._context = context
        self._storage = cast("InMemoryStorage", context._storage)
        self._options = options

    def _bounds(self) -> tuple[Any, Any]:
        """``(start, stop)`` in encoded form; either may be None."""
        codec = self._storage.codec
        options = self._options

        start: Any = None
        if options.start_encoded is not None:
            start = options.start_encoded
        elif options.start is not None:
            try:
                start = codec.encode_key(options.start)
            except Exception as e:
                raise StorageOperationError(f"Failed to encode start key: {e}") from e

        stop: Any = None
        brk = options.break_filter
        if isinstance(brk, PrefixFilter) and brk.prefix:
            # Keys under the prefix sit in [encode(prefix), upper_bound_of_prefix).
            # The break filter would end the scan at the first key outside
            # anyway; stopping the page there just avoids reading past it.
            try:
                if options.reverse:
                    stop = codec.encode_key(brk.prefix)
                else:
                    stop = codec.upper_bound_of_prefix(brk.prefix)  # type: ignore[attr-defined]
            except Exception:
                stop = None
        return start, stop

    def _iterate_impl(self, iterator_type: _IteratorType) -> Generator[object, None, None]:
        """Walk the view page by page, applying filters and limit."""
        context = self._context
        context._require_active()
        codec = self._storage.codec
        options = self._options
        reverse = options.reverse
        break_filter = options.break_filter
        key_filter = options.filter
        limit = options.limit
        max_depth = options.max_depth
        need_values = iterator_type is not _IteratorType.KEYS

        if limit is not None and limit <= 0:
            return
        bound, stop = self._bounds()
        inclusive = True
        page = _FIRST_PAGE
        if limit is not None and key_filter is None:
            page = max(1, min(page, limit))

        count = 0
        while True:
            context._require_active()
            try:
                rows, cover, done = context._page(
                    bound, inclusive=inclusive, reverse=reverse, n=page, stop=stop
                )
            except TypeError as e:
                # A start_encoded of the wrong type cannot be ordered against the keys.
                raise StorageOperationError(f"Failed to scan: {e}") from e
            skip_to = None
            for _, (key, encoded) in rows:
                if break_filter is not None and not break_filter.matches(key):
                    return
                if max_depth is not None and len(key) > max_depth:
                    # Below the walked depth: seek over the whole subtree.
                    skip_to = options.skip_bound(codec, key)
                    break
                if key_filter is not None and not key_filter.matches(key):
                    continue
                if not need_values:
                    yield key
                else:
                    try:
                        value = codec.decode_value(encoded)
                    except Exception as e:
                        raise StorageOperationError(f"Failed to decode value: {e}") from e
                    yield value if iterator_type is _IteratorType.VALUES else (key, value)
                count += 1
                if count == limit:
                    return
            if skip_to is not None:
                bound, inclusive = skip_to, True
                continue
            if done:
                return
            bound, inclusive = cover, False
            page = min(page * 2, _MAX_PAGE)

    def items(self) -> Generator[tuple[Key, Value], None, None]:
        """Iterate over ``(key, value)`` pairs in scan order."""
        return cast(
            "Generator[tuple[Key, Value], None, None]", self._iterate_impl(_IteratorType.ITEMS)
        )

    def keys(self) -> Generator[Key, None, None]:
        """Iterate over keys only; values are not decoded."""
        return cast("Generator[Key, None, None]", self._iterate_impl(_IteratorType.KEYS))

    def values(self) -> Generator[Value, None, None]:
        """Iterate over values only."""
        return cast("Generator[Value, None, None]", self._iterate_impl(_IteratorType.VALUES))
