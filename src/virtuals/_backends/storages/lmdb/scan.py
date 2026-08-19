"""Scan iterator for LMDB storage."""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING, cast

from virtuals.tkv.storage import (
    ScanProtocol,
    StorageOperationError,
    StorageScanOptions,
)


if TYPE_CHECKING:
    from collections.abc import Generator

    from virtuals.tkv.types import Key, Value

    from .context import ContextBase
    from .storage import LMDBStorage


__all__ = ["LMDBScan"]


class _IteratorType(Enum):
    """Type of scan iteration."""

    KEYS = auto()
    VALUES = auto()
    ITEMS = auto()


class LMDBScan(ScanProtocol):
    """Scan iterator for LMDB storage.

    Uses a cursor over the current transaction. Forward scans use
    `iternext`; reverse scans use `iterprev`. Start bounds are honored
    with `set_range` (forward) or a manual seek (reverse), keeping the
    filter/break_filter/limit semantics identical to other adapters.
    """

    def __init__(self, context: ContextBase, options: StorageScanOptions) -> None:
        """Initialize scan iterator."""
        self._context = context
        self._storage = cast("LMDBStorage", context._storage)
        self._options = options

    def _iterate_impl(self, iterator_type: _IteratorType) -> Generator[object, None, None]:
        """Core iteration implementation."""
        txn = self._context._require_active()
        codec = self._storage.codec
        options = self._options

        need_values = iterator_type != _IteratorType.KEYS

        start_key_encoded: bytes | None = None
        if options.start_encoded is not None:
            # Raw pre-encoded bound (e.g. codec.upper_bound_of_prefix output).
            start_key_encoded = cast("bytes", options.start_encoded)
        elif options.start is not None:
            try:
                start_key_encoded = codec.encode_key(options.start)
            except Exception as e:
                raise StorageOperationError(f"Failed to encode start key: {e}") from e

        try:
            cursor = txn.cursor()
        except Exception as e:
            raise StorageOperationError(f"Failed to create cursor: {e}") from e

        try:
            positioned = self._seek(cursor, options, start_key_encoded)
            if not positioned:
                return

            count = 0

            while True:
                try:
                    encoded_key = cursor.key()
                    encoded_value = cursor.value() if need_values else None
                except Exception as e:
                    raise StorageOperationError(f"Failed to read cursor: {e}") from e

                if not encoded_key:
                    break

                try:
                    key = codec.decode_key(encoded_key)
                except Exception as e:
                    raise StorageOperationError(f"Failed to decode key: {e}") from e

                if options.break_filter is not None and not options.break_filter.matches(key):
                    break

                if options.filter is not None and not options.filter.matches(key):
                    if not self._advance(cursor, options.reverse):
                        break
                    continue

                if options.limit is not None and count >= options.limit:
                    break

                value = None
                if need_values and encoded_value is not None:
                    try:
                        value = codec.decode_value(encoded_value)
                    except Exception as e:
                        raise StorageOperationError(f"Failed to decode value: {e}") from e

                if iterator_type == _IteratorType.KEYS:
                    yield key
                elif iterator_type == _IteratorType.VALUES:
                    yield value
                else:
                    yield (key, value)

                count += 1

                if not self._advance(cursor, options.reverse):
                    break
        finally:
            try:
                cursor.close()
            except Exception as e:
                # Cursor close is best-effort; the transaction owning it will
                # release the cursor when it ends. Log and move on so the
                # underlying scan error (if any) is what propagates.
                import logging

                logging.getLogger(__name__).debug("LMDB cursor close failed: %s", e)

    @staticmethod
    def _seek(cursor: object, options: StorageScanOptions, start_key_encoded: bytes | None) -> bool:
        """Position cursor at the first item in-range.

        Returns True if the cursor is on a valid item; False if the range
        is empty (nothing to yield).
        """
        try:
            if options.reverse:
                if start_key_encoded is None:
                    return bool(cursor.last())  # type: ignore[attr-defined]
                if cursor.set_range(start_key_encoded):  # type: ignore[attr-defined]
                    # set_range lands on the first key >= start; for a reverse
                    # scan we want the first key <= start, so step back if we
                    # overshot. If we land exactly on start, keep it (inclusive).
                    if cursor.key() != start_key_encoded:  # type: ignore[attr-defined]
                        if not cursor.prev():  # type: ignore[attr-defined]
                            return False
                    return True
                # No key >= start exists -> tail of db is <= start.
                return bool(cursor.last())  # type: ignore[attr-defined]
            else:
                if start_key_encoded is None:
                    return bool(cursor.first())  # type: ignore[attr-defined]
                return bool(cursor.set_range(start_key_encoded))  # type: ignore[attr-defined]
        except Exception as e:
            raise StorageOperationError(f"Failed to seek cursor: {e}") from e

    @staticmethod
    def _advance(cursor: object, reverse: bool) -> bool:
        """Step cursor forward or backward. Returns False on exhaustion."""
        try:
            if reverse:
                return bool(cursor.prev())  # type: ignore[attr-defined]
            return bool(cursor.next())  # type: ignore[attr-defined]
        except Exception as e:
            raise StorageOperationError(f"Failed to advance cursor: {e}") from e

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
