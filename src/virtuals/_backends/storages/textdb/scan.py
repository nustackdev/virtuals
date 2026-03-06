"""Scan iterator for text storage."""

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
    from .storage import TextStorage


__all__ = ["TextScan"]


class _IteratorType(Enum):
    """Type of scan iteration."""

    KEYS = auto()
    VALUES = auto()
    ITEMS = auto()


class TextScan(ScanProtocol):
    """Scan iterator for text storage.

    Provides iteration over key-value pairs with filtering and ordering.
    """

    def __init__(self, context: ContextBase, options: StorageScanOptions) -> None:
        """Initialize scan.

        Args:
            context: Storage context (transaction or snapshot)
            options: Scan configuration
        """
        self._context = context
        self._storage = cast("TextStorage", context._storage)
        self._options = options

    def _iterate_impl(self, iterator_type: _IteratorType) -> Generator[object, None, None]:
        """Core iteration implementation.

        Args:
            iterator_type: Type of iteration (keys/values/items)

        Yields:
            Keys, values, or (key, value) tuples based on iterator_type
        """
        state = self._context._require_active()
        options = self._options
        codec = self._storage.codec

        # Encode start bound for comparison
        start_str = codec.encode_key(options.start) if options.start is not None else None

        # Get all encoded keys and sort lexicographically
        encoded_keys = sorted(state.keys(), reverse=options.reverse)

        # Iterate over sorted encoded keys
        count = 0
        for key_str in encoded_keys:
            # Check start bound (compare encoded keys)
            # Start is inclusive
            if start_str is not None:
                if options.reverse:
                    # For reverse, start from keys <= start
                    if key_str > start_str:
                        continue
                else:
                    # For forward, start from keys >= start
                    if key_str < start_str:
                        continue

            # Decode key
            try:
                key = codec.decode_key(key_str)
            except Exception as e:
                raise StorageOperationError(f"Failed to decode key {key_str}: {e}") from e

            # Check break_filter first - stop iteration if key doesn't match
            if options.break_filter is not None:
                if not options.break_filter.matches(key):
                    break

            # Check filter - skip keys that don't match
            if options.filter is not None:
                if not options.filter.matches(key):
                    continue

            # Check result limit
            if options.limit is not None and count >= options.limit:
                break

            # Decode value if needed
            if iterator_type != _IteratorType.KEYS:
                try:
                    value = codec.decode_value(state[key_str])
                except Exception as e:
                    raise StorageOperationError(f"Failed to decode value for {key_str}: {e}") from e
            else:
                value = None

            # Yield based on iterator type
            if iterator_type == _IteratorType.KEYS:
                yield key
            elif iterator_type == _IteratorType.VALUES:
                yield value
            elif iterator_type == _IteratorType.ITEMS:
                yield (key, value)

            count += 1

    def items(self) -> Generator[tuple[Key, Value], None, None]:
        """Iterate over (key, value) tuples.

        Yields:
            Tuples of (key, value) for each item in scan range.

        Raises:
            StorageOperationError: If iteration fails.
        """
        return cast(
            "Generator[tuple[Key, Value], None, None]", self._iterate_impl(_IteratorType.ITEMS)
        )

    def keys(self) -> Generator[Key, None, None]:
        """Iterate over keys only.

        Yields:
            Keys in scan range.

        Raises:
            StorageOperationError: If iteration fails.
        """
        return cast("Generator[Key, None, None]", self._iterate_impl(_IteratorType.KEYS))

    def values(self) -> Generator[Value, None, None]:
        """Iterate over values only.

        Yields:
            Values in scan range.

        Raises:
            StorageOperationError: If iteration fails.
        """
        return cast("Generator[Value, None, None]", self._iterate_impl(_IteratorType.VALUES))
