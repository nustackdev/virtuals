"""Scan iterator for in-memory storage."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from enum import Enum, auto
from typing import TYPE_CHECKING, cast

from virtuals.tkv.storage import (
    ScanProtocol,
    StorageOperationError,
    StorageScanOptions,
)

from .state import TransactionState


if TYPE_CHECKING:
    from collections.abc import Generator

    from virtuals.tkv.types import Key, Value

    from .context import ContextBase
    from .storage import InMemoryStorage


__all__ = ["InMemoryScan"]


class _IteratorType(Enum):
    """Type of scan iteration."""

    KEYS = auto()
    VALUES = auto()
    ITEMS = auto()


class InMemoryScan(ScanProtocol):
    """Scan iterator for in-memory storage.

    Provides iteration over key-value pairs with filtering and ordering.
    """

    def __init__(self, context: ContextBase, options: StorageScanOptions) -> None:
        """Initialize scan.

        Args:
            context: Storage context (transaction or snapshot)
            options: Scan configuration
        """
        self._context = context
        self._storage = cast("InMemoryStorage", context._storage)
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

        # Encode start bound for comparison. `start_encoded` wins when set —
        # lets callers pass codec sentinels like `upper_bound_of_prefix`.
        if options.start_encoded is not None:
            start_str = options.start_encoded
        elif options.start is not None:
            start_str = codec.encode_key(options.start)
        else:
            start_str = None

        # Every scan sorts the whole store: O(N log N) by design, the start
        # bound and `max_depth` skips are then bisects on the sorted list.
        encoded_keys = sorted(state.keys())
        n = len(encoded_keys)
        reverse = options.reverse
        step = -1 if reverse else 1
        max_depth = options.max_depth

        # Start is inclusive: forward the first key >= start, reverse the
        # last key <= start.
        if start_str is None:
            i = n - 1 if reverse else 0
        elif reverse:
            i = bisect_right(encoded_keys, start_str) - 1
        else:
            i = bisect_left(encoded_keys, start_str)

        count = 0
        while 0 <= i < n:
            key_str = encoded_keys[i]

            # Decode key
            try:
                key = codec.decode_key(key_str)
            except Exception as e:
                raise StorageOperationError(f"Failed to decode key {key_str}: {e}") from e

            # Check break_filter first - stop iteration if key doesn't match
            if options.break_filter is not None:
                if not options.break_filter.matches(key):
                    break

            # Below the walked depth: jump over the whole subtree
            if max_depth is not None and len(key) > max_depth:
                bound = options.skip_bound(codec, key)
                if reverse:
                    i = bisect_right(encoded_keys, bound) - 1
                else:
                    i = bisect_left(encoded_keys, bound)
                continue

            # Check filter - skip keys that don't match
            if options.filter is not None:
                if not options.filter.matches(key):
                    i += step
                    continue

            # Check result limit
            if options.limit is not None and count >= options.limit:
                break

            # Get and decode value if needed
            if iterator_type != _IteratorType.KEYS:
                try:
                    if isinstance(state, TransactionState):
                        value_encoded = state.get(key_str)
                    else:
                        value_encoded = state[key_str]
                    value = codec.decode_value(value_encoded)
                except KeyError:
                    i += step
                    continue
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
            i += step

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
