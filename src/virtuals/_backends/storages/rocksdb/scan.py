"""Scan iterator for RocksDB storage."""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING, cast

from virtuals.tkv.storage import (
    ScanProtocol,
    StorageOperationError,
    StorageScanOptions,
)

from .context import _is_missing_file_error


# Bounded restarts when a secondary scan touches an SST the primary
# compacted away. Mirrors the point-read retry budget in `context.py`.
_SCAN_STALE_RETRIES = 6


if TYPE_CHECKING:
    from collections.abc import Generator

    from virtuals.tkv.types import Key, Value

    from .context import ContextBase
    from .storage import RocksDBStorage


__all__ = ["RocksDBScan"]


class _IteratorType(Enum):
    """Type of scan iteration."""

    KEYS = auto()
    VALUES = auto()
    ITEMS = auto()


class RocksDBScan(ScanProtocol):
    """Optimized scan iterator implementation conforming to ScanProtocol.

    Provides Pythonic iteration interface over a range of keys.

    Key optimizations:
    1. Uses iterkeys() when only keys needed (no value I/O from disk)
    2. Uses iteritems() when values needed
    3. Uses filter/break_filter for flexible key filtering
    """

    def __init__(
        self,
        context: ContextBase,
        options: StorageScanOptions,
    ) -> None:
        """Initialize scan iterator.

        Args:
            context: Storage context (transaction/snapshot)
            options: Scan configuration
        """
        self._context = context
        self._storage = cast("RocksDBStorage", context._storage)
        self._options = options

    def _iterate_impl(self, iterator_type: _IteratorType) -> Generator[object, None, None]:
        """Stale-secondary-safe wrapper around `_scan_once`.

        A read-only secondary can be pinned to a manifest version that
        references an SST the primary already compacted away; an iterator
        touching that file raises an IO error. The fix is the same as for
        point reads: catch up to the current manifest and retry.

        A scan cannot retry a single key, so it restarts the whole pass and
        skips the items already emitted. This is exact for append-only /
        immutable ranges (the ledger's per-block tx dicts -- a synced block
        never changes); for a range mutated concurrently a restart may shift
        which rows land after the skip, but that is strictly better than
        crashing the reader, and a secondary scan is already only
        eventually-consistent across refreshes.
        """
        storage = self._storage
        yielded = 0
        attempt = 0
        while True:
            skip = yielded
            try:
                for item in self._scan_once(iterator_type):
                    if skip > 0:
                        skip -= 1
                        continue
                    yielded += 1
                    yield item
                return
            except Exception as e:
                if not (storage._is_secondary and _is_missing_file_error(e)):
                    raise
                attempt += 1
                if attempt > _SCAN_STALE_RETRIES:
                    raise
                storage.force_catch_up_with_primary()

    def _scan_once(self, iterator_type: _IteratorType) -> Generator[object, None, None]:
        """One full iteration pass over the configured range.

        May raise mid-stream; `_iterate_impl` handles a stale-secondary
        failure by restarting this pass and skipping already-emitted items.

        Args:
            iterator_type: Type of iteration (keys/values/items)

        Yields:
            Keys, values, or (key, value) tuples based on iterator_type
        """
        txn = self._context._require_active()
        codec = self._storage.codec
        options = self._options

        # Create appropriate iterator based on what we need
        # Note: iterkeys() doesn't support reverse iteration (skip_back),
        # so we must use iteritems() for reverse scans
        need_values = iterator_type != _IteratorType.KEYS
        use_items_iterator = need_values or options.reverse
        try:
            if use_items_iterator:
                iterator = txn.iteritems()  # Read both keys and values
            else:
                iterator = txn.iterkeys()  # Read only keys from disk (forward only)
        except Exception as e:
            raise StorageOperationError(f"Failed to create iterator: {e}") from e

        try:
            # Encode start bound for comparison. `start_encoded` (raw bytes)
            # wins when provided — lets callers pass codec sentinels like
            # `upper_bound_of_prefix` that can't be spelled as a plain tuple.
            if options.start_encoded is not None:
                start_key_encoded = cast("bytes", options.start_encoded)
            elif options.start:
                start_key_encoded = codec.encode_key(options.start)
            else:
                start_key_encoded = b""

            # Seek to start based on direction
            try:
                if options.reverse:
                    # For reverse, start from end
                    iterator.seek_to_last()
                else:
                    # For forward, seek to start
                    if start_key_encoded:
                        iterator.seek(start_key_encoded)
                    else:
                        iterator.seek_to_first()
            except ValueError:
                # Iterator exhausted immediately
                return

            # Track count for limit
            count = 0

            # Iterate through range
            while True:
                try:
                    if use_items_iterator:
                        encoded_key, encoded_value = iterator.get()
                    else:
                        encoded_key = iterator.get()
                        encoded_value = None
                except (ValueError, IndexError):
                    # Iterator exhausted
                    break

                # Check start bound (compare encoded keys)
                if start_key_encoded:
                    if options.reverse:
                        # For reverse, skip keys > start
                        if encoded_key > start_key_encoded:
                            try:
                                iterator.skip_back()
                            except ValueError:
                                break
                            continue
                    else:
                        # For forward, skip keys < start
                        if encoded_key < start_key_encoded:
                            try:
                                iterator.skip()
                            except ValueError:
                                break
                            continue

                # Decode key (needed for filtering)
                try:
                    key = codec.decode_key(encoded_key)
                except Exception as e:
                    raise StorageOperationError(f"Failed to decode key: {e}") from e

                # Check break_filter first - stop iteration if key doesn't match
                if options.break_filter is not None:
                    if not options.break_filter.matches(key):
                        break

                # Check filter - skip keys that don't match
                if options.filter is not None:
                    if not options.filter.matches(key):
                        try:
                            if options.reverse:
                                iterator.skip_back()
                            else:
                                iterator.skip()
                        except ValueError:
                            break
                        continue

                # Check limit
                if options.limit is not None and count >= options.limit:
                    break

                # Decode value if needed
                value = None
                if need_values and encoded_value is not None:
                    try:
                        value = codec.decode_value(encoded_value)
                    except Exception as e:
                        raise StorageOperationError(f"Failed to decode value: {e}") from e

                # Yield based on iterator type
                if iterator_type == _IteratorType.KEYS:
                    yield key
                elif iterator_type == _IteratorType.VALUES:
                    yield value
                elif iterator_type == _IteratorType.ITEMS:
                    yield (key, value)

                count += 1

                # Advance iterator
                try:
                    if options.reverse:
                        iterator.skip_back()
                    else:
                        iterator.skip()
                except ValueError:
                    break

        except Exception as e:
            raise StorageOperationError(f"Failed during scan: {e}") from e
        finally:
            # Release iterator to free C++ resources
            del iterator

    def keys(self) -> Generator[Key, None, None]:
        """Iterate over keys only - uses iterkeys() for minimal I/O.

        Yields:
            Keys in scan range

        Raises:
            StorageOperationError: If iteration fails
        """
        return cast(
            "Generator[Key, None, None]", self._iterate_impl(iterator_type=_IteratorType.KEYS)
        )

    def values(self) -> Generator[Value, None, None]:
        """Iterate over values only - must use iteritems().

        Yields:
            Values in scan range

        Raises:
            StorageOperationError: If iteration fails
        """
        return cast(
            "Generator[Value, None, None]", self._iterate_impl(iterator_type=_IteratorType.VALUES)
        )

    def items(self) -> Generator[tuple[Key, Value], None, None]:
        """Iterate over (key, value) tuples - uses iteritems().

        Yields:
            Tuples of (key, value) for each item in scan range

        Raises:
            StorageOperationError: If iteration fails
        """
        return cast(
            "Generator[tuple[Key, Value], None, None]",
            self._iterate_impl(iterator_type=_IteratorType.ITEMS),
        )
