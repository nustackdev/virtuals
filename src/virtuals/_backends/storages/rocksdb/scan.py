"""Scan iterator for RocksDB storage."""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING, Any, cast

from virtuals.tkv.storage import (
    ScanProtocol,
    StorageClosedError,
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
    1. Uses iterkeys() when only keys needed (no value I/O from disk), in
       both directions
    2. Uses iteritems() when values needed
    3. Seeks straight to the start bound (seek / seek_for_prev)
    4. Uses filter/break_filter for flexible key filtering
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

        The rdbpy iterator lives only as long as the context: committing,
        aborting or closing the context (or the storage) invalidates it, and
        resuming the scan after that raises `StorageClosedError`.

        Args:
            iterator_type: Type of iteration (keys/values/items)

        Yields:
            Keys, values, or (key, value) tuples based on iterator_type
        """
        txn = self._context._require_active()
        codec = self._storage.codec
        options = self._options

        need_values = iterator_type != _IteratorType.KEYS

        # `start_encoded` (raw bytes) wins when provided: lets callers pass
        # codec sentinels like `upper_bound_of_prefix` that can't be spelled
        # as a plain tuple.
        start_key_encoded: bytes | None = None
        if options.start_encoded is not None:
            start_key_encoded = cast("bytes", options.start_encoded)
        elif options.start:
            try:
                start_key_encoded = codec.encode_key(options.start)
            except Exception as e:
                raise StorageOperationError(f"Failed to encode start key: {e}") from e

        try:
            iterator = txn.iteritems() if need_values else txn.iterkeys()
        except RuntimeError as e:
            raise StorageClosedError(f"Scan context is closed: {e}") from e
        except Exception as e:
            raise StorageOperationError(f"Failed to create iterator: {e}") from e

        self._seek(iterator, options.reverse, start_key_encoded)

        count = 0
        context = self._context

        while True:
            # A secondary's snapshot wraps the shared DB handle, so closing the
            # snapshot does not invalidate the iterator; check the context.
            if context.is_closed:
                raise StorageClosedError("Scan used after its context ended")
            entry = self._read(iterator)
            if entry is None:
                break
            if need_values:
                encoded_key, encoded_value = entry
            else:
                encoded_key, encoded_value = entry, None

            try:
                key = codec.decode_key(encoded_key)
            except Exception as e:
                raise StorageOperationError(f"Failed to decode key: {e}") from e

            if options.break_filter is not None and not options.break_filter.matches(key):
                break

            if options.filter is not None and not options.filter.matches(key):
                if not self._advance(iterator, options.reverse):
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

            if not self._advance(iterator, options.reverse):
                break

    @staticmethod
    def _seek(iterator: Any, reverse: bool, start_key_encoded: bytes | None) -> None:  # noqa: ANN401
        """Position the iterator on the first in-range key.

        Forward lands on the first key >= start, reverse on the last key
        <= start (`seek_for_prev`). An empty range leaves the iterator
        invalid, which `_read` reports as exhausted.
        """
        try:
            if reverse:
                if start_key_encoded:
                    iterator.seek_for_prev(start_key_encoded)
                else:
                    iterator.seek_to_last()
            elif start_key_encoded:
                iterator.seek(start_key_encoded)
            else:
                iterator.seek_to_first()
        except RuntimeError as e:
            raise StorageClosedError(f"Scan used after its context ended: {e}") from e
        except Exception as e:
            raise StorageOperationError(f"Failed during scan: {e}") from e

    @staticmethod
    def _read(iterator: Any) -> Any:  # noqa: ANN401
        """Current entry (key, or (key, value)), or None when exhausted."""
        try:
            return iterator.get()
        except ValueError:
            return None
        except RuntimeError as e:
            raise StorageClosedError(f"Scan used after its context ended: {e}") from e
        except Exception as e:
            raise StorageOperationError(f"Failed during scan: {e}") from e

    @staticmethod
    def _advance(iterator: Any, reverse: bool) -> bool:  # noqa: ANN401
        """Step forward or backward. Returns False on exhaustion."""
        try:
            if reverse:
                iterator.skip_back()
            else:
                iterator.skip()
        except ValueError:
            return False
        except RuntimeError as e:
            raise StorageClosedError(f"Scan used after its context ended: {e}") from e
        except Exception as e:
            raise StorageOperationError(f"Failed during scan: {e}") from e
        return True

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
