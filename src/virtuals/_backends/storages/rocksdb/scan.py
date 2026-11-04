"""Scan iterator for RocksDB storage.

Rows are fetched in batches, so a write made through the same transaction
while a scan is in flight is seen only once the scan moves past the batch
it already holds. Whether an in-flight scan sees such a write is
unspecified across backends (contract section 1); callers should not rely
on either outcome. Deleting keys the scan has already yielded is safe.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING, cast

from virtuals.tkv.storage import (
    ScanProtocol,
    StorageClosedError,
    StorageOperationError,
    StorageScanOptions,
)

from .context import ScanHandle, classify_rdbpy, is_missing_file_error, secondary_view_lost


# Rows fetched per native call. The first batch is small so a scan that
# stops early (limit, break filter) reads little past its end; later
# batches grow so a long scan pays one native call per hundreds of rows.
_BATCH_FIRST = 8
_BATCH_MAX = 1024


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
    """Scan iterator implementation conforming to ScanProtocol.

    Provides Pythonic iteration interface over a range of keys, read at the
    context's snapshot (a transaction's own, merged with its pending
    writes, or a Snapshot's).

    Key optimizations:
    1. Key-only cursors when only keys are needed (no value I/O from disk),
       in both directions
    2. Seeks straight to the start bound (seek / seek_for_prev)
    3. Rows come in native batches, so Python pays per batch, not per row
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
        """Stale-secondary-aware wrapper around `_scan_once`.

        A read-only secondary can be pinned to a manifest version that
        references an SST the primary already compacted away; an iterator
        touching that file raises an IO error. As for point reads, the
        storage catches up so the next snapshot reads, and this scan raises
        the retryable ``StorageTransactionConflictError``: restarting it on
        the new state would mix two versions into one snapshot's view.
        """
        storage = self._storage
        try:
            yield from self._scan_once(iterator_type)
        except Exception as e:
            cause = e.__cause__
            if storage._is_secondary and cause is not None and is_missing_file_error(cause):
                raise secondary_view_lost(storage, cause) from cause
            raise

    def _scan_once(self, iterator_type: _IteratorType) -> Generator[object, None, None]:
        """One full iteration pass over the configured range.

        May raise mid-stream; `_iterate_impl` handles a stale-secondary
        failure by restarting this pass and skipping already-emitted items.

        The rdbpy iterator lives only as long as the context: committing,
        aborting or closing the context (or the storage) invalidates it, and
        resuming the scan after that raises `StorageClosedError`. The
        context closes the iterators of its live scans when it ends, so a
        suspended scan does not pin the iterator's files until collected.

        Args:
            iterator_type: Type of iteration (keys/values/items)

        Yields:
            Keys, values, or (key, value) tuples based on iterator_type
        """
        context = self._context
        context._require_active()
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
            iterator = context._iterator(keys_only=not need_values, reverse=options.reverse)  # type: ignore[attr-defined]
        except Exception as e:
            raise classify_rdbpy(e, "Failed to create iterator") from e
        handle = ScanHandle(iterator)  # lives in this frame: the registry forgets it with the scan
        context._register_scan(handle)

        try:
            self._seek(handle, options.reverse, start_key_encoded)

            count = 0
            max_depth = options.max_depth
            first_batch = _BATCH_FIRST
            if options.limit is not None:
                first_batch = max(1, min(first_batch, options.limit))
            batch_size = first_batch
            while True:
                rows = self._next_batch(handle, batch_size)
                if not rows:
                    return
                batch_size = min(batch_size * 4, _BATCH_MAX)
                for row in rows:
                    # Rows already fetched outlive the context (and on a
                    # secondary the shared DB handle outlives the snapshot,
                    # and a catch-up retires its view), so check the
                    # context before every row, not the cursor.
                    if context.is_closed:
                        raise StorageClosedError("Scan used after its context ended")
                    context._require_active()
                    if need_values:
                        encoded_key, encoded_value = row
                    else:
                        encoded_key, encoded_value = row, None

                    try:
                        key = codec.decode_key(encoded_key)
                    except Exception as e:
                        raise StorageOperationError(f"Failed to decode key: {e}") from e

                    if options.break_filter is not None and not options.break_filter.matches(key):
                        return

                    if max_depth is not None and len(key) > max_depth:
                        # Below the walked depth: one seek over the whole subtree,
                        # dropping the rest of this batch.
                        self._seek(handle, options.reverse, options.skip_bound(codec, key))
                        batch_size = first_batch
                        break

                    if options.filter is not None and not options.filter.matches(key):
                        continue

                    if options.limit is not None and count >= options.limit:
                        return

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
        finally:
            handle.close()

    @staticmethod
    def _seek(handle: ScanHandle, reverse: bool, start_key_encoded: bytes | None) -> None:
        """Position the iterator on the first in-range key.

        Forward lands on the first key >= start, reverse on the last key
        <= start (`seek_for_prev`). An empty range leaves the iterator
        invalid, which the first batch reports as exhausted.
        """
        try:
            if reverse:
                if start_key_encoded:
                    handle.step("seek_for_prev", start_key_encoded)
                else:
                    handle.step("seek_to_last")
            elif start_key_encoded:
                handle.step("seek", start_key_encoded)
            else:
                handle.step("seek_to_first")
        except StorageClosedError:
            raise
        except Exception as e:
            raise classify_rdbpy(e, "Failed during scan") from e

    @staticmethod
    def _next_batch(handle: ScanHandle, n: int) -> list:
        """Up to ``n`` rows from the cursor in its direction; empty when exhausted."""
        try:
            return handle.step("next_batch", n)
        except StorageClosedError:
            raise
        except Exception as e:
            raise classify_rdbpy(e, "Failed during scan") from e

    def keys(self) -> Generator[Key, None, None]:
        """Iterate over keys only - a key-only cursor for minimal I/O.

        Yields:
            Keys in scan range

        Raises:
            StorageOperationError: If iteration fails
        """
        return cast("Generator[Key, None, None]", self._iterate_impl(_IteratorType.KEYS))

    def values(self) -> Generator[Value, None, None]:
        """Iterate over values only.

        Yields:
            Values in scan range

        Raises:
            StorageOperationError: If iteration fails
        """
        return cast("Generator[Value, None, None]", self._iterate_impl(_IteratorType.VALUES))

    def items(self) -> Generator[tuple[Key, Value], None, None]:
        """Iterate over (key, value) tuples.

        Yields:
            Tuples of (key, value) for each item in scan range

        Raises:
            StorageOperationError: If iteration fails
        """
        return cast(
            "Generator[tuple[Key, Value], None, None]", self._iterate_impl(_IteratorType.ITEMS)
        )
