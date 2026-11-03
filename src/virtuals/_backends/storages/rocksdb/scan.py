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

from .context import ScanHandle, classify_rdbpy, is_missing_file_error


# Bounded restarts when a secondary scan touches an SST the primary
# compacted away. Mirrors the point-read retry budget in `context.py`.
_SCAN_STALE_RETRIES = 6

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
                cause = e.__cause__
                if not (
                    storage._is_secondary and cause is not None and is_missing_file_error(cause)
                ):
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
            self._seek(iterator, options.reverse, start_key_encoded)

            count = 0
            max_depth = options.max_depth
            first_batch = _BATCH_FIRST
            if options.limit is not None:
                first_batch = max(1, min(first_batch, options.limit))
            batch_size = first_batch
            while True:
                rows = self._next_batch(iterator, batch_size)
                if not rows:
                    return
                batch_size = min(batch_size * 4, _BATCH_MAX)
                for row in rows:
                    # Rows already fetched outlive the context (and on a
                    # secondary the shared DB handle outlives the snapshot),
                    # so check the context before every row, not the cursor.
                    if context.is_closed:
                        raise StorageClosedError("Scan used after its context ended")
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
                        self._seek(iterator, options.reverse, options.skip_bound(codec, key))
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
    def _seek(iterator: Any, reverse: bool, start_key_encoded: bytes | None) -> None:  # noqa: ANN401
        """Position the iterator on the first in-range key.

        Forward lands on the first key >= start, reverse on the last key
        <= start (`seek_for_prev`). An empty range leaves the iterator
        invalid, which the first batch reports as exhausted.
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
        except Exception as e:
            raise classify_rdbpy(e, "Failed during scan") from e

    @staticmethod
    def _next_batch(iterator: Any, n: int) -> list:  # noqa: ANN401
        """Up to ``n`` rows from the cursor in its direction; empty when exhausted."""
        try:
            return iterator.next_batch(n)
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
