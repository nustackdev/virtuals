# Storage Layer: Scan Operation

`scan` provides ordered, streaming access to key ranges.

It is a **system-level iteration primitive**, not a collection.

## Availability

`scan` is supported in:

- Snapshot
- Transaction

`scan` is not supported in:

- WriteBatch

## Semantics

- Iteration is backend-driven
- Results are streamed, not materialized
- Order follows the storage engine's native key order
- No guarantees beyond the active context

## Constraints

- `scan` does not imply isolation beyond the context
- `scan` does not observe uncommitted writes outside the context
- a scan in flight may or may not see writes its own transaction makes after it started; only scans started after a write are guaranteed to see it (see [contract](contract.md), section 1)
- `scan` performs no implicit buffering or retries

## Role

`scan` exists to:

- traverse keyspaces
- power higher-level iteration utilities
- expose backend iteration without abstraction leaks

Any stronger guarantees belong in higher layers.

## Mechanics

scan(self, options: StorageScanOptions) -> ScanProtocol:

StorageScanOptions (a frozen dataclass, every field optional):
    start: first key visited, always inclusive. None means the first key
        of the store (forward) or the last (reverse).
    reverse: False walks forward in key order, True walks backward.
    limit: stop after this many yielded rows. None means no cap.
    filter: a key is yielded only when it matches.
    break_filter: the scan stops at the first key that does not match
        (a PrefixFilter here bounds the walk to a subtree).
    start_encoded: a pre-encoded start bound; wins over start when set.
    max_depth: deepest key length walked row by row; longer keys are
        skipped with one seek per subtree, so a child scan costs
        O(children). Must be at least 1.

ScanProtocol:
    def items(self) -> Generator[tuple[key.Key, Value], None, None]:
        """Iterate over (key, value) tuples.

        Yields:
            Tuples of (key, value) for each item in scan range.

        Raises:
            StorageOperationError: If iteration fails.
        """
        ...

    def keys(self) -> Generator[key.Key, None, None]:
        """Iterate over keys only.

        Yields:
            Keys in scan range.

        Raises:
            StorageOperationError: If iteration fails.
        """
        ...

    def values(self) -> Generator[Value, None, None]:
        """Iterate over values only.

        Yields:
            Values in scan range.

        Raises:
            StorageOperationError: If iteration fails.
        """
        ...

Only keys/values are yielded if all Filters match, otherwise items are skipped.
Filters are hashable.
