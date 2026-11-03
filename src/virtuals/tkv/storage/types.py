"""Storage type definitions.

Defines data structures and type aliases used across the storage layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias


if TYPE_CHECKING:
    from ..codec import CodecProtocol
    from ..filter import Filter
    from ..types import Key
    from .context import SnapshotProtocol, TransactionProtocol, WriteBatchProtocol


@dataclass(frozen=True, kw_only=True)
class StorageScanOptions:
    """Options for range scan operations.

    Defines the starting position, direction, limits, and filtering
    for iterating over key ranges.

    Attributes:
        start: Starting key (inclusive). None means from beginning.
        reverse: If True, scan in reverse order.
        limit: Maximum number of results. None means unlimited.
        filter: Filter to include keys (skip if not matched).
        break_filter: Filter to break iteration (stop when not matched).
            Used for efficient prefix scans - break when past prefix range.

    Filter vs break_filter:
        - filter: If key doesn't match, SKIP it (continue to next)
        - break_filter: If key doesn't match, STOP iteration entirely

    Examples:
        >>> # Scan all keys starting from ("users",)
        >>> options = StorageScanOptions(start=("users",))

        >>> # Scan with prefix filter (efficient - breaks when past prefix)
        >>> from virtuals.tkv.storage.filter import PrefixFilter, LengthFilter
        >>> options = StorageScanOptions(
        ...     start=("users",),
        ...     break_filter=PrefixFilter(prefix=("users",)),
        ...     filter=LengthFilter(length=3),  # only length-3 keys
        ... )

        >>> # Scan children: prefix + length filter
        >>> options = StorageScanOptions(
        ...     start=("users",),
        ...     break_filter=PrefixFilter(prefix=("users",)),
        ...     filter=PrefixFilter(prefix=("users",)) & LengthFilter(length=2),
        ... )
    """

    start: Key | None = None
    reverse: bool = False
    limit: int | None = None
    filter: Filter | None = None
    break_filter: Filter | None = None
    start_encoded: object | None = None
    """Pre-encoded start bound. If set, backends use it directly instead of
    encoding ``start`` — the door to raw sentinels like a codec's
    ``upper_bound_of_prefix`` output (which cannot be expressed as a plain
    tuple key). When both are set, ``start_encoded`` wins."""
    max_depth: int | None = None
    """Deepest key length the scan walks row by row. A key longer than this
    is not visited: the scan jumps over the whole subtree of its ancestor
    at ``max_depth`` with one seek (past it forward, onto it reverse), see
    ``skip_bound``. Child scans set ``len(site) + 1`` so they cost
    O(children), not O(descendants). Shorter keys are still visited, so
    pair it with a filter when they are unwanted."""

    def skip_bound(self, codec: CodecProtocol, key: Key) -> object:
        """Encoded seek target that leaves the subtree ``key`` sits in.

        ``key`` is longer than ``max_depth``. Its ancestor at ``max_depth``
        roots the subtree to skip: forward, the codec's upper bound of that
        ancestor is the first encoded key past the subtree; reverse, the
        ancestor's own encoding is the last key at or below it (a reverse
        seek lands on the ancestor when it exists, else just below it).
        """
        ancestor = key[: self.max_depth]
        if self.reverse:
            return codec.encode_key(ancestor)
        return codec.upper_bound_of_prefix(ancestor)


StorageContextType: TypeAlias = "SnapshotProtocol | WriteBatchProtocol | TransactionProtocol"


__all__ = [
    "StorageContextType",
    "StorageScanOptions",
]
