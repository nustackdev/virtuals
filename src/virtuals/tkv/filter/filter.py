"""Unified filter system for storage operations.

Provides composable, hashable filters for key matching.
Used by both scan operations and observer subscriptions.

Also provides:
- `Filter.to_dict()`  -- literal dict representation of a filter (for wire transport).
- `filter_from_dict(d)`  -- reconstruct any filter from its dict.
- `canonicalize(f)`  -- return an equivalent filter with algebraic simplifications
  applied bottom-up (flatten nested And/Or, absorb PassAll/PassNone identities,
  collapse PrefixFilter(()) / SuffixFilter(()) / all-wildcard patterns, dedup
  and sort constituents so `a & b` and `b & a` normalize to the same filter).
- `filter_hash(f)`  -- SHA-256 of the canonical JSON of `f`. Stable across
  processes (unlike `hash(f)`, which relies on PYTHONHASHSEED-salted `hash(tuple)`).
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ..types import Key


__all__ = [
    "WILDCARD",
    "And",
    "Filter",
    "LengthFilter",
    "Or",
    "PassAll",
    "PassNone",
    "PrefixFilter",
    "SuffixFilter",
    "WildcardFilter",
    "canonicalize",
    "filter_from_dict",
    "filter_hash",
]


WILDCARD = "*"
"""Wildcard segment matching any single key component."""


class Filter(ABC):
    """Base filter for key matching. Hashable and composable.

    Filters determine which keys match certain criteria. They are:
    - Immutable and hashable (for use in sets/dicts)
    - Composable via & (and) and | (or) operators
    - Short-circuit evaluated for efficiency
    - Serializable via `to_dict()` / `filter_from_dict()` for cross-process transport.

    Examples:
        >>> # Match keys starting with ("users",)
        >>> f = PrefixFilter(prefix=("users",))
        >>> f.matches(("users", "alice"))  # True

        >>> # Compose filters
        >>> f = PrefixFilter(prefix=("users",)) & LengthFilter(length=3)
        >>> f.matches(("users", "alice", "profile"))  # True
        >>> f.matches(("users", "alice"))  # False (length mismatch)

        >>> # Or composition
        >>> f = PrefixFilter(prefix=("users",)) | PrefixFilter(prefix=("posts",))
        >>> f.matches(("users", "alice"))  # True
        >>> f.matches(("posts", "123"))  # True
    """

    @abstractmethod
    def matches(self, key: Key) -> bool:
        """Check if key matches this filter.

        Args:
            key: Key tuple to check.

        Returns:
            True if key matches filter criteria.
        """
        ...

    @abstractmethod
    def __hash__(self) -> int:
        """Return hash for use in sets and dicts."""
        ...

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """Check equality with another filter."""
        ...

    def to_dict(self) -> dict:
        """Return a literal dict representation of this filter.

        Built-in filters (Prefix/Suffix/Length/Wildcard/And/Or/PassAll/PassNone)
        override this. Custom user filter subclasses that need cross-process
        transport (`filter_hash`, redis observer, etc.) must override this;
        process-local custom filters can leave the default in place.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement to_dict(). "
            "Override to_dict() to enable cross-process transport for this filter."
        )

    def __and__(self, other: Filter) -> And:
        """Combine with AND logic: self & other."""
        return And(filters=(self, other))

    def __or__(self, other: Filter) -> Or:
        """Combine with OR logic: self | other."""
        return Or(filters=(self, other))


@dataclass(frozen=True, slots=True)
class PrefixFilter(Filter):
    """Match keys starting with prefix.

    Examples:
        >>> f = PrefixFilter(prefix=("users",))
        >>> f.matches(("users", "alice"))  # True
        >>> f.matches(("users", "alice", "profile"))  # True
        >>> f.matches(("posts",))  # False
    """

    prefix: Key
    """Prefix that keys must start with."""

    def matches(self, key: Key) -> bool:
        """Check if key starts with the prefix."""
        if len(key) < len(self.prefix):
            return False
        return key[: len(self.prefix)] == self.prefix

    def __hash__(self) -> int:
        """Return hash based on prefix."""
        return hash(("prefix", self.prefix))

    def __eq__(self, other: object) -> bool:
        """Check equality."""
        if not isinstance(other, PrefixFilter):
            return NotImplemented
        return self.prefix == other.prefix

    def to_dict(self) -> dict:
        return {"type": "prefix", "prefix": list(self.prefix)}


@dataclass(frozen=True, slots=True)
class SuffixFilter(Filter):
    """Match keys ending with suffix.

    Examples:
        >>> f = SuffixFilter(suffix=("profile",))
        >>> f.matches(("users", "alice", "profile"))  # True
        >>> f.matches(("posts", "123", "profile"))  # True
        >>> f.matches(("users", "alice"))  # False
    """

    suffix: Key
    """Suffix that keys must end with."""

    def matches(self, key: Key) -> bool:
        """Check if key ends with the suffix."""
        if len(key) < len(self.suffix):
            return False
        if not self.suffix:  # Empty suffix matches everything
            return True
        return key[-len(self.suffix) :] == self.suffix

    def __hash__(self) -> int:
        """Return hash based on suffix."""
        return hash(("suffix", self.suffix))

    def __eq__(self, other: object) -> bool:
        """Check equality."""
        if not isinstance(other, SuffixFilter):
            return NotImplemented
        return self.suffix == other.suffix

    def to_dict(self) -> dict:
        return {"type": "suffix", "suffix": list(self.suffix)}


@dataclass(frozen=True, slots=True)
class LengthFilter(Filter):
    """Match keys with exact length.

    Examples:
        >>> f = LengthFilter(length=3)
        >>> f.matches(("a", "b", "c"))  # True
        >>> f.matches(("a", "b"))  # False
        >>> f.matches(("a", "b", "c", "d"))  # False
    """

    length: int
    """Exact length that keys must have."""

    def matches(self, key: Key) -> bool:
        """Check if key has the exact length."""
        return len(key) == self.length

    def __hash__(self) -> int:
        """Return hash based on length."""
        return hash(("length", self.length))

    def __eq__(self, other: object) -> bool:
        """Check equality."""
        if not isinstance(other, LengthFilter):
            return NotImplemented
        return self.length == other.length

    def to_dict(self) -> dict:
        return {"type": "length", "length": self.length}


@dataclass(frozen=True, slots=True)
class WildcardFilter(Filter):
    """Match keys with wildcard patterns.

    Use WILDCARD ('*') to match any single segment in the pattern.

    Examples:
        >>> f = WildcardFilter(pattern=("users", WILDCARD, "profile"))
        >>> f.matches(("users", "alice", "profile"))  # True
        >>> f.matches(("users", "bob", "profile"))  # True
        >>> f.matches(("users", "alice", "settings"))  # False
        >>> f.matches(("users", "alice"))  # False (length mismatch)
    """

    pattern: Key
    """Pattern with wildcards. WILDCARD matches any single segment."""

    def matches(self, key: Key) -> bool:
        """Check if key matches the wildcard pattern."""
        if len(key) != len(self.pattern):
            return False
        return all(p == WILDCARD or k == p for k, p in zip(key, self.pattern, strict=True))

    def __hash__(self) -> int:
        """Return hash based on pattern."""
        return hash(("wildcard", self.pattern))

    def __eq__(self, other: object) -> bool:
        """Check equality."""
        if not isinstance(other, WildcardFilter):
            return NotImplemented
        return self.pattern == other.pattern

    def to_dict(self) -> dict:
        return {"type": "wildcard", "pattern": list(self.pattern)}


@dataclass(frozen=True, slots=True)
class And(Filter):
    """Combine filters with AND logic. All must match.

    Short-circuits on first False for efficiency.

    Examples:
        >>> f = And(
        ...     filters=(
        ...         PrefixFilter(prefix=("users",)),
        ...         LengthFilter(length=3),
        ...     )
        ... )
        >>> f.matches(("users", "alice", "profile"))  # True
        >>> f.matches(("users", "alice"))  # False (length mismatch)
    """

    filters: tuple[Filter, ...]
    """Filters to combine with AND logic."""

    def matches(self, key: Key) -> bool:
        """Check if key matches all filters. Short-circuits on first False."""
        return all(f.matches(key) for f in self.filters)

    def __hash__(self) -> int:
        """Return hash based on all filters."""
        return hash(("and", self.filters))

    def __eq__(self, other: object) -> bool:
        """Check equality."""
        if not isinstance(other, And):
            return NotImplemented
        return self.filters == other.filters

    def __and__(self, other: Filter) -> And:
        """Flatten nested Ands: (a & b) & c -> And(a, b, c)."""
        if isinstance(other, And):
            return And(filters=self.filters + other.filters)
        return And(filters=(*self.filters, other))

    def to_dict(self) -> dict:
        return {"type": "and", "filters": [f.to_dict() for f in self.filters]}


@dataclass(frozen=True, slots=True)
class Or(Filter):
    """Combine filters with OR logic. Any must match.

    Short-circuits on first True for efficiency.

    Examples:
        >>> f = Or(
        ...     filters=(
        ...         PrefixFilter(prefix=("users",)),
        ...         PrefixFilter(prefix=("posts",)),
        ...     )
        ... )
        >>> f.matches(("users", "alice"))  # True
        >>> f.matches(("posts", "123"))  # True
        >>> f.matches(("comments",))  # False
    """

    filters: tuple[Filter, ...]
    """Filters to combine with OR logic."""

    def matches(self, key: Key) -> bool:
        """Check if key matches any filter. Short-circuits on first True."""
        return any(f.matches(key) for f in self.filters)

    def __hash__(self) -> int:
        """Return hash based on all filters."""
        return hash(("or", self.filters))

    def __eq__(self, other: object) -> bool:
        """Check equality."""
        if not isinstance(other, Or):
            return NotImplemented
        return self.filters == other.filters

    def __or__(self, other: Filter) -> Or:
        """Flatten nested Ors: (a | b) | c -> Or(a, b, c)."""
        if isinstance(other, Or):
            return Or(filters=self.filters + other.filters)
        return Or(filters=(*self.filters, other))

    def to_dict(self) -> dict:
        return {"type": "or", "filters": [f.to_dict() for f in self.filters]}


@dataclass(frozen=True, slots=True)
class PassAll(Filter):
    """Filter that always matches. Identity element for And.

    Examples:
        >>> f = PassAll()
        >>> f.matches(("any", "key"))  # True
        >>> f.matches(())  # True
    """

    def matches(self, key: Key) -> bool:
        """Always returns True."""
        return True

    def __hash__(self) -> int:
        """Return hash."""
        return hash(("passall",))

    def __eq__(self, other: object) -> bool:
        """Check equality."""
        if not isinstance(other, PassAll):
            return NotImplemented
        return True

    def to_dict(self) -> dict:
        return {"type": "pass_all"}


@dataclass(frozen=True, slots=True)
class PassNone(Filter):
    """Filter that never matches. Identity element for Or.

    Examples:
        >>> f = PassNone()
        >>> f.matches(("any", "key"))  # False
        >>> f.matches(())  # False
    """

    def matches(self, key: Key) -> bool:
        """Always returns False."""
        return False

    def __hash__(self) -> int:
        """Return hash."""
        return hash(("passnone",))

    def __eq__(self, other: object) -> bool:
        """Check equality."""
        if not isinstance(other, PassNone):
            return NotImplemented
        return True

    def to_dict(self) -> dict:
        return {"type": "pass_none"}


# ---------------------------------------------------------------------------
# Serialization / canonicalization
# ---------------------------------------------------------------------------


_LEAF_FROM_DICT = {
    "prefix": lambda d: PrefixFilter(prefix=tuple(d["prefix"])),
    "suffix": lambda d: SuffixFilter(suffix=tuple(d["suffix"])),
    "length": lambda d: LengthFilter(length=int(d["length"])),
    "wildcard": lambda d: WildcardFilter(pattern=tuple(d["pattern"])),
    "pass_all": lambda d: PassAll(),
    "pass_none": lambda d: PassNone(),
}


def filter_from_dict(d: dict) -> Filter:
    """Reconstruct a filter from its dict form (as produced by `to_dict()`).

    Accepts both raw and canonical dicts. Unknown `type` values raise
    ValueError. Custom user filters must be registered separately if they
    need cross-process transport.
    """
    t = d.get("type")
    if t in _LEAF_FROM_DICT:
        return _LEAF_FROM_DICT[t](d)
    if t == "and":
        return And(filters=tuple(filter_from_dict(c) for c in d["filters"]))
    if t == "or":
        return Or(filters=tuple(filter_from_dict(c) for c in d["filters"]))
    raise ValueError(f"Unknown filter type: {t!r}")


def _sort_key(f: Filter) -> str:
    """Stable sort key for filters: canonical JSON of their dict form."""
    return json.dumps(f.to_dict(), sort_keys=True, separators=(",", ":"))


def canonicalize(f: Filter) -> Filter:
    """Return an equivalent filter with algebraic simplifications applied bottom-up.

    Aggressive: identical logic normalizes to identical filters so cross-process
    channel dedup works.

    Rules applied recursively:
    - `PrefixFilter(())`, `SuffixFilter(())`, all-wildcard `WildcardFilter` -> `PassAll`.
      (Well, all-wildcard collapses to `LengthFilter(len(pattern))` first.)
    - Flatten nested `And`/`Or`.
    - `And`: drop `PassAll` (identity), absorb `PassNone` (annihilator).
      Empty `And` -> `PassAll`. Single-element `And(f,)` -> `f`.
    - `Or`: drop `PassNone` (identity), absorb `PassAll` (annihilator).
      Empty `Or` -> `PassNone`. Single-element `Or(f,)` -> `f`.
    - `And`/`Or` constituents dedup + sort by canonical dict form.

    Custom (unknown) filter types are returned as-is.
    """
    # Leaf simplifications
    if isinstance(f, PrefixFilter) and len(f.prefix) == 0:
        return PassAll()
    if isinstance(f, SuffixFilter) and len(f.suffix) == 0:
        return PassAll()
    if isinstance(f, WildcardFilter):
        if all(seg == WILDCARD for seg in f.pattern):
            return LengthFilter(length=len(f.pattern))
        return f

    if isinstance(f, And):
        children = [canonicalize(sub) for sub in f.filters]
        # Flatten nested And
        flat: list[Filter] = []
        for c in children:
            if isinstance(c, And):
                flat.extend(c.filters)
            else:
                flat.append(c)
        # Absorb PassNone
        if any(isinstance(c, PassNone) for c in flat):
            return PassNone()
        # Drop PassAll (identity)
        filtered = [c for c in flat if not isinstance(c, PassAll)]
        # Dedup by canonical form
        seen: set[str] = set()
        dedup: list[Filter] = []
        for c in filtered:
            k = _sort_key(c)
            if k not in seen:
                seen.add(k)
                dedup.append(c)
        # Sort so `a & b` and `b & a` canonicalize the same
        dedup.sort(key=_sort_key)
        # Collapse
        if len(dedup) == 0:
            return PassAll()
        if len(dedup) == 1:
            return dedup[0]
        return And(filters=tuple(dedup))

    if isinstance(f, Or):
        children = [canonicalize(sub) for sub in f.filters]
        flat = []
        for c in children:
            if isinstance(c, Or):
                flat.extend(c.filters)
            else:
                flat.append(c)
        if any(isinstance(c, PassAll) for c in flat):
            return PassAll()
        filtered = [c for c in flat if not isinstance(c, PassNone)]
        seen = set()
        dedup = []
        for c in filtered:
            k = _sort_key(c)
            if k not in seen:
                seen.add(k)
                dedup.append(c)
        dedup.sort(key=_sort_key)
        if len(dedup) == 0:
            return PassNone()
        if len(dedup) == 1:
            return dedup[0]
        return Or(filters=tuple(dedup))

    # Leaves without simplification (LengthFilter, PrefixFilter(non-empty), etc)
    # and any custom user filters.
    return f


def filter_hash(f: Filter) -> str:
    """SHA-256 hex digest of the canonical JSON of `f`.

    Stable across processes (unlike Python's built-in `hash`, which
    randomizes tuple hashes via PYTHONHASHSEED). Two filters with
    equivalent logic (after canonicalization) produce the same hash.
    """
    canon = canonicalize(f)
    payload = json.dumps(canon.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
