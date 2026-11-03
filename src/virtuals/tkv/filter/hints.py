"""Routing hints: coarse, server-indexable supersets of a filter.

A filter's `routing_hints()` returns a tuple of `RoutingHint`s such that every
key the filter matches is covered by at least one hint. Hints may over-cover
(the exact `Filter.matches` runs afterwards), but must never under-cover.

- `()` means "matches nothing".
- `ALL` (a single `all` hint) is the safe default for unknown filters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal


if TYPE_CHECKING:
    from ..types import Key


__all__ = [
    "ALL",
    "RoutingHint",
    "and_hints",
    "or_hints",
]


HintKind = Literal["prefix", "suffix", "length", "wildcard", "all"]


@dataclass(frozen=True, slots=True)
class RoutingHint:
    """One coarse routing entry.

    - prefix: key starts with `segments`.
    - suffix: key ends with `segments`.
    - length: key has exactly `length` segments.
    - wildcard: key has `length` segments and `key[positions[i]] == segments[i]`.
    - all: every key.
    """

    kind: HintKind
    segments: Key = ()
    """Prefix / suffix segments; wildcard: fixed segments in position order."""
    length: int = 0
    """Length hint; wildcard key length."""
    positions: tuple[int, ...] = ()
    """Wildcard: 0-based fixed positions, ascending."""

    def covers(self, key: Key) -> bool:
        """Reference semantics (per segment `==`, same as `Filter.matches`)."""
        if self.kind == "all":
            return True
        if self.kind == "length":
            return len(key) == self.length
        if self.kind == "prefix":
            d = len(self.segments)
            return len(key) >= d and key[:d] == self.segments
        if self.kind == "suffix":
            d = len(self.segments)
            return len(key) >= d and key[len(key) - d :] == self.segments
        if self.kind == "wildcard":
            return len(key) == self.length and all(
                key[p] == s for p, s in zip(self.positions, self.segments, strict=True)
            )
        raise ValueError(f"Unknown hint kind: {self.kind!r}")

    def rank(self) -> tuple[int, int]:
        """Selectivity rank, lower is more selective (used by `And`)."""
        if self.kind == "prefix":
            return (0, -len(self.segments))
        if self.kind == "wildcard":
            return (1, -len(self.segments))
        if self.kind == "suffix":
            return (2, -len(self.segments))
        if self.kind == "length":
            return (3, 0)
        return (9, 0)


ALL: tuple[RoutingHint, ...] = (RoutingHint("all"),)
"""Covers every key. Safe default."""


def _score(hints: tuple[RoutingHint, ...]) -> tuple[tuple[int, int], int]:
    return (max(h.rank() for h in hints), len(hints))


def and_hints(children: list[tuple[RoutingHint, ...]]) -> tuple[RoutingHint, ...]:
    """Hints for an And: the most selective child (any child is a valid superset)."""
    if not children:
        return ALL
    if any(not c for c in children):
        return ()
    return min(children, key=_score)  # min keeps the first on ties


def or_hints(children: list[tuple[RoutingHint, ...]]) -> tuple[RoutingHint, ...]:
    """Hints for an Or: the union of the children's hints, deduped in order."""
    out: list[RoutingHint] = []
    for c in children:
        for h in c:
            if h.kind == "all":
                return ALL
            if h not in out:
                out.append(h)
    return tuple(out)
