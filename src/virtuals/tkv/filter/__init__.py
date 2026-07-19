"""Unified filter system for storage operations.

Provides composable, hashable filters for key matching.
Used by both scan operations and observer subscriptions.
"""

from __future__ import annotations

from .filter import (
    WILDCARD,
    And,
    Filter,
    LengthFilter,
    Or,
    PassAll,
    PassNone,
    PrefixFilter,
    SuffixFilter,
    WildcardFilter,
    canonicalize,
    filter_from_dict,
    filter_hash,
)


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
