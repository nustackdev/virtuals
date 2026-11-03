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
from .hints import ALL, RoutingHint


__all__ = [
    "ALL",
    "WILDCARD",
    "And",
    "Filter",
    "LengthFilter",
    "Or",
    "PassAll",
    "PassNone",
    "PrefixFilter",
    "RoutingHint",
    "SuffixFilter",
    "WildcardFilter",
    "canonicalize",
    "filter_from_dict",
    "filter_hash",
]
