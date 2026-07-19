"""Tests for filter serialization, canonicalization, and cross-process hashing.

Covers:
- `to_dict()` / `filter_from_dict()` round-trip for every built-in filter type.
- `canonicalize()` algebraic simplifications: leaf folding, identity/annihilator
  absorption for And/Or, flatten nested composites, dedup + sort constituents.
- `filter_hash()` stability across constituent order and across equivalent
  spellings of the same logic.
"""

from __future__ import annotations

import pytest

from virtuals.tkv.filter import (
    WILDCARD,
    And,
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


# =============================================================================
# to_dict / filter_from_dict round-trip
# =============================================================================


@pytest.mark.parametrize(
    "f",
    [
        PrefixFilter(prefix=("users",)),
        PrefixFilter(prefix=("users", "alice", "profile")),
        PrefixFilter(prefix=()),
        SuffixFilter(suffix=("profile",)),
        SuffixFilter(suffix=()),
        LengthFilter(length=3),
        LengthFilter(length=0),
        WildcardFilter(pattern=("users", WILDCARD, "profile")),
        WildcardFilter(pattern=(WILDCARD, WILDCARD, WILDCARD)),
        PassAll(),
        PassNone(),
    ],
)
def test_leaf_roundtrip(f):
    """Every built-in leaf filter round-trips through to_dict/from_dict."""
    assert filter_from_dict(f.to_dict()) == f


def test_and_roundtrip():
    f = And(filters=(PrefixFilter(prefix=("users",)), LengthFilter(length=3)))
    assert filter_from_dict(f.to_dict()) == f


def test_or_roundtrip():
    f = Or(filters=(PrefixFilter(prefix=("users",)), PrefixFilter(prefix=("posts",))))
    assert filter_from_dict(f.to_dict()) == f


def test_nested_composite_roundtrip():
    f = And(
        filters=(
            Or(filters=(PrefixFilter(prefix=("a",)), PrefixFilter(prefix=("b",)))),
            LengthFilter(length=2),
        )
    )
    assert filter_from_dict(f.to_dict()) == f


def test_from_dict_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown filter type"):
        filter_from_dict({"type": "no_such_filter"})


# =============================================================================
# Canonicalization: leaf folding
# =============================================================================


def test_empty_prefix_canonicalizes_to_pass_all():
    assert canonicalize(PrefixFilter(prefix=())) == PassAll()


def test_empty_suffix_canonicalizes_to_pass_all():
    assert canonicalize(SuffixFilter(suffix=())) == PassAll()


def test_all_wildcard_pattern_canonicalizes_to_length_filter():
    assert canonicalize(WildcardFilter(pattern=(WILDCARD, WILDCARD, WILDCARD))) == LengthFilter(
        length=3
    )


def test_non_empty_leaves_unchanged():
    p = PrefixFilter(prefix=("users",))
    assert canonicalize(p) == p
    lf = LengthFilter(length=5)
    assert canonicalize(lf) == lf


# =============================================================================
# Canonicalization: And rules
# =============================================================================


def test_and_drops_pass_all():
    f = And(filters=(PassAll(), PrefixFilter(prefix=("users",))))
    assert canonicalize(f) == PrefixFilter(prefix=("users",))


def test_and_absorbs_pass_none():
    f = And(filters=(PrefixFilter(prefix=("users",)), PassNone()))
    assert canonicalize(f) == PassNone()


def test_and_single_element_collapses():
    f = And(filters=(PrefixFilter(prefix=("users",)),))
    assert canonicalize(f) == PrefixFilter(prefix=("users",))


def test_and_empty_becomes_pass_all():
    f = And(filters=())
    assert canonicalize(f) == PassAll()


def test_and_flattens_nested():
    inner = And(filters=(PrefixFilter(prefix=("a",)), LengthFilter(length=2)))
    outer = And(filters=(inner, SuffixFilter(suffix=("b",))))
    result = canonicalize(outer)
    assert isinstance(result, And)
    # Three constituents after flatten
    assert len(result.filters) == 3


def test_and_dedups_identical_constituents():
    f = And(filters=(PrefixFilter(prefix=("users",)), PrefixFilter(prefix=("users",))))
    assert canonicalize(f) == PrefixFilter(prefix=("users",))


def test_and_orders_constituents_stably():
    a = PrefixFilter(prefix=("users",))
    b = LengthFilter(length=3)
    ab = canonicalize(And(filters=(a, b)))
    ba = canonicalize(And(filters=(b, a)))
    assert ab == ba


# =============================================================================
# Canonicalization: Or rules
# =============================================================================


def test_or_drops_pass_none():
    f = Or(filters=(PassNone(), PrefixFilter(prefix=("users",))))
    assert canonicalize(f) == PrefixFilter(prefix=("users",))


def test_or_absorbs_pass_all():
    f = Or(filters=(PrefixFilter(prefix=("users",)), PassAll()))
    assert canonicalize(f) == PassAll()


def test_or_single_element_collapses():
    f = Or(filters=(PrefixFilter(prefix=("users",)),))
    assert canonicalize(f) == PrefixFilter(prefix=("users",))


def test_or_empty_becomes_pass_none():
    f = Or(filters=())
    assert canonicalize(f) == PassNone()


def test_or_flattens_nested():
    inner = Or(filters=(PrefixFilter(prefix=("a",)), PrefixFilter(prefix=("b",))))
    outer = Or(filters=(inner, PrefixFilter(prefix=("c",))))
    result = canonicalize(outer)
    assert isinstance(result, Or)
    assert len(result.filters) == 3


def test_or_dedups_identical_constituents():
    f = Or(filters=(PrefixFilter(prefix=("users",)), PrefixFilter(prefix=("users",))))
    assert canonicalize(f) == PrefixFilter(prefix=("users",))


def test_or_orders_constituents_stably():
    a = PrefixFilter(prefix=("users",))
    b = PrefixFilter(prefix=("posts",))
    ab = canonicalize(Or(filters=(a, b)))
    ba = canonicalize(Or(filters=(b, a)))
    assert ab == ba


# =============================================================================
# filter_hash: cross-process stability
# =============================================================================


def test_hash_stable_across_constituent_order_and():
    a = PrefixFilter(prefix=("users",))
    b = LengthFilter(length=3)
    assert filter_hash(a & b) == filter_hash(b & a)


def test_hash_stable_across_constituent_order_or():
    a = PrefixFilter(prefix=("users",))
    b = PrefixFilter(prefix=("posts",))
    assert filter_hash(a | b) == filter_hash(b | a)


def test_hash_equivalent_spellings_of_pass_all():
    assert filter_hash(PassAll()) == filter_hash(PrefixFilter(prefix=()))
    assert filter_hash(PassAll()) == filter_hash(SuffixFilter(suffix=()))


def test_hash_all_wildcard_equals_length_filter():
    f = WildcardFilter(pattern=(WILDCARD, WILDCARD, WILDCARD))
    assert filter_hash(f) == filter_hash(LengthFilter(length=3))


def test_hash_distinct_filters_distinct_hashes():
    a = PrefixFilter(prefix=("users",))
    b = PrefixFilter(prefix=("posts",))
    assert filter_hash(a) != filter_hash(b)


def test_hash_is_hex_sha256():
    h = filter_hash(PrefixFilter(prefix=("users",)))
    assert isinstance(h, str)
    assert len(h) == 64
    int(h, 16)  # parses as hex


def test_hash_deterministic():
    """Same filter, called twice, same hash. Baseline stability check."""
    f = And(filters=(PrefixFilter(prefix=("users",)), LengthFilter(length=3)))
    assert filter_hash(f) == filter_hash(f)


def test_hash_and_absorbing_pass_none_matches_pass_none():
    f = And(filters=(PrefixFilter(prefix=("users",)), PassNone()))
    assert filter_hash(f) == filter_hash(PassNone())


def test_hash_or_absorbing_pass_all_matches_pass_all():
    f = Or(filters=(PrefixFilter(prefix=("users",)), PassAll()))
    assert filter_hash(f) == filter_hash(PassAll())


# =============================================================================
# Post-canonicalize semantics preserved (matches() still gives same answers)
# =============================================================================


@pytest.mark.parametrize(
    "f,key,expected",
    [
        (And(filters=(PrefixFilter(prefix=("users",)), LengthFilter(length=3))), ("users", "a", "b"), True),
        (And(filters=(PrefixFilter(prefix=("users",)), LengthFilter(length=3))), ("users", "a"), False),
        (Or(filters=(PrefixFilter(prefix=("a",)), PrefixFilter(prefix=("b",)))), ("a", "x"), True),
        (Or(filters=(PrefixFilter(prefix=("a",)), PrefixFilter(prefix=("b",)))), ("c",), False),
        (PrefixFilter(prefix=()), ("anything",), True),  # canonicalizes to PassAll
        (WildcardFilter(pattern=(WILDCARD, WILDCARD)), ("a", "b"), True),  # canonicalizes to LengthFilter(2)
        (WildcardFilter(pattern=(WILDCARD, WILDCARD)), ("a",), False),
    ],
)
def test_canonicalize_preserves_matches(f, key, expected):
    """canonicalize() must be semantics-preserving on matches()."""
    assert canonicalize(f).matches(key) is expected
