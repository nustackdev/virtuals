"""Tests for Filter.routing_hints() and RoutingHint.covers()."""

from __future__ import annotations

import random
from dataclasses import dataclass

import pytest

from virtuals.tkv.filter import (
    ALL,
    WILDCARD,
    And,
    Filter,
    LengthFilter,
    Or,
    PassAll,
    PassNone,
    PrefixFilter,
    RoutingHint,
    SuffixFilter,
    WildcardFilter,
    canonicalize,
)


@dataclass(frozen=True, slots=True)
class EvenLength(Filter):
    """Custom filter with no routing_hints override."""

    def matches(self, key) -> bool:
        return len(key) % 2 == 0

    def __hash__(self) -> int:
        return hash(("even",))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, EvenLength)

    def to_dict(self) -> dict:
        return {"type": "even"}


P = PrefixFilter
S = SuffixFilter
L = LengthFilter
W = WildcardFilter


# =============================================================================
# Per type
# =============================================================================


@pytest.mark.parametrize(
    ("f", "hints"),
    [
        (P(()), ALL),
        (P(("a", 1)), (RoutingHint("prefix", segments=("a", 1)),)),
        (S(()), ALL),
        (S(("x",)), (RoutingHint("suffix", segments=("x",)),)),
        (L(0), (RoutingHint("length", length=0),)),
        (L(3), (RoutingHint("length", length=3),)),
        (L(-1), ()),
        (W(()), (RoutingHint("length", length=0),)),
        (W((WILDCARD, WILDCARD)), (RoutingHint("length", length=2),)),
        (
            W(("a", WILDCARD, 7)),
            (RoutingHint("wildcard", segments=("a", 7), length=3, positions=(0, 2)),),
        ),
        (PassAll(), ALL),
        (PassNone(), ()),
        (EvenLength(), ALL),
    ],
)
def test_leaf_hints(f: Filter, hints: tuple[RoutingHint, ...]) -> None:
    assert f.routing_hints() == hints


def test_and_picks_most_selective() -> None:
    f = And((L(3), S(("x",)), P(("a",)), W(("a", WILDCARD, "b"))))
    assert f.routing_hints() == (RoutingHint("prefix", segments=("a",)),)


def test_and_deeper_prefix_wins() -> None:
    f = And((P(("a",)), P(("a", "b"))))
    assert f.routing_hints() == (RoutingHint("prefix", segments=("a", "b")),)


def test_and_ranks_wildcard_over_suffix_over_length() -> None:
    w = W(("a", WILDCARD))
    assert And((L(2), S(("x",)), w)).routing_hints() == w.routing_hints()
    assert And((L(2), S(("x",)))).routing_hints() == S(("x",)).routing_hints()
    assert And((EvenLength(), L(2))).routing_hints() == L(2).routing_hints()


def test_and_prefers_fewer_entries_on_tie() -> None:
    two = Or((P(("a",)), P(("b",))))
    one = P(("c",))
    assert And((two, one)).routing_hints() == one.routing_hints()


def test_and_tie_keeps_first() -> None:
    assert And((P(("a",)), P(("b",)))).routing_hints() == P(("a",)).routing_hints()


def test_and_any_empty_child_is_empty() -> None:
    assert And((P(("a",)), PassNone())).routing_hints() == ()
    assert And((P(("a",)), L(-1))).routing_hints() == ()


def test_and_empty_is_all() -> None:
    assert And(()).routing_hints() == ALL


def test_or_union_dedup_in_order() -> None:
    f = Or((P(("a",)), L(2), P(("a",)), PassNone()))
    assert f.routing_hints() == (
        RoutingHint("prefix", segments=("a",)),
        RoutingHint("length", length=2),
    )


def test_or_with_all_is_all() -> None:
    assert Or((P(("a",)), EvenLength())).routing_hints() == ALL
    assert Or((P(("a",)), P(()))).routing_hints() == ALL


def test_or_empty_is_empty() -> None:
    assert Or(()).routing_hints() == ()


def test_canonical_order_gives_one_hint_set() -> None:
    a, b = P(("a",)), P(("b",))
    assert canonicalize(a & b).routing_hints() == canonicalize(b & a).routing_hints()
    assert canonicalize(a | b).routing_hints() == canonicalize(b | a).routing_hints()


# =============================================================================
# covers
# =============================================================================


def test_covers_semantics() -> None:
    assert RoutingHint("all").covers(())
    assert RoutingHint("length", length=0).covers(())
    assert not RoutingHint("length", length=1).covers(())
    assert RoutingHint("prefix", segments=("a",)).covers(("a", 1))
    assert not RoutingHint("prefix", segments=("a", 1)).covers(("a",))
    assert RoutingHint("suffix", segments=(1,)).covers(("a", 1))
    assert not RoutingHint("suffix", segments=("a", 1)).covers((1,))
    w = RoutingHint("wildcard", segments=("a",), length=2, positions=(1,))
    assert w.covers((0, "a"))
    assert not w.covers(("a", 0))
    assert not w.covers((0, "a", 0))


def test_covers_bool_int_like_matches() -> None:
    # Filter.matches uses ==, so True matches 1; hints must agree.
    assert P((1,)).matches((True,))
    assert RoutingHint("prefix", segments=(1,)).covers((True,))
    assert not RoutingHint("prefix", segments=("1",)).covers((1,))


# =============================================================================
# Fuzz: matches(k) implies covers
# =============================================================================


SEGS: list = [
    "a", "b", "", "*", "1", "ü", "日本", "a" * 40,
    0, 1, -1, 2, True, False, 2**53, 2**53 + 1, 2**60, -(2**63), 2**64 + 1,
]  # fmt: skip


HOT: list = ["a", 1, True, -1, 2**60, "ü"]  # dense subset so filters hit often


def _seg(r: random.Random):
    return r.choice(HOT if r.random() < 0.7 else SEGS)


def _key(r: random.Random) -> tuple:
    return tuple(_seg(r) for _ in range(r.randint(0, 5)))


def _filter(r: random.Random, depth: int = 0) -> Filter:
    roll = r.random()
    if depth < 3 and roll < 0.3:
        kids = tuple(_filter(r, depth + 1) for _ in range(r.randint(0, 4)))
        return And(kids) if r.random() < 0.5 else Or(kids)
    leaf = r.randrange(8)
    if leaf == 0:
        return P(tuple(_seg(r) for _ in range(r.randint(0, 3))))
    if leaf == 1:
        return S(tuple(_seg(r) for _ in range(r.randint(0, 3))))
    if leaf == 2:
        return L(r.randint(-1, 5))
    if leaf == 3:
        return W(tuple(WILDCARD if r.random() < 0.5 else _seg(r) for _ in range(r.randint(0, 4))))
    if leaf == 4:
        return PassAll()
    if leaf == 5:
        return PassNone()
    if leaf == 6:
        return EvenLength()
    # biased toward keys that will match: a prefix of a real key
    k = _key(r)
    return P(k[: r.randint(0, len(k))])


def _leaves(f: Filter) -> list[Filter]:
    if isinstance(f, (And, Or)):
        return [x for c in f.filters for x in _leaves(c)]
    return [f]


def _near_key(r: random.Random, f: Filter) -> tuple:
    """A key satisfying one random leaf of f, so And/Or filters often match."""
    leaf = r.choice(_leaves(f) or [PassAll()])
    pad = tuple(_seg(r) for _ in range(r.randint(0, 2)))
    if isinstance(leaf, P):
        return leaf.prefix + pad
    if isinstance(leaf, S):
        return pad + leaf.suffix
    if isinstance(leaf, L):
        return tuple(_seg(r) for _ in range(max(leaf.length, 0)))
    if isinstance(leaf, W):
        return tuple(_seg(r) if p == WILDCARD else p for p in leaf.pattern)
    return _key(r)


@pytest.mark.parametrize("seed", range(4))
def test_fuzz_matches_implies_covers(seed: int) -> None:
    r = random.Random(seed)  # noqa: S311
    matched = selective = 0
    for _ in range(1500):
        f = _filter(r)
        variants = (f, canonicalize(f))
        hints = [v.routing_hints() for v in variants]
        for i in range(20):
            k = _key(r) if i % 2 else _near_key(r, f)
            if not f.matches(k):
                continue
            matched += 1
            selective += hints[1] != ALL
            for v, hs in zip(variants, hints, strict=True):
                assert v.matches(k)
                assert any(h.covers(k) for h in hs), (v, hs, k)
    assert matched > 5000
    assert selective > 5000  # enough matches go through non-trivial hints
