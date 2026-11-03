"""SubscriptionRegistry.match equals brute-force Filter.matches (indexed wildcards included)."""

from __future__ import annotations

import random

import pytest

from virtuals.tkv.filter import (
    WILDCARD,
    And,
    LengthFilter,
    Or,
    PassAll,
    PrefixFilter,
    SuffixFilter,
    WildcardFilter,
    filter_hash,
)
from virtuals.tkv.observer import Subscription, SubscriptionOptions, SubscriptionRegistry


SEGS = ["a", "b", "*", 0, 1, True, -3]


def _key(r: random.Random) -> tuple:
    return tuple(r.choice(SEGS) for _ in range(r.randint(0, 4)))


def _filter(r: random.Random, depth: int = 0):
    kind = r.choice(["p", "s", "l", "w", "w", "w", "and", "or", "all"] if depth < 2 else "pslww")
    if kind == "p":
        return PrefixFilter(prefix=_key(r))
    if kind == "s":
        return SuffixFilter(suffix=_key(r))
    if kind == "l":
        return LengthFilter(length=r.randint(0, 4))
    if kind == "w":
        n = r.randint(1, 4)
        return WildcardFilter(pattern=tuple(r.choice([WILDCARD, *SEGS]) for _ in range(n)))
    if kind == "all":
        return PassAll()
    cs = tuple(_filter(r, depth + 1) for _ in range(r.randint(1, 3)))
    return And(filters=cs) if kind == "and" else Or(filters=cs)


@pytest.mark.parametrize("seed", range(5))
def test_match_equals_brute_force(seed):
    r = random.Random(seed)  # noqa: S311
    reg = SubscriptionRegistry()
    subs = [
        Subscription(_options=SubscriptionOptions(filter=_filter(r)), _observer=None)
        for _ in range(120)
    ]
    for s in subs:
        reg.add(s)
    keys = [_key(r) for _ in range(300)]

    def check(live):
        for k in keys:
            got = set(reg.match(k))
            exp = {s for s in live if s.filter.matches(k)}
            assert got == exp, (k, [s.filter for s in got ^ exp])

    check(subs)
    gone = subs[::2]
    for s in gone:
        reg.remove(s)
    check(subs[1::2])
    for s in subs[1::2]:
        reg.remove(s)
    assert len(reg) == 0
    assert reg._wildcard_index == {}
    assert reg._wildcard_sigs == {}


def test_subscription_filter_hash_cached():
    f = Or(filters=(PrefixFilter(prefix=("a",)), LengthFilter(length=2)))
    s = Subscription(_options=SubscriptionOptions(filter=f), _observer=None)
    assert s.filter_hash == filter_hash(f)
    assert s.filter_hash is s.filter_hash
