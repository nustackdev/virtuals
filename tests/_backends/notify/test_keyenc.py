"""Tests for the notify key / hint encoding."""

from __future__ import annotations

import random
from enum import IntEnum

import pytest

from virtuals._backends.notify.keyenc import enc_key, enc_seg, hint_entries
from virtuals.tkv.filter import ALL, RoutingHint


class Color(IntEnum):
    RED = 1


class Name(str):
    pass


# =============================================================================
# Segments
# =============================================================================


def test_int_vs_str_distinct() -> None:
    assert enc_seg(1) == b"i1;"
    assert enc_seg("1") == b"s1:1"
    assert enc_seg(1) != enc_seg("1")


def test_bool_encodes_as_int_like_eq() -> None:
    assert enc_seg(True) == enc_seg(1)
    assert enc_seg(False) == enc_seg(0)


def test_int_and_str_subclasses() -> None:
    assert enc_seg(Color.RED) == enc_seg(1)
    assert enc_seg(Name("ab")) == enc_seg("ab")


@pytest.mark.parametrize("v", [-1, 0, 2**53, 2**53 + 1, 2**64 + 1, -(2**63), -(2**100)])
def test_big_and_negative_ints_exact(v: int) -> None:
    assert enc_seg(v) == b"i" + str(v).encode() + b";"
    assert enc_seg(v) != enc_seg(v + 1)


def test_unicode_length_is_utf8_bytes() -> None:
    assert enc_seg("ü") == b"s2:\xc3\xbc"
    assert enc_seg("日本") == b"s6:" + "日本".encode()
    assert enc_seg("") == b"s0:"


def test_lone_surrogate_encodes() -> None:
    assert enc_seg("\ud800") == b"s3:\xed\xa0\x80"


@pytest.mark.parametrize("v", [1.0, None, b"x", (1,)])
def test_other_types_raise(v: object) -> None:
    with pytest.raises(TypeError):
        enc_seg(v)


def test_concat_is_prefix_free() -> None:
    # Without length prefixes ("a","bc") and ("ab","c") would collide.
    assert b"".join(enc_key(("a", "bc"))) != b"".join(enc_key(("ab", "c")))
    assert b"".join(enc_key(("a", "sb"))) != b"".join(enc_key(("as", "b")))
    assert b"".join(enc_key((1, 2))) != b"".join(enc_key((12,)))
    assert b"".join(enc_key(("s1:a",))) != b"".join(enc_key(("a",)))


# =============================================================================
# Hint entries
# =============================================================================


def test_hint_entries_per_kind() -> None:
    hints = (
        RoutingHint("prefix", segments=("a", 1)),
        RoutingHint("suffix", segments=("x",)),
        RoutingHint("length", length=3),
        RoutingHint("wildcard", segments=("a", -2), length=4, positions=(0, 2)),
    )
    assert hint_entries(hints) == [
        (b"p", b"2", b"s1:ai1;"),
        (b"s", b"1", b"s1:x"),
        (b"l", b"3", b""),
        (b"w", b"4:0,2", b"s1:ai-2;"),
    ]


def test_hint_entries_all_and_empty() -> None:
    assert hint_entries(ALL) == [(b"a", b"", b"")]
    assert hint_entries((RoutingHint("prefix", segments=("a",)), *ALL)) == [(b"a", b"", b"")]
    assert hint_entries(()) == []


def test_hint_entries_dedup() -> None:
    h = RoutingHint("prefix", segments=(1,))
    assert hint_entries((h, RoutingHint("prefix", segments=(True,)), h)) == [(b"p", b"1", b"i1;")]


def test_unencodable_hint_falls_back_to_all() -> None:
    hints = (RoutingHint("prefix", segments=("a",)), RoutingHint("prefix", segments=(1.5,)))
    assert hint_entries(hints) == [(b"a", b"", b"")]


# =============================================================================
# Fuzz: encoded covers == RoutingHint.covers
# =============================================================================


def _enc_covers(entry: tuple[bytes, bytes, bytes], segs: list[bytes]) -> bool:
    """What the Lua router does: compare opaque byte strings only."""
    kind, field, enc = entry
    n = len(segs)
    if kind == b"a":
        return True
    if kind == b"l":
        return n == int(field)
    if kind == b"p":
        d = int(field)
        return n >= d and b"".join(segs[:d]) == enc
    if kind == b"s":
        d = int(field)
        return n >= d and b"".join(segs[n - d :]) == enc
    length, _, pos = field.partition(b":")
    positions = [int(p) for p in pos.split(b",")]
    return n == int(length) and b"".join(segs[p] for p in positions) == enc


SEGS: list = ["a", "ab", "b", "", "as", "sb", "s", "1", "-1", "i1;", "s1:a", "ü", "日本", 1, -1, 12, True, False, 0, 2**53 + 1, 2**64]  # fmt: skip


def _seg(r: random.Random):
    return r.choice(SEGS)


def _hint(r: random.Random) -> RoutingHint:
    kind = r.choice(["prefix", "suffix", "length", "wildcard", "all"])
    if kind in ("prefix", "suffix"):
        return RoutingHint(kind, segments=tuple(_seg(r) for _ in range(r.randint(1, 3))))
    if kind == "length":
        return RoutingHint("length", length=r.randint(0, 4))
    if kind == "wildcard":
        n = r.randint(1, 4)
        positions = tuple(sorted(r.sample(range(n), r.randint(1, n))))
        return RoutingHint(
            "wildcard", segments=tuple(_seg(r) for _ in positions), length=n, positions=positions
        )
    return RoutingHint("all")


@pytest.mark.parametrize("seed", range(4))
def test_fuzz_encoding_preserves_covers(seed: int) -> None:
    r = random.Random(seed)  # noqa: S311
    covered = 0
    for _ in range(1000):
        h = _hint(r)
        (entry,) = hint_entries((h,))
        for _ in range(20):
            k = tuple(_seg(r) for _ in range(r.randint(0, 5)))
            want = h.covers(k)
            covered += want
            assert _enc_covers(entry, enc_key(k)) == want, (h, k)
    assert covered > 2000
