"""The chunked sorted index behind the in-memory storage, checked against sorted()."""

from __future__ import annotations

import random
from bisect import bisect_left, bisect_right

import pytest

from virtuals._backends.storages.mem import sorted_keys
from virtuals._backends.storages.mem.sorted_keys import SortedKeys


def _reference_page(ref: list, bound, *, inclusive: bool, reverse: bool, n: int, stop=None):
    if not reverse:
        if bound is None:
            lo = 0
        else:
            lo = bisect_left(ref, bound) if inclusive else bisect_right(ref, bound)
        hi = len(ref) if stop is None else bisect_left(ref, stop)
        return ref[lo : min(hi, lo + n)]
    if bound is None:
        hi = len(ref)
    else:
        hi = bisect_right(ref, bound) if inclusive else bisect_left(ref, bound)
    lo = 0 if stop is None else bisect_left(ref, stop)
    return ref[max(lo, hi - n) : hi][::-1]


@pytest.fixture(autouse=True)
def small_load(monkeypatch):
    """Tiny sublists so splits and merges happen within a few hundred keys."""
    monkeypatch.setattr(sorted_keys, "_LOAD", 8)


def test_empty():
    sk: SortedKeys[bytes] = SortedKeys()
    assert len(sk) == 0
    assert sk.first() is None and sk.last() is None
    assert sk.page(None, inclusive=True, reverse=False, n=5) == []
    assert sk.page(b"x", inclusive=True, reverse=True, n=5) == []
    assert not sk.discard(b"x")
    assert b"x" not in sk


def test_bulk_load_matches_sorted():
    rng = random.Random(1)  # noqa: S311
    keys = {rng.randbytes(3) for _ in range(500)}
    sk = SortedKeys(keys)
    assert list(sk) == sorted(keys)
    assert len(sk) == len(keys)
    assert sk.first() == min(keys) and sk.last() == max(keys)


@pytest.mark.parametrize("seed", range(5))
def test_random_adds_and_deletes(seed):
    rng = random.Random(seed)  # noqa: S311
    sk: SortedKeys[bytes] = SortedKeys()
    present: set[bytes] = set()
    for _ in range(600):
        key = rng.randbytes(2)
        if key in present and rng.random() < 0.6:
            assert sk.discard(key)
            present.discard(key)
        elif key not in present:
            sk.add(key)
            present.add(key)
        else:
            assert key in sk
    ref = sorted(present)
    assert list(sk) == ref
    assert len(sk) == len(ref)
    assert sk.first() == (ref[0] if ref else None)
    assert sk.last() == (ref[-1] if ref else None)
    for key in ref[:50]:
        assert key in sk
    assert all(len(sub) <= 2 * sorted_keys._LOAD for sub in sk._lists)
    assert sk._maxes == [sub[-1] for sub in sk._lists]


@pytest.mark.parametrize("seed", range(5))
def test_page_matches_reference(seed):
    rng = random.Random(100 + seed)  # noqa: S311
    present = {rng.randbytes(2) for _ in range(300)}
    sk = SortedKeys(present)
    ref = sorted(present)
    candidates = [*ref, b"", b"\xff\xff\xff", *(rng.randbytes(2) for _ in range(40))]
    for _ in range(400):
        bound = rng.choice([None, *candidates])
        stop = rng.choice([None, *candidates])
        kw = {
            "inclusive": rng.random() < 0.5,
            "reverse": rng.random() < 0.5,
            "n": rng.choice([1, 2, 5, 17, 1000]),
        }
        assert sk.page(bound, stop=stop, **kw) == _reference_page(ref, bound, stop=stop, **kw)


def test_paging_by_reseek_covers_everything_both_ways():
    rng = random.Random(7)  # noqa: S311
    present = {rng.randbytes(2) for _ in range(250)}
    sk = SortedKeys(present)
    for reverse in (False, True):
        seen: list[bytes] = []
        bound = None
        while True:
            page = sk.page(bound, inclusive=bound is None, reverse=reverse, n=7)
            seen.extend(page)
            if len(page) < 7:
                break
            bound = page[-1]
        assert seen == sorted(present, reverse=reverse)


def test_deletes_fold_thin_sublists():
    sk = SortedKeys(range(200))
    assert len(sk._lists) > 1
    for i in range(190):
        assert sk.discard(i)
    assert list(sk) == list(range(190, 200))
    assert len(sk._lists) <= 2


def test_string_keys():
    sk = SortedKeys(["b.", "a.", "a.[x]."])
    assert list(sk) == ["a.", "a.[x].", "b."]
    assert sk.page("a.", inclusive=True, reverse=False, n=10, stop="a.~") == ["a.", "a.[x]."]
