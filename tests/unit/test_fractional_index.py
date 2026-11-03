"""Tests for the fractional index key generator."""

from __future__ import annotations

import random

import pytest

from virtuals._views.fractional_index import (
    BASE_62_DIGITS,
    generate_key_between,
    generate_n_keys_between,
    generate_suffixed_key_between,
    validate_order_key,
)


# ============================================================================
# REFERENCE VECTORS (rocicorp fractional-indexing test suite)
# ============================================================================


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (None, None, "a0"),
        (None, "a0", "Zz"),
        (None, "Zz", "Zy"),
        ("a0", None, "a1"),
        ("a1", None, "a2"),
        ("a0", "a1", "a0V"),
        ("a1", "a2", "a1V"),
        ("a0V", "a1", "a0l"),
        ("Zz", "a0", "ZzV"),
        ("Zz", "a1", "a0"),
        (None, "Y00", "Xzzz"),
        ("bzz", None, "c000"),
        ("a0", "a0V", "a0G"),
        ("a0", "a0G", "a08"),
        ("b125", "b129", "b127"),
        ("a0", "a1V", "a1"),
        ("Zz", "a1V", "a0"),
        (None, "a0V", "a0"),
        (None, "b999", "b99"),
        (None, "A000000000000000000000000001", "A000000000000000000000000000V"),
        ("zzzzzzzzzzzzzzzzzzzzzzzzzzz", None, "zzzzzzzzzzzzzzzzzzzzzzzzzzzV"),
    ],
)
def test_reference_vectors(a, b, expected):
    assert generate_key_between(a, b) == expected


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (None, "A00000000000000000000000000"),
        ("a00", None),
        ("a00", "a1"),
        ("0", "1"),
        ("a1", "a0"),
        ("a1", "a1"),
        ("", "a0"),
    ],
)
def test_invalid_inputs_raise(a, b):
    with pytest.raises(ValueError):
        generate_key_between(a, b)


def test_n_keys_between():
    assert generate_n_keys_between(None, None, 5) == ["a0", "a1", "a2", "a3", "a4"]
    assert generate_n_keys_between("a4", None, 3) == ["a5", "a6", "a7"]
    assert generate_n_keys_between(None, "a0", 3) == ["Zx", "Zy", "Zz"]
    keys = generate_n_keys_between("a0", "a2", 20)
    assert len(keys) == 20
    assert keys == sorted(keys)
    assert len(set(keys)) == 20
    assert all("a0" < k < "a2" for k in keys)
    assert generate_n_keys_between(None, None, 0) == []


# ============================================================================
# INVARIANTS
# ============================================================================


def _between(k, a, b):
    validate_order_key(k)
    if a is not None:
        assert a < k, (a, k, b)
    if b is not None:
        assert k < b, (a, k, b)


@pytest.mark.parametrize("gen", [generate_key_between, generate_suffixed_key_between])
def test_append_run_stays_sorted_and_short(gen):
    keys = [gen(None, None)]
    for _ in range(3000):
        keys.append(gen(keys[-1], None))
    assert keys == sorted(keys)
    assert len(set(keys)) == len(keys)
    # Integer part grows by one char per 62 keys at most: log growth.
    assert max(len(k) for k in keys) <= 4 + 4


@pytest.mark.parametrize("gen", [generate_key_between, generate_suffixed_key_between])
def test_prepend_run_stays_sorted_and_short(gen):
    keys = [gen(None, None)]
    for _ in range(3000):
        keys.insert(0, gen(None, keys[0]))
    assert keys == sorted(keys)
    assert len(set(keys)) == len(keys)
    assert max(len(k) for k in keys) <= 4 + 4


@pytest.mark.parametrize("gen", [generate_key_between, generate_suffixed_key_between])
@pytest.mark.parametrize("seed", range(5))
def test_random_inserts_keep_order_and_uniqueness(gen, seed):
    rng = random.Random(seed)  # noqa: S311
    keys: list[str] = []
    for _ in range(2000):
        i = rng.randint(0, len(keys))
        a = keys[i - 1] if i > 0 else None
        b = keys[i] if i < len(keys) else None
        k = gen(a, b)
        _between(k, a, b)
        keys.insert(i, k)
    assert keys == sorted(keys)
    assert len(set(keys)) == len(keys)


@pytest.mark.parametrize("gen", [generate_key_between, generate_suffixed_key_between])
def test_repeated_insert_at_same_gap_grows_slowly(gen):
    """Inserting right after a fixed key again and again: about one char per 6."""
    a = gen(None, None)
    b = gen(a, None)
    keys = [a, b]
    for _ in range(300):
        k = gen(keys[0], keys[1])
        _between(k, keys[0], keys[1])
        keys.insert(1, k)
    assert keys == sorted(keys)
    assert len(set(keys)) == len(keys)
    assert max(len(k) for k in keys) < 300 / 3


@pytest.mark.parametrize("gen", [generate_key_between, generate_suffixed_key_between])
def test_repeated_insert_before_fixed_key_grows_slowly(gen):
    a = gen(None, None)
    b = gen(a, None)
    keys = [a, b]
    for _ in range(300):
        k = gen(keys[-2], keys[-1])
        _between(k, keys[-2], keys[-1])
        keys.insert(len(keys) - 1, k)
    assert keys == sorted(keys)
    assert len(set(keys)) == len(keys)
    assert max(len(k) for k in keys) < 300 / 3


# ============================================================================
# SUFFIXED KEYS
# ============================================================================


def test_suffixed_key_is_valid_and_between():
    for a, b in [
        (None, None),
        (None, "a0"),
        ("a0", None),
        ("a0", "a1"),
        ("a0", "a0V"),
        ("a0", "a01"),
        ("a0", "a015"),
        ("a0", "a0015"),
        ("a1abcd", "a2efgh"),
        (None, "a1abcd"),
        ("Zz", "a0"),
    ]:
        for _ in range(50):
            k = generate_suffixed_key_between(a, b)
            _between(k, a, b)


def test_suffixed_key_below_prefix_bound():
    """When the plain key is a prefix of b, random tails never cross b."""
    for a, b in [
        ("a0", "a015"),
        ("a0", "a0015"),
        ("a0", "a01zzzz"),
        ("a0", "a1abcd"),
        (None, "a0abcd"),
    ]:
        plain = generate_key_between(a, b)
        assert b.startswith(plain)
        for _ in range(200):
            k = generate_suffixed_key_between(a, b)
            _between(k, a, b)
            assert not k.startswith(plain) or a is not None


def test_suffixed_keys_from_same_neighbours_differ():
    seen = {generate_suffixed_key_between("a0", "a1") for _ in range(500)}
    assert len(seen) == 500
    for k in seen:
        _between(k, "a0", "a1")


def test_suffixed_key_tail_alphabet():
    k = generate_suffixed_key_between(None, None, suffix_length=6)
    assert k.startswith("a0")
    assert len(k) == 8
    assert all(ch in BASE_62_DIGITS for ch in k)
    assert not k.endswith("0")


def test_suffixed_keys_chain_as_inputs():
    """Keys with tails are ordinary keys for the next call, in any direction."""
    rng = random.Random(7)  # noqa: S311
    keys = [generate_suffixed_key_between(None, None)]
    for _ in range(1500):
        i = rng.randint(0, len(keys))
        a = keys[i - 1] if i > 0 else None
        b = keys[i] if i < len(keys) else None
        k = generate_suffixed_key_between(a, b)
        _between(k, a, b)
        keys.insert(i, k)
    assert keys == sorted(keys)
    assert len(set(keys)) == len(keys)
