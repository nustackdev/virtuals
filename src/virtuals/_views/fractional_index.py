"""Fractional index order keys.

A port of the rocicorp ``fractional-indexing`` algorithm (base62 digits,
variable-length integer part whose first char encodes its length) plus a
random suffix so independent writers that insert between the same two
neighbours never produce the same key.

Key shape::

    <head><integer digits><fraction digits>

``head`` is ``a``..``z`` for integer parts of length 2..27 (positive side)
and ``Z``..``A`` for length 2..27 on the negative side. Keys compare as
plain strings, which is byte order for the base62 alphabet, so a storage
backend that iterates keys in byte order iterates them in index order.

Public entry points:

- ``generate_key_between(a, b)``: the plain rocicorp key. Deterministic.
- ``generate_suffixed_key_between(a, b)``: the key the view stores. The
  plain key is extended with random base62 chars; when the plain key is a
  prefix of ``b`` the plain step is repeated with that key as the upper
  bound, so the random tail can never cross ``b``. The result is itself a
  valid key, so it is a fine input for the next call. Two writers that
  compute a key from the same neighbours get keys that differ in the
  random tail and both sort strictly between those neighbours.
- ``generate_n_suffixed_keys_between(a, b, n)``: ``n`` evenly spread
  suffixed keys, used to respace a run of rows.
"""

from __future__ import annotations

import secrets


__all__ = [
    "BASE_62_DIGITS",
    "generate_key_between",
    "generate_n_keys_between",
    "generate_n_suffixed_keys_between",
    "generate_suffixed_key_between",
    "validate_order_key",
]

BASE_62_DIGITS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
"""Digit alphabet, in ascending byte order."""

_ZERO = BASE_62_DIGITS[0]
_SMALLEST_INTEGER = "A" + _ZERO * 26
_INTEGER_ZERO = "a" + _ZERO

DEFAULT_SUFFIX_LENGTH = 4
"""Random chars appended by ``generate_suffixed_key_between``."""


# =============================================================================
# rocicorp port
# =============================================================================


def _midpoint(a: str, b: str | None) -> str:
    """Return a string strictly between ``a`` and ``b`` (``b`` None means +inf).

    Both are fraction strings (no integer part). Neither may end in zero.
    """
    if b is not None and a >= b:
        raise ValueError(f"{a!r} >= {b!r}")
    if a.endswith(_ZERO) or (b is not None and b.endswith(_ZERO)):
        raise ValueError("trailing zero")
    if b is not None:
        # Strip the longest common prefix; `a` is padded with zeros as we go.
        n = 0
        while (a[n] if n < len(a) else _ZERO) == (b[n] if n < len(b) else None):
            n += 1
        if n > 0:
            return b[:n] + _midpoint(a[n:], b[n:])
    digit_a = BASE_62_DIGITS.index(a[0]) if a else 0
    digit_b = BASE_62_DIGITS.index(b[0]) if b is not None else len(BASE_62_DIGITS)
    if digit_b - digit_a > 1:
        # Round half up, as the JS reference does (Math.round).
        mid = (digit_a + digit_b + 1) // 2
        return BASE_62_DIGITS[mid]
    # First digits are consecutive.
    if b is not None and len(b) > 1:
        return b[:1]
    return BASE_62_DIGITS[digit_a] + _midpoint(a[1:], None)


def _integer_length(head: str) -> int:
    if "a" <= head <= "z":
        return ord(head) - ord("a") + 2
    if "A" <= head <= "Z":
        return ord("Z") - ord(head) + 2
    raise ValueError(f"invalid order key head: {head!r}")


def _integer_part(key: str) -> str:
    length = _integer_length(key[0])
    if length > len(key):
        raise ValueError(f"invalid order key: {key!r}")
    return key[:length]


def _validate_integer(integer: str) -> None:
    if len(integer) != _integer_length(integer[0]):
        raise ValueError(f"invalid integer part of order key: {integer!r}")


def validate_order_key(key: str) -> None:
    """Raise ``ValueError`` unless ``key`` is a well-formed order key."""
    if not key:
        raise ValueError("invalid order key: ''")
    if key == _SMALLEST_INTEGER:
        raise ValueError(f"invalid order key: {key!r}")
    integer = _integer_part(key)
    fraction = key[len(integer) :]
    if fraction.endswith(_ZERO):
        raise ValueError(f"invalid order key: {key!r}")
    for ch in key[1:]:
        if ch not in BASE_62_DIGITS:
            raise ValueError(f"invalid order key: {key!r}")


def _increment_integer(integer: str) -> str | None:
    """Next integer part, or None at the largest integer."""
    _validate_integer(integer)
    head, digs = integer[0], list(integer[1:])
    carry = True
    i = len(digs) - 1
    while carry and i >= 0:
        d = BASE_62_DIGITS.index(digs[i]) + 1
        if d == len(BASE_62_DIGITS):
            digs[i] = _ZERO
        else:
            digs[i] = BASE_62_DIGITS[d]
            carry = False
        i -= 1
    if not carry:
        return head + "".join(digs)
    if head == "Z":
        return "a" + _ZERO
    if head == "z":
        return None
    h = chr(ord(head) + 1)
    if h > "a":
        digs.append(_ZERO)
    else:
        digs.pop()
    return h + "".join(digs)


def _decrement_integer(integer: str) -> str | None:
    """Previous integer part, or None at the smallest integer."""
    _validate_integer(integer)
    head, digs = integer[0], list(integer[1:])
    borrow = True
    i = len(digs) - 1
    while borrow and i >= 0:
        d = BASE_62_DIGITS.index(digs[i]) - 1
        if d == -1:
            digs[i] = BASE_62_DIGITS[-1]
        else:
            digs[i] = BASE_62_DIGITS[d]
            borrow = False
        i -= 1
    if not borrow:
        return head + "".join(digs)
    if head == "a":
        return "Z" + BASE_62_DIGITS[-1]
    if head == "A":
        return None
    h = chr(ord(head) - 1)
    if h < "Z":
        digs.append(BASE_62_DIGITS[-1])
    else:
        digs.pop()
    return h + "".join(digs)


def generate_key_between(a: str | None, b: str | None) -> str:
    """Return a key strictly between ``a`` and ``b``.

    ``None`` for ``a`` means "before everything", ``None`` for ``b`` means
    "after everything". Both bounds must be valid keys with ``a < b``.
    """
    if a is not None:
        validate_order_key(a)
    if b is not None:
        validate_order_key(b)
    if a is not None and b is not None and a >= b:
        raise ValueError(f"{a!r} >= {b!r}")
    if a is None:
        if b is None:
            return _INTEGER_ZERO
        ib = _integer_part(b)
        fb = b[len(ib) :]
        if ib == _SMALLEST_INTEGER:
            return ib + _midpoint("", fb)
        if ib < b:
            return ib
        res = _decrement_integer(ib)
        if res is None:
            raise ValueError("cannot decrement any more")
        return res
    if b is None:
        ia = _integer_part(a)
        fa = a[len(ia) :]
        i = _increment_integer(ia)
        return ia + _midpoint(fa, None) if i is None else i
    ia = _integer_part(a)
    fa = a[len(ia) :]
    ib = _integer_part(b)
    fb = b[len(ib) :]
    if ia == ib:
        return ia + _midpoint(fa, fb)
    i = _increment_integer(ia)
    if i is None:
        raise ValueError("cannot increment any more")
    if i < b:
        return i
    return ia + _midpoint(fa, None)


def generate_n_keys_between(a: str | None, b: str | None, n: int) -> list[str]:
    """Return ``n`` ascending keys strictly between ``a`` and ``b``."""
    if n == 0:
        return []
    if n == 1:
        return [generate_key_between(a, b)]
    if b is None:
        out = [generate_key_between(a, None)]
        for _ in range(n - 1):
            out.append(generate_key_between(out[-1], None))
        return out
    if a is None:
        out = [generate_key_between(None, b)]
        for _ in range(n - 1):
            out.append(generate_key_between(None, out[-1]))
        out.reverse()
        return out
    mid = n // 2
    c = generate_key_between(a, b)
    return [*generate_n_keys_between(a, c, mid), c, *generate_n_keys_between(c, b, n - mid - 1)]


# =============================================================================
# Suffixed keys
# =============================================================================


def _random_suffix(length: int) -> str:
    """Random base62 chars; the last one is never zero so the key stays valid."""
    if length < 1:
        raise ValueError("suffix length must be >= 1")
    body = "".join(secrets.choice(BASE_62_DIGITS) for _ in range(length - 1))
    return body + secrets.choice(BASE_62_DIGITS[1:])


def generate_suffixed_key_between(
    a: str | None,
    b: str | None,
    *,
    suffix_length: int = DEFAULT_SUFFIX_LENGTH,
) -> str:
    """Return a key strictly between ``a`` and ``b`` with a random tail.

    Takes the plain ``generate_key_between`` key ``k`` and appends random
    digits. ``k > a`` and ``k`` is never a prefix of ``a``, so anything
    appended to ``k`` stays above ``a``. If ``k`` is a proper prefix of
    ``b`` (the integer part of ``b``, or the consecutive-digit midpoint
    case) a random tail could cross ``b``, so ``k`` itself becomes the
    upper bound and the plain step runs again, ``generate_key_between(a, k)``,
    until the key is no longer a prefix of ``b``. Each step shortens a
    prefix of ``b``, so the loop ends. The key then differs from ``b`` at
    a position inside itself, so anything appended stays below ``b``.

    Stepping the bound down (rather than digging into ``b``'s remainder)
    matters when ``b`` is a long key just above a short integer: the key
    stays near ``a``'s length instead of copying ``b``'s zeros, which is
    what lets a window respace shrink keys next to a long neighbour.

    The result is a valid key and sorts strictly between the bounds. Keys
    inserted again and again at one gap grow about one char per four
    inserts (one per six after a fixed key, one per four before one).
    """
    k = generate_key_between(a, b)
    while b is not None and b.startswith(k):
        k = generate_key_between(a, k)
    return k + _random_suffix(suffix_length)


def generate_n_suffixed_keys_between(
    a: str | None,
    b: str | None,
    n: int,
    *,
    suffix_length: int = DEFAULT_SUFFIX_LENGTH,
) -> list[str]:
    """Return ``n`` ascending suffixed keys strictly between ``a`` and ``b``.

    Open ends chain plain steps (integer increments or decrements, so an
    append or prepend run stays short); a bounded gap is split at its
    midpoint recursively, so the keys are evenly spread and about
    ``log2(n)`` steps deep.
    """
    if n <= 0:
        return []
    if n == 1:
        return [generate_suffixed_key_between(a, b, suffix_length=suffix_length)]
    if b is None:
        out = [generate_suffixed_key_between(a, None, suffix_length=suffix_length)]
        for _ in range(n - 1):
            out.append(generate_suffixed_key_between(out[-1], None, suffix_length=suffix_length))
        return out
    if a is None:
        out = [generate_suffixed_key_between(None, b, suffix_length=suffix_length)]
        for _ in range(n - 1):
            out.append(generate_suffixed_key_between(None, out[-1], suffix_length=suffix_length))
        out.reverse()
        return out
    mid = n // 2
    c = generate_suffixed_key_between(a, b, suffix_length=suffix_length)
    return [
        *generate_n_suffixed_keys_between(a, c, mid, suffix_length=suffix_length),
        c,
        *generate_n_suffixed_keys_between(c, b, n - mid - 1, suffix_length=suffix_length),
    ]
