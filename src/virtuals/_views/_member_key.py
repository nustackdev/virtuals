"""Storage key of a set member: a hash of its canonical pickle.

A member is keyed by ``sha256(pickle(value))``. For a value with no
unordered container inside, the pickled object is the value itself, so the
key is the one every earlier release wrote. A frozenset (or any hashable
``collections.abc.Set``) pickles in iteration order, which differs between
processes (hash randomization) and between two equal sets built in a
different order, so it is replaced before pickling by a tagged tuple of
its elements' canonical pickles, sorted; tuples are walked for the same
reason. Equal members then share one key in every process.
"""

from __future__ import annotations

import hashlib
import pickle  # nosec: F401
from collections.abc import Set as SetABC


_PROTOCOL = 4
_UNORDERED_TAG = b"\x00virtuals.set-member.unordered\x00"


def _canonical(value: object) -> object:
    """``value`` itself unless an unordered container sits inside it."""
    if isinstance(value, SetABC):
        return (
            _UNORDERED_TAG,
            tuple(sorted(pickle.dumps(_canonical(e), protocol=_PROTOCOL) for e in value)),
        )
    if isinstance(value, tuple):
        items = tuple(_canonical(e) for e in value)
        if all(a is b for a, b in zip(items, value, strict=True)):
            return value  # unchanged: pickle the original, keys stay as written before
        return items
    return value


def member_key(value: object) -> str:
    """Hex key for a set member. Raises like ``pickle.dumps`` on an unpicklable value."""
    pickled = pickle.dumps(_canonical(value), protocol=_PROTOCOL)
    return hashlib.sha256(pickled).hexdigest()[:64]
