"""Typed, self-delimiting key segment encoding for server-side routing.

Lua only compares opaque byte strings, so ints never pass through Lua numbers
(doubles, lossy above 2**53). Encoding preserves Python `==` per segment:
`enc_seg(a) == enc_seg(b)` iff `a == b` for str / int / bool segments.

- int (incl. bool, int subclasses): `b"i" + str(int(v)) + b";"`  (True == 1)
- str (incl. subclasses): `b"s" + <utf8 byte length> + b":" + utf8 bytes`
- anything else: TypeError

Segments are self-delimiting, so a concatenation is prefix-free and the prefix
at depth d is the concat of the first d encoded segments.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterable

    from ...tkv.filter import RoutingHint
    from ...tkv.types import Key


__all__ = ["enc_key", "enc_seg", "hint_entries"]

logger = logging.getLogger(__name__)

HintEntry = tuple[bytes, bytes, bytes]
"""(kind, field, enc): kind in p/s/l/w/a; field is depth, length or `n:pos,...`."""

_ALL_ENTRY: HintEntry = (b"a", b"", b"")


def enc_seg(v: object) -> bytes:
    """Encode one key segment. Raises TypeError for non str/int segments."""
    if isinstance(v, int):
        return b"i%d;" % int(v)
    if isinstance(v, str):
        u = v.encode("utf-8", "surrogatepass")
        return b"s%d:" % len(u) + u
    raise TypeError(f"Unsupported key segment type: {type(v).__name__}")


def enc_key(key: Key) -> list[bytes]:
    """Encode every segment of a key."""
    return [enc_seg(v) for v in key]


def _entry(h: RoutingHint) -> HintEntry:
    if h.kind == "prefix":
        return (b"p", b"%d" % len(h.segments), b"".join(enc_key(h.segments)))
    if h.kind == "suffix":
        return (b"s", b"%d" % len(h.segments), b"".join(enc_key(h.segments)))
    if h.kind == "length":
        return (b"l", b"%d" % h.length, b"")
    if h.kind == "wildcard":
        field = b"%d:" % h.length + b",".join(b"%d" % p for p in h.positions)
        return (b"w", field, b"".join(enc_key(h.segments)))
    if h.kind == "all":
        return _ALL_ENTRY
    raise TypeError(f"Unknown hint kind: {h.kind!r}")


def hint_entries(hints: Iterable[RoutingHint]) -> list[HintEntry]:
    """Map hints to deduped `(kind, field, enc)` triples.

    If any hint fails to encode (e.g. a non str/int segment in a filter), the
    whole set falls back to `all`, so routing never under-delivers.
    """
    out: list[HintEntry] = []
    try:
        for h in hints:
            e = _entry(h)
            if e == _ALL_ENTRY:
                return [_ALL_ENTRY]
            if e not in out:
                out.append(e)
    except TypeError as exc:
        logger.warning("Routing hint not encodable, falling back to all: %s", exc)
        return [_ALL_ENTRY]
    return out
