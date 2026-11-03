"""Chunked sorted list of encoded keys with native seek.

The ordered index behind the in-memory storage. Keys are kept in a list of
sorted sublists (the SortedList layout): a bisect over the sublist maxima
finds the sublist, a bisect inside it finds the slot. Insert and delete
move at most one sublist's worth of pointers, and a seek is two bisects.
Membership is not answered here; the storage's dict does that in O(1).

Only the operations the storage needs are provided: ``add`` of a key known
to be absent, ``discard``, ``first``/``last``, and ``page``, which returns
up to ``n`` keys from a bound in either direction. Scans page by re-seeking
from the last key they emitted, so no iterator is ever held open across
a mutation.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right, insort
from typing import TYPE_CHECKING, Generic, TypeVar


if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


__all__ = ["SortedKeys"]


K = TypeVar("K")

# Sublists split at twice this and merge with a neighbour below half of it.
_LOAD = 1000


class SortedKeys(Generic[K]):
    """Sorted set of mutually comparable keys, stored as sorted sublists."""

    __slots__ = ("_len", "_lists", "_maxes")

    def __init__(self, keys: Iterable[K] = ()) -> None:
        """Build the index, bulk-loading ``keys`` when given."""
        self._lists: list[list[K]] = []
        self._maxes: list[K] = []
        self._len = 0
        values = sorted(keys)
        if values:
            self._lists = [values[i : i + _LOAD] for i in range(0, len(values), _LOAD)]
            self._maxes = [sub[-1] for sub in self._lists]
            self._len = len(values)

    def __len__(self) -> int:
        return self._len

    def __iter__(self) -> Iterator[K]:
        for sub in self._lists:
            yield from sub

    def __contains__(self, key: object) -> bool:
        pos, idx = self._locate_left(key)  # type: ignore[arg-type]
        return (
            pos < len(self._lists) and idx < len(self._lists[pos]) and self._lists[pos][idx] == key
        )

    def first(self) -> K | None:
        """Smallest key, or None when empty."""
        return self._lists[0][0] if self._lists else None

    def last(self) -> K | None:
        """Largest key, or None when empty."""
        return self._lists[-1][-1] if self._lists else None

    def add(self, key: K) -> None:
        """Insert ``key``, which must not be present."""
        maxes = self._maxes
        if not maxes:
            self._lists.append([key])
            maxes.append(key)
            self._len = 1
            return
        pos = bisect_left(maxes, key)
        if pos == len(maxes):
            pos -= 1
        sub = self._lists[pos]
        insort(sub, key)
        maxes[pos] = sub[-1]
        self._len += 1
        if len(sub) > 2 * _LOAD:
            half = len(sub) >> 1
            self._lists.insert(pos + 1, sub[half:])
            del sub[half:]
            maxes[pos] = sub[-1]
            maxes.insert(pos + 1, self._lists[pos + 1][-1])

    def discard(self, key: K) -> bool:
        """Remove ``key`` if present. Returns True when it was."""
        pos, idx = self._locate_left(key)
        lists = self._lists
        if pos == len(lists):
            return False
        sub = lists[pos]
        if idx == len(sub) or sub[idx] != key:
            return False
        del sub[idx]
        self._len -= 1
        if not sub:
            del lists[pos]
            del self._maxes[pos]
            return True
        self._maxes[pos] = sub[-1]
        if len(sub) < _LOAD // 2 and len(lists) > 1:
            # Fold a thin sublist into a neighbour so deletes never leave
            # the index as a long chain of near-empty lists.
            other = pos + 1 if pos + 1 < len(lists) else pos - 1
            lo, hi = (pos, other) if other > pos else (other, pos)
            lists[lo].extend(lists[hi])
            del lists[hi]
            del self._maxes[hi]
            self._maxes[lo] = lists[lo][-1]
            if len(lists[lo]) > 2 * _LOAD:
                sub = lists[lo]
                half = len(sub) >> 1
                lists.insert(lo + 1, sub[half:])
                del sub[half:]
                self._maxes[lo] = sub[-1]
                self._maxes.insert(lo + 1, lists[lo + 1][-1])
        return True

    def page(
        self,
        bound: K | None,
        *,
        inclusive: bool,
        reverse: bool,
        n: int,
        stop: K | None = None,
    ) -> list[K]:
        """Up to ``n`` keys from ``bound`` in scan order.

        Forward, keys ascend from ``bound`` and stay below ``stop``
        (exclusive). Reverse, keys descend from ``bound`` and stay at or
        above ``stop`` (inclusive), which mirrors how a prefix range is
        bounded: ``[encode(prefix), upper_bound_of_prefix(prefix))``.
        ``bound`` None starts at the first (or last) key. Fewer than ``n``
        keys back means the range is exhausted.
        """
        if n <= 0 or not self._lists:
            return []
        out: list[K] = []
        if not reverse:
            if bound is None:
                pos, idx = 0, 0
            elif inclusive:
                pos, idx = self._locate_left(bound)
            else:
                pos, idx = self._locate_right(bound)
            lists = self._lists
            while pos < len(lists) and len(out) < n:
                sub = lists[pos]
                take = n - len(out)
                out.extend(sub[idx : idx + take])
                pos += 1
                idx = 0
            if stop is not None and out and out[-1] >= stop:
                del out[bisect_left(out, stop) :]
            return out

        if bound is None:
            pos = len(self._lists) - 1
            idx = len(self._lists[pos])
        elif inclusive:
            pos, idx = self._locate_right(bound)
        else:
            pos, idx = self._locate_left(bound)
        # (pos, idx) is the first slot past the range; walk backwards from it.
        lists = self._lists
        if pos == len(lists):
            pos -= 1
            idx = len(lists[pos])
        while pos >= 0 and len(out) < n:
            sub = lists[pos]
            take = n - len(out)
            lo = max(idx - take, 0)
            out.extend(sub[lo:idx][::-1])
            pos -= 1
            if pos >= 0:
                idx = len(lists[pos])
        if stop is not None:
            while out and out[-1] < stop:
                out.pop()
        return out

    def _locate_left(self, key: K) -> tuple[int, int]:
        """(sublist, slot) of the first key >= ``key``; sublist == len when past the end."""
        pos = bisect_left(self._maxes, key)
        if pos == len(self._maxes):
            return pos, 0
        return pos, bisect_left(self._lists[pos], key)

    def _locate_right(self, key: K) -> tuple[int, int]:
        """(sublist, slot) of the first key > ``key``; sublist == len when past the end."""
        pos = bisect_right(self._maxes, key)
        if pos == len(self._maxes):
            return pos, 0
        return pos, bisect_right(self._lists[pos], key)
