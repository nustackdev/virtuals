"""Committed state of the in-memory storage, with versioned reads for snapshots.

Live state is a dict from encoded key to ``(key, value)`` for O(1) point
reads, plus a ``SortedKeys`` index of the encoded keys for seek, range and
reverse scans. Every commit is applied under one lock as a whole, so a
reader never sees half of it.

Snapshots are multi-version reads over an undo log. A commit sequence
number advances on every commit; a snapshot pins the sequence it opened
at. While any snapshot is open, each commit records, per key it touches,
the value that key had before the commit (or ABSENT). A snapshot pinned at
``p`` reads a key as the oldest recorded version written by a commit after
``p``, or the live value when there is none. Nothing is copied on begin,
and nothing is recorded when no snapshot is open. When the oldest snapshot
closes, every recorded version no open snapshot can need is dropped.

Scans are served in pages: a page is a slice of the live index merged with
the overlay that applies (recorded versions for a snapshot, pending writes
for a transaction). ``merge_pages`` is the shared merge; it returns the
page together with the key the next page must seek past, so iteration
re-seeks instead of holding an iterator over a structure other threads
may change.
"""

from __future__ import annotations

import threading
from bisect import bisect_right
from collections import deque
from operator import itemgetter
from typing import TYPE_CHECKING, Any, TypeAlias

from .sorted_keys import SortedKeys


if TYPE_CHECKING:
    from virtuals.tkv.types import Key


__all__ = [
    "ABSENT",
    "DELETED",
    "LIVE",
    "Encoded",
    "Entry",
    "Marker",
    "VersionedStore",
    "merge_pages",
]


class Marker:
    """A named sentinel."""

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:
        return self._name


ABSENT = Marker("ABSENT")
"""No value: the key does not exist in the view being read."""

DELETED = Marker("DELETED")
"""Pending delete in a transaction's or batch's write set."""

LIVE = Marker("LIVE")
"""A snapshot overlay entry saying: no newer version recorded, read the live value."""


Encoded: TypeAlias = object
"""An encoded key: whatever the codec produces (bytes or str in practice). All keys of
one storage are mutually comparable, and the index orders them with ``<``."""

_commit_seq = itemgetter(0)

Entry: TypeAlias = tuple[Any, Any]
"""``(decoded key, encoded value)`` as kept in live state and pending write sets."""


def merge_pages(
    base: list[tuple[Any, Entry]],
    overlay: list[tuple[Any, Any]],
    n: int,
    reverse: bool,
) -> tuple[list[tuple[Any, Entry]], Any, bool]:
    """Merge one page of base rows with one page of overlay entries.

    Both inputs are in scan order and hold at most ``n`` rows; fewer than
    ``n`` means that source is exhausted. An overlay entry is an ``Entry``
    that replaces the base row, ``ABSENT`` to hide it, or ``LIVE`` to keep
    the base row as is.

    Returns ``(rows, cover, done)``: the merged rows up to ``cover``, the
    last key both sources are known to agree on (rows past it from the
    longer source are left for the next page, which seeks past ``cover``),
    and whether both sources are exhausted. ``cover`` is None when done.
    """
    base_more = len(base) == n
    over_more = len(overlay) == n
    if base_more and over_more:
        a, b = base[-1][0], overlay[-1][0]
        cover = (a if a > b else b) if reverse else (a if a < b else b)
    elif base_more:
        cover = base[-1][0]
    elif over_more:
        cover = overlay[-1][0]
    else:
        cover = None

    out: list[tuple[Any, Entry]] = []
    i = j = 0
    nb, no = len(base), len(overlay)
    while i < nb or j < no:
        if j == no:
            enc, entry = base[i]
            i += 1
        elif i == nb:
            enc, entry = overlay[j]
            j += 1
            if entry is LIVE:
                continue  # not live either: the key does not exist
        else:
            benc, oenc = base[i][0], overlay[j][0]
            if benc == oenc:
                enc, entry = overlay[j]
                if entry is LIVE:
                    entry = base[i][1]
                i += 1
                j += 1
            elif (benc > oenc) if reverse else (benc < oenc):
                enc, entry = base[i]
                i += 1
            else:
                enc, entry = overlay[j]
                j += 1
                if entry is LIVE:
                    continue
        if cover is not None and ((enc < cover) if reverse else (enc > cover)):
            break
        if entry is not ABSENT:
            out.append((enc, entry))
    return out, cover, cover is None


class VersionedStore:
    """Live state plus the versions open snapshots still need."""

    def __init__(self) -> None:
        """Start empty at sequence 0."""
        self.lock = threading.Lock()
        self._live: dict[Any, Entry] = {}
        self._index: SortedKeys[Any] = SortedKeys()
        self._seq = 0
        # Versions recorded for open snapshots, per key in commit order:
        # (commit seq, value before that commit or ABSENT).
        self._history: dict[Any, list[tuple[int, Any]]] = {}
        self._history_index: SortedKeys[Any] = SortedKeys()
        # Same records in global commit order, for trimming oldest first.
        self._undo_log: deque[tuple[int, Any]] = deque()
        # Open snapshots: pinned seq -> count. Insertion order is seq order.
        self._pins: dict[int, int] = {}

    def reset(self) -> None:
        """Drop everything, keeping the lock."""
        with self.lock:
            self._live = {}
            self._index = SortedKeys()
            self._seq = 0
            self._history = {}
            self._history_index = SortedKeys()
            self._undo_log = deque()
            self._pins = {}

    def __len__(self) -> int:
        return len(self._live)

    # ------------------------------------------------------------------
    # Live reads (what a write transaction sees under the writer slot)
    # ------------------------------------------------------------------

    def get(self, enc: Encoded) -> Entry | Marker:
        """Live entry for ``enc``, or ABSENT."""
        with self.lock:
            return self._live.get(enc, ABSENT)

    def page(
        self,
        bound: Encoded | None,
        *,
        inclusive: bool,
        reverse: bool,
        n: int,
        stop: Encoded | None = None,
    ) -> list[tuple[Any, Entry]]:
        """Up to ``n`` live rows from ``bound`` in scan order. See ``SortedKeys.page``."""
        with self.lock:
            encs = self._index.page(bound, inclusive=inclusive, reverse=reverse, n=n, stop=stop)
            live = self._live
            return [(e, live[e]) for e in encs]

    # ------------------------------------------------------------------
    # Snapshot reads
    # ------------------------------------------------------------------

    def pin(self) -> int:
        """Register an open snapshot at the current sequence and return it."""
        with self.lock:
            seq = self._seq
            self._pins[seq] = self._pins.get(seq, 0) + 1
            return seq

    def unpin(self, seq: int) -> None:
        """Release a snapshot pinned at ``seq`` and drop versions nobody needs."""
        with self.lock:
            count = self._pins.get(seq, 0) - 1
            if count > 0:
                self._pins[seq] = count
                return
            self._pins.pop(seq, None)
            self._trim()

    def _trim(self) -> None:
        """Drop recorded versions no open snapshot can read."""
        pins = self._pins
        floor = next(iter(pins)) if pins else None
        log = self._undo_log
        drop: dict[Encoded, int] = {}
        while log and (floor is None or log[0][0] <= floor):
            _, enc = log.popleft()
            drop[enc] = drop.get(enc, 0) + 1
        history = self._history
        for enc, count in drop.items():
            entries = history[enc]
            if count >= len(entries):
                del history[enc]
                self._history_index.discard(enc)
            else:
                del entries[:count]  # the oldest versions come first

    def _version_at(self, enc: Encoded, seq: int) -> Entry | Marker:
        """Entry of ``enc`` as of ``seq``: a recorded version, ABSENT, or LIVE."""
        entries = self._history.get(enc)
        if entries is None:
            return LIVE
        i = bisect_right(entries, seq, key=_commit_seq)
        return entries[i][1] if i < len(entries) else LIVE

    def get_at(self, enc: Encoded, seq: int) -> Entry | Marker:
        """Entry for ``enc`` as it stood at sequence ``seq``, or ABSENT."""
        with self.lock:
            entry = self._version_at(enc, seq)
            if entry is LIVE:
                return self._live.get(enc, ABSENT)
            return entry

    def page_at(
        self,
        seq: int,
        bound: Encoded | None,
        *,
        inclusive: bool,
        reverse: bool,
        n: int,
        stop: Encoded | None = None,
    ) -> tuple[list[tuple[Any, Entry]], Any, bool]:
        """One page of rows as of ``seq``. Returns ``(rows, cover, done)`` as ``merge_pages``."""
        with self.lock:
            encs = self._index.page(bound, inclusive=inclusive, reverse=reverse, n=n, stop=stop)
            live = self._live
            base = [(e, live[e]) for e in encs]
            if not self._history:
                return base, (base[-1][0] if len(base) == n else None), len(base) < n
            encs = self._history_index.page(
                bound, inclusive=inclusive, reverse=reverse, n=n, stop=stop
            )
            overlay = [(e, self._version_at(e, seq)) for e in encs]
            return merge_pages(base, overlay, n, reverse)

    # ------------------------------------------------------------------
    # Commit
    # ------------------------------------------------------------------

    def apply(self, ops: dict[Any, Entry], modified: set[Key]) -> int:
        """Apply a write set atomically and return the new sequence.

        ``ops`` maps encoded key to ``(key, value)`` or ``(key, DELETED)``.
        Keys of deletes that removed an existing row are added to
        ``modified`` (deleting an absent key changes nothing).
        """
        with self.lock:
            self._seq += 1
            seq = self._seq
            live, index = self._live, self._index
            record = bool(self._pins)
            history, history_index, log = self._history, self._history_index, self._undo_log
            for enc, (key, value) in ops.items():
                old = live.get(enc, ABSENT)
                if value is DELETED:
                    if old is ABSENT:
                        continue
                    del live[enc]
                    index.discard(enc)
                    modified.add(key)
                else:
                    live[enc] = (key, value)
                    if old is ABSENT:
                        index.add(enc)
                if record:
                    entries = history.get(enc)
                    if entries is None:
                        history[enc] = [(seq, old)]
                        history_index.add(enc)
                    else:
                        entries.append((seq, old))
                    log.append((seq, enc))
            return seq
