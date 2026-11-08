"""RingView: a capped log. Appends go to the end, the oldest entries drop.

Layout under the view's container::

    site/
    ├── 0: entry        # oldest kept (sequence numbers only grow)
    ├── 1: entry
    └── ...

Every entry is one primitive row keyed by its sequence number, an int
that only grows: a compound entry (``[ts, stream, text]``) is stored whole,
never decomposed. Sequence numbers are dense from the oldest kept entry to
the newest, so ``len`` and a read by position cost two seeks and a get,
whatever the capacity.

Capacity is a property of the class: ``RingView.of(1000)`` is a ring of at
most 1000 entries. Reads work through any ring class (a ring opened by its
stored structure is a plain ``RingView``); writes need a capacity and raise
TypeError without one. Shrinking the capacity drops the extra entries on
the next append.

Cost. ``append`` reads the newest row (one seek), writes one row, and
reads the oldest rows up to the first it keeps (one seek, two rows in a
full ring), deleting the ones that fall out: a constant number of keys.
``extend`` does the same once for the whole batch. No stored length, no
key shared with any other container.

Cursor. A sequence number marks a position for good: ``entries(after=seq)``
resumes after it even once it has dropped out, so a reader can follow a
ring by its last seen sequence number.

Concurrency. Two writers appending at once read the same newest row and
pick the same sequence number. sqlite and lmdb serialize writers, so the
second sees the first; rocksdb and mem keep one of the two entries. A ring
is meant for one writer (a run writing its own output).

Observability: ``on_change`` covers every row; ``on_children_change``
fires once per appended row and once per dropped one.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache
from typing import TYPE_CHECKING, ClassVar, cast

from virtuals.container import ContainerProtocol, ContainerStructure
from virtuals.tkv.filter import WildcardFilter
from virtuals.tkv.observer import SubscriptionOptions
from virtuals.types import EMPTY, is_empty
from virtuals.view import ObservableBase, ViewBase


if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

    from virtuals.types import Value


__all__ = ["RingView"]


class RingView(ObservableBase, ViewBase):
    """A capped log of primitive entries, oldest first.

    Example::

        >>> out = root.open_child("out", RingView.of(3))
        >>> out.extend(["a", "b", "c", "d"])
        >>> list(out)
        ['b', 'c', 'd']
        >>> out[0], out[-1], len(out)
        ('b', 'd', 3)
        >>> list(out.entries(after=2))
        [(3, 'd')]
    """

    STRUCTURE: ClassVar[ContainerStructure] = ContainerStructure(18)
    PROTOCOL: ClassVar[ContainerProtocol] = (
        ContainerProtocol.INDEXED | ContainerProtocol.MUTABLE | ContainerProtocol.SIZED
    )
    CAPACITY: ClassVar[int | None] = None
    """Most entries kept. None: the ring reads but refuses writes."""

    @classmethod
    def of(cls, capacity: int) -> type[RingView]:
        """The ring class keeping at most ``capacity`` entries. One class per capacity."""
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError(f"ring capacity must be an int >= 1, got {capacity!r}")
        return _ring_of(cls, capacity)

    # -- Sequence numbers ---------------------------------------------------

    def _edge_seq(self, *, newest: bool) -> int | None:
        """Sequence number of the oldest or newest entry, None when empty. One seek."""
        for seq in self.container.iter_child_keys(strict=True, reverse=newest, limit=1):
            return cast("int", seq)
        return None

    def _rows(
        self, *, after: int | None = None, limit: int | None = None, reverse: bool = False
    ) -> Generator[tuple[int, Value], None, None]:
        """``(seq, entry)`` rows in order, strictly past ``after``. One scan."""
        rows = self.container.iter_children(strict=True, reverse=reverse, cursor=after, limit=limit)
        for seq, info in rows:
            yield cast("int", seq), cast("Value", info.primitive_value)

    # -- Read ---------------------------------------------------------------

    def __len__(self) -> int:
        """Entries kept. Two seeks: sequence numbers are dense."""
        oldest = self._edge_seq(newest=False)
        if oldest is None:
            return 0
        newest = cast("int", self._edge_seq(newest=True))
        return newest - oldest + 1

    def __bool__(self) -> bool:
        """True if the ring holds an entry. One seek."""
        return self._edge_seq(newest=False) is not None

    def __getitem__(self, index: int) -> Value:
        """Entry by position, oldest first; negative counts from the newest."""
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError(f"ring indices must be integers, got {type(index).__name__}")
        edge = self._edge_seq(newest=index < 0)
        if edge is None:
            raise IndexError("ring index out of range")
        seq = edge + index if index >= 0 else edge + index + 1
        entry = self.container.get_child_primitive_unsafe(seq) if seq >= 0 else EMPTY
        if is_empty(entry):
            raise IndexError("ring index out of range")
        return cast("Value", entry)

    def __iter__(self) -> Generator[Value, None, None]:
        """Entries, oldest first."""
        for _seq, entry in self._rows():
            yield entry

    def __reversed__(self) -> Generator[Value, None, None]:
        """Entries, newest first."""
        for _seq, entry in self._rows(reverse=True):
            yield entry

    def entries(
        self, *, after: int | None = None, limit: int | None = None
    ) -> Generator[tuple[int, Value], None, None]:
        """``(seq, entry)`` oldest first, past sequence number ``after``, at most ``limit``.

        ``after`` need not be kept any more: a reader that fell behind
        resumes at the oldest entry still there. Cost: one seek plus
        O(limit) rows.
        """
        return self._rows(after=after, limit=limit)

    def extract(self) -> list[Value]:
        """All entries kept, oldest first."""
        return list(self)

    # -- Write --------------------------------------------------------------

    def append(self, entry: Value) -> int:
        """Add ``entry`` as the newest; drop the oldest past capacity. Returns its seq."""
        return cast("int", self._push([entry]))

    def extend(self, entries: Iterable[Value]) -> None:
        """Append every entry in order. Writes nothing for an empty iterable."""
        self._push(entries)

    def store(self, entries: Iterable[Value], *, replace: bool = True) -> None:
        """Replace the contents with ``entries`` (the last ``capacity`` of them).

        Sequence numbers restart from 0. Cost: O(entries kept before and after).
        ``replace=False`` skips clearing, for a ring known to be new (its
        parent just created it); on a ring holding entries it appends.
        """
        self._require_capacity()
        batch = self._reject_empty_in(list(entries))
        self.ensure_created()
        if replace:
            self.container.clear_children(validate=False)
        self._push(cast("list[Value]", batch))

    def clear(self) -> None:
        """Drop every entry. Sequence numbers restart from 0."""
        if not self.container.exists():
            return  # clear of nothing is not a write: the view stays absent
        self.container.clear_children(validate=False)

    def _push(self, entries: Iterable[Value]) -> int | None:
        """Write ``entries`` after the newest, then drop what falls past capacity.

        An oversized batch still numbers every entry, but writes only the
        last ``capacity`` of them. Returns the newest seq, None if nothing
        was given.
        """
        capacity = self._require_capacity()
        batch = cast("list[Value]", self._reject_empty_in(list(entries)))
        if not batch:
            return None
        self.ensure_created()
        newest = self._edge_seq(newest=True)
        first = 0 if newest is None else newest + 1
        last = first + len(batch) - 1
        for seq in range(max(first, last - capacity + 1), last + 1):
            self.container.put_child_primitive(
                seq, batch[seq - first], validate=False, validate_parent=False
            )
        self._trim(last - capacity)
        return last

    def _trim(self, through: int) -> None:
        """Delete every row with a sequence number up to ``through``. Reads to the first kept."""
        if through < 0:
            return
        dropped = []
        for seq in self.container.iter_child_keys(strict=False):
            if cast("int", seq) > through:
                break
            dropped.append(seq)
        for seq in dropped:
            self.container.delete_child_primitive_unsafe(seq)

    def _require_capacity(self) -> int:
        capacity = type(self).CAPACITY
        if capacity is None:
            raise TypeError("a ring without capacity is read-only: open it as RingView.of(n)")
        return capacity

    # -- Observability ------------------------------------------------------

    def on_children_change(self) -> SubscriptionOptions:
        """An entry appended or dropped (one row each)."""
        return SubscriptionOptions(WildcardFilter(pattern=(*self.container.site, "*")))


@cache
def _ring_of(base: type[RingView], capacity: int) -> type[RingView]:
    name = f"{base.__name__}Of{capacity}"
    return cast(
        "type[RingView]",
        type(name, (base,), {"CAPACITY": capacity, "__module__": base.__module__}),
    )


Sequence.register(RingView)
