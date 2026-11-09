"""Index checks shared by the sequence views, worded the way ``list`` words them."""

from __future__ import annotations


def require_int_index(address: object, kind: str = "list") -> None:
    """TypeError for an index that is not an int (a bool counts, as in Python)."""
    if not isinstance(address, int):
        raise TypeError(f"{kind} indices must be integers or slices, not {type(address).__name__}")
