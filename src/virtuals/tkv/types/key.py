"""Key types definitions.

Keys are tuples of strings and integers, used for identifying entries in storage systems.
"""

from __future__ import annotations

from typing import TypeAlias


__all__ = [
    "Key",
    "KeySegment",
]

KeySegment: TypeAlias = str | int
Key: TypeAlias = tuple[KeySegment, ...]
