"""Core definitions for Key location.

Keys are tuples of strings and integers, used for identifying
entries in storage systems.

Keys are primarily used storage and tree layers.
"""

from __future__ import annotations

from virtuals.tkv.types import Key, KeySegment


__all__ = [
    "Key",
    "KeySegment",
]
