"""Type definitions for key codec operations."""

from __future__ import annotations

from typing import TypeVar

from virtuals.tkv.types import Key, KeySegment


__all__ = [
    "EncodedBinaryKey",
    "EncodedKeyT",
    "EncodedStringKey",
    "Key",
    "KeySegment",
]

# Generic type for encoded keys
EncodedKeyT = TypeVar("EncodedKeyT")

# Encoded key types for different codec implementations
EncodedBinaryKey = bytes
EncodedStringKey = str
