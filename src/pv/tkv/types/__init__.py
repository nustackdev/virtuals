"""PV types module.

This module defines the primitive types and special values.
"""

from __future__ import annotations

from .empty import EMPTY, Empty, is_empty
from .key import Key, KeySegment
from .observer import CallbackFn
from .value import (
    CompositeValue,
    PrimitiveValue,
    Value,
)


__all__ = [
    "EMPTY",
    "CallbackFn",
    "CompositeValue",
    "Empty",
    "Key",
    "KeySegment",
    "PrimitiveValue",
    "Value",
    "is_empty",
]
