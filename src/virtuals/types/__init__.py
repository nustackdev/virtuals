"""Virtuals types module."""

from __future__ import annotations

from .empty import (
    EMPTY,
    Empty,
    is_empty,
)
from .not_set import NOT_SET, NotSet, is_notset
from .tkv import (
    CallbackFn,
    CompositeValue,
    PrimitiveValue,
    Value,
)


__all__ = [
    "EMPTY",
    "NOT_SET",
    "CallbackFn",
    "CompositeValue",
    "Empty",
    "NotSet",
    "PrimitiveValue",
    "Value",
    "is_empty",
    "is_notset",
]
