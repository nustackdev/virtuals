"""Virtuals - Virtual Python collections over any storage."""

from __future__ import annotations

from .container import Container
from .types import (
    NOT_SET,
    Empty,
    NotSet,
    Value,
    is_empty,
    is_notset,
)
from .view import View


__all__ = [
    "NOT_SET",
    "Container",
    "Empty",
    "NotSet",
    "Value",
    "View",
    "is_empty",
    "is_notset",
]
