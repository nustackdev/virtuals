"""Virtuals - Virtual Python collections over any storage."""

from __future__ import annotations

from .container import Container
from .navigator import Navigator
from .types import (
    NOT_SET,
    Empty,
    NotSet,
    Value,
    is_empty,
    is_notset,
)
from .view import View, ViewRegistry


__all__ = [
    "NOT_SET",
    "Container",
    "Empty",
    "Navigator",
    "NotSet",
    "Value",
    "View",
    "ViewRegistry",
    "is_empty",
    "is_notset",
]
