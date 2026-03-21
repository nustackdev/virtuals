"""Virtuals - Virtual Python collections over any storage."""

from __future__ import annotations

from .container import Container
from .navigator import Navigator, ViewScope
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
    "Navigator",
    "NotSet",
    "Value",
    "View",
    "ViewScope",
    "is_empty",
    "is_notset",
]
