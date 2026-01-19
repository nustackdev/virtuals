"""PV - Polymorphic Views for Key-Value stores.

KV stores it. PV shapes it.
"""

from __future__ import annotations

from ._exception import PVError
from .container import Container
from .typing import (
    NOT_SET,
    Empty,
    Invalid,
    NotSet,
    Sentinel,
    Value,
    is_empty,
    is_invalid,
    is_notset,
    is_sentinel,
)
from .view import View


__all__ = [
    "NOT_SET",
    "Container",
    "Empty",
    "Invalid",
    "NotSet",
    "PVError",
    "Sentinel",
    "Value",
    "View",
    "is_empty",
    "is_invalid",
    "is_notset",
    "is_sentinel",
]
