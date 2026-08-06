"""Storage common types."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

from .key import Key


__all__ = [
    "CallbackFn",
]

CallbackFn: TypeAlias = Callable[[Key], None]
