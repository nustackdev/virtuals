"""Storage common types."""

from __future__ import annotations

from collections.abc import Callable

from .key import Key


__all__ = [
    "CallbackFn",
]

type CallbackFn = Callable[[Key], None]
