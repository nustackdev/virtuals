"""Special sentinel values for ABC modules."""

from __future__ import annotations

from typing import TypeGuard


__all__ = [
    "EMPTY",
    "Empty",
    "is_empty",
]


class Empty:
    """Sentinel for non-existent values."""

    def __repr__(self) -> str:
        """String representation for debugging."""
        return "<Empty>"

    def __str__(self) -> str:
        """String representation for display."""
        return "Empty"

    def __bool__(self) -> bool:
        """Boolean evaluation, always False."""
        return False

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Empty)

    def __hash__(self) -> int:
        return hash("Empty")


EMPTY = Empty()


def is_empty(value: object) -> TypeGuard[Empty]:
    """Check if value is Empty sentinel."""
    return isinstance(value, Empty)
