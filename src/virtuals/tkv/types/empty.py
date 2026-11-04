"""Special sentinel values for ABC modules."""

from __future__ import annotations

from typing import TypeGuard


__all__ = [
    "EMPTY",
    "Empty",
    "is_empty",
]


class Empty:
    """Sentinel for non-existent values.

    A true singleton: ``Empty()`` is ``EMPTY``, and so is a copy, a deep
    copy or a pickle round trip (across processes included). So an
    identity test against ``EMPTY`` is exact everywhere, and the raw
    storage doors can refuse it at zero cost per put.
    """

    __slots__ = ()
    _instance: Empty | None = None

    def __new__(cls) -> Empty:
        """Return the one instance."""
        inst = cls._instance
        if inst is None:
            inst = cls._instance = super().__new__(cls)
        return inst

    def __reduce__(self) -> str:
        return "EMPTY"  # unpickles to this module's EMPTY, never a second instance

    def __copy__(self) -> Empty:
        return self

    def __deepcopy__(self, memo: dict) -> Empty:
        return self

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
