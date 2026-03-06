"""Copy-on-write state overlay for transactions."""

from __future__ import annotations

from typing import Any


__all__ = ["TransactionState"]


class TransactionState:
    """Copy-on-write state overlay for transactions.

    Provides isolated view over shared parent state without copying.
    Only modified keys are stored locally.
    """

    def __init__(self, parent: dict[str, Any]) -> None:
        """Initialize overlay with parent state reference.

        Args:
            parent: Shared parent state (not copied)
        """
        self._parent = parent
        self._local: dict[str, Any] = {}  # Modified keys
        self._deleted: set[str] = set()  # Deleted keys

    def get(self, key: str) -> object:
        """Get value by key.

        Args:
            key: Key to retrieve

        Returns:
            Value at key

        Raises:
            KeyError: If key not found
        """
        if key in self._deleted:
            raise KeyError(key)
        if key in self._local:
            return self._local[key]
        return self._parent[key]  # Read-through to parent

    def __contains__(self, key: str) -> bool:
        """Check if key exists in overlay.

        Args:
            key: Key to check

        Returns:
            True if key exists and not deleted
        """
        if key in self._deleted:
            return False
        return key in self._local or key in self._parent

    def __setitem__(self, key: str, value: object) -> None:
        """Set key-value pair in overlay.

        Args:
            key: Key to set
            value: Value to store
        """
        self._deleted.discard(key)
        self._local[key] = value

    def __delitem__(self, key: str) -> None:
        """Delete key from overlay.

        Args:
            key: Key to delete
        """
        self._deleted.add(key)
        self._local.pop(key, None)

    def keys(self) -> set[str]:
        """Get all visible keys (parent + local - deleted).

        Returns:
            Set of all visible keys
        """
        all_keys = set(self._parent.keys()) | set(self._local.keys())
        return all_keys - self._deleted

    def to_dict(self) -> dict[str, Any]:
        """Merge overlay to final state dictionary.

        Returns:
            Merged state with all modifications applied
        """
        result = self._parent.copy()
        for k in self._deleted:
            result.pop(k, None)
        result.update(self._local)
        return result
