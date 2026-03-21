"""FlatDictView - Flat dict view with length tracking but no nesting.

Lightweight dict view with:
- Length tracking via metadata
- Primitives only (no nested containers)
- No observables
- No functional operations
"""

from __future__ import annotations

from collections.abc import ItemsView, KeysView, MutableMapping, ValuesView
from typing import TYPE_CHECKING, ClassVar, cast

from virtuals.container import ContainerProtocol, ContainerStructure
from virtuals.types import EMPTY, Empty, Value, is_empty
from virtuals.view import (
    ChildPrimitiveSetBase,
    MetadataBasedChildrenCountBase,
    UnsafePrimitiveOpsBase,
    ViewBase,
)


if TYPE_CHECKING:
    from collections.abc import Generator
    from collections.abc import Mapping as PyMapping


__all__ = [
    "FlatDictView",
]


class FlatDictView(
    MetadataBasedChildrenCountBase,
    ChildPrimitiveSetBase,
    UnsafePrimitiveOpsBase,
    ViewBase,
):
    """Flat dict view with length tracking but no nested containers.

    Stores only primitive values (int, float, str, bool, bytes, None).
    Includes __len__ support via metadata tracking.

    Use when:
    - You need len() support
    - You only store primitives
    - You don't need nested containers or observables

    Example:
        >>> scores = nav.root(tx)
        >>> scores["alice"] = 100
        >>> scores["bob"] = 95
        >>> print(len(scores))  # 2
    """

    STRUCTURE: ClassVar[ContainerStructure] = ContainerStructure(12)  # New structure ID
    PROTOCOL: ClassVar[ContainerProtocol] = ContainerProtocol.MAPPING | ContainerProtocol.MUTABLE
    CONTAINER_CLS: ClassVar[type] = dict

    @classmethod
    def is_address_static(cls, address: object) -> bool:
        """Dict keys are always passthrough — no normalization needed."""
        return True

    def normalize_address(self, address: str | int) -> str | int:
        """Passthrough - no normalization needed."""
        return address

    def __getitem__(self, key: str | int) -> Value:
        """Get primitive value for key.

        Args:
            key: Key to retrieve

        Returns:
            Primitive value

        Raises:
            KeyError: If key not found
        """
        value = self.container.get_child_primitive(key)
        if isinstance(value, Empty):
            raise KeyError(key)
        return value

    def __setitem__(self, key: str | int, value: Value) -> None:
        """Set primitive value for key.

        Args:
            key: Key to set
            value: Primitive value to store
        """
        self.ensure_created()
        is_new = not self.container.exists_child(key)
        self.container.put_child_primitive(key, value)
        if is_new:
            self._increment_length()

    def __delitem__(self, key: str | int) -> None:
        """Delete key.

        Args:
            key: Key to delete

        Raises:
            KeyError: If key not found
        """
        self.ensure_created()
        self.container.delete_child(key)
        self._update_count()

    def __contains__(self, key: str | int) -> bool:
        """Check if key exists."""
        return self.container.exists_child(key)

    def __iter__(self) -> Generator[str | int, None, None]:
        """Iterate over keys."""
        yield from self.container.iter_child_keys(validate=False)

    def get(self, key: str | int, default: Value | Empty = EMPTY) -> Value | Empty:
        """Get value with default fallback."""
        try:
            return self[key]
        except KeyError:
            return default

    def pop(self, key: str | int, default: Value | Empty = EMPTY) -> Value | Empty:
        """Remove and return value."""
        try:
            value = self[key]
            del self[key]
            return value
        except KeyError:
            if is_empty(default):
                raise
            return default

    def keys(self) -> KeysView[str | int]:
        """Get all keys as a set-like view."""
        return KeysView(self)  # type: ignore[arg-type]

    def values(self) -> ValuesView[Value]:
        """Get all values as a collection view."""
        return _FlatValuesView(self)

    def items(self) -> ItemsView[str | int, Value]:
        """Get all items as a set-like view."""
        return _FlatItemsView(self)

    def clear(self) -> None:
        """Remove all items."""
        self.ensure_created()
        self.container.clear_children(validate=False)
        self._set_length(0)

    def update(self, other: PyMapping[str | int, Value] | None = None, **kwargs: Value) -> None:
        """Update from dict or kwargs."""
        if other:
            for key, value in other.items():
                self[key] = value
        for key, value in kwargs.items():
            self[key] = value  # type: ignore[index]

    def extract(self) -> dict[str | int, Value]:
        """Extract all items as dict."""
        return dict(self.items())

    def store(self, value: PyMapping[str | int, Value], *, replace: bool = True) -> None:
        """Store dict contents.

        Args:
            value: Mapping to store
            replace: If True, clear existing content first
        """
        self.ensure_created()
        if replace:
            self.clear()

        count = 0
        for key, val in value.items():
            self.container.put_child_primitive(key, val)
            count += 1

        if replace:
            self._set_length(count)
        else:
            # When appending, adjust length for actually new keys
            self._update_count()


class _FlatValuesView(ValuesView):
    """Efficient ValuesView for FlatDictView — single-pass iteration."""

    _mapping: FlatDictView

    def __iter__(self) -> Generator[Value, None, None]:  # type: ignore[override]
        for _k, v in self._mapping.container.iter_children(validate=False):
            yield cast("Value", v.primitive_value)


class _FlatItemsView(ItemsView):
    """Efficient ItemsView for FlatDictView — single-pass iteration."""

    _mapping: FlatDictView

    def __iter__(self) -> Generator[tuple[str | int, Value], None, None]:  # type: ignore[override]
        for k, v in self._mapping.container.iter_children(validate=False):
            yield k, cast("Value", v.primitive_value)


MutableMapping.register(FlatDictView)
