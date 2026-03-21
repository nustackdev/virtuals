"""LightDictView - Minimal dict view stripped of all overhead.

Ultra-lightweight dict view with no:
- Length tracking
- Observable support
- Nested views support (primitives only)
- Functional operations
- Child navigation
"""

from __future__ import annotations

from collections.abc import ItemsView, KeysView, MutableMapping, ValuesView
from typing import TYPE_CHECKING, ClassVar, cast

from virtuals.container import ContainerProtocol, ContainerStructure
from virtuals.types import EMPTY, Empty, Value
from virtuals.view import ChildPrimitiveSetBase, UnsafePrimitiveOpsBase, ViewBase


if TYPE_CHECKING:
    from collections.abc import Generator
    from collections.abc import Mapping as PyMapping


__all__ = [
    "LightDictView",
]


class LightDictView(
    ChildPrimitiveSetBase,
    UnsafePrimitiveOpsBase,
    ViewBase,
):
    """Ultra-lightweight dict view for maximum performance.

    Stores only primitive values (int, float, str, bool, bytes, None).
    No nested containers, no observables, no length tracking.

    Use when:
    - You need maximum set/get performance
    - You only store primitives
    - You don't need len() or iteration often

    Example:
        >>> cache = nav.root(tx)
        >>> cache["key"] = "value"
        >>> print(cache["key"])
    """

    STRUCTURE: ClassVar[ContainerStructure] = ContainerStructure(11)  # New structure ID
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
        self.container.put_child_primitive(key, value)

    def __delitem__(self, key: str | int) -> None:
        """Delete key.

        Args:
            key: Key to delete

        Raises:
            KeyError: If key not found
        """
        self.ensure_created()
        self.container.delete_child(key)

    def __contains__(self, key: str | int) -> bool:
        """Check if key exists."""
        return self.container.exists_child(key)

    def __len__(self) -> int:
        """Count number of keys."""
        return sum(1 for _ in self.keys())

    def __iter__(self) -> Generator[str | int, None, None]:
        """Iterate over keys."""
        yield from self.container.iter_child_keys(validate=False)

    def get(self, key: str | int, default: Value | Empty = EMPTY) -> Value | Empty:
        """Get value with default fallback."""
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self) -> KeysView[str | int]:
        """Get all keys as a set-like view."""
        return KeysView(self)  # type: ignore[arg-type]

    def values(self) -> ValuesView[Value]:
        """Get all values as a collection view."""
        return _LightValuesView(self)

    def items(self) -> ItemsView[str | int, Value]:
        """Get all items as a set-like view."""
        return _LightItemsView(self)

    def clear(self) -> None:
        """Remove all items."""
        self.ensure_created()
        self.container.clear_children(validate=False)

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
        for key, val in value.items():
            self[key] = val


class _LightValuesView(ValuesView):
    """Efficient ValuesView for LightDictView — single-pass iteration."""

    _mapping: LightDictView

    def __iter__(self) -> Generator[Value, None, None]:  # type: ignore[override]
        for _k, v in self._mapping.container.iter_children(validate=False):
            yield cast("Value", v.primitive_value)


class _LightItemsView(ItemsView):
    """Efficient ItemsView for LightDictView — single-pass iteration."""

    _mapping: LightDictView

    def __iter__(self) -> Generator[tuple[str | int, Value], None, None]:  # type: ignore[override]
        for k, v in self._mapping.container.iter_children(validate=False):
            yield k, cast("Value", v.primitive_value)


MutableMapping.register(LightDictView)
