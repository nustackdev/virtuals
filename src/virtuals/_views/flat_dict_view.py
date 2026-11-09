"""FlatDictView - Flat dict view, no nesting.

Lightweight dict view with:
- Live length (no stored counter, so no key shared between writers)
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
    LiveChildrenCountBase,
    UnsafePrimitiveOpsBase,
    ViewBase,
    is_view_key,
)

from ._keys_view import ReversibleKeysView


if TYPE_CHECKING:
    from collections.abc import Generator
    from collections.abc import Mapping as PyMapping


__all__ = [
    "FlatDictView",
]


class FlatDictView(
    LiveChildrenCountBase,
    ChildPrimitiveSetBase,
    UnsafePrimitiveOpsBase,
    ViewBase,
):
    """Flat dict view with no nested containers.

    Stores only primitive values (int, float, str, bool, bytes, None).
    ``len`` counts the keys live; every write touches a constant number
    of keys.

    Use when:
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
            KeyError: If key not found, or a key the view cannot hold
        """
        if not is_view_key(key):
            raise KeyError(key)
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
        self._reject_empty(key, value)
        self.ensure_created()
        self._put_child_primitive(key, value)

    def __delitem__(self, key: str | int) -> None:
        """Delete key. KeyError if the key is missing, like a dict; an absent view stays absent.

        Args:
            key: Key to delete
        """
        if not is_view_key(key) or not self.container.exists_child(key):
            raise KeyError(key)
        self.container.delete_child(key)

    def __contains__(self, key: str | int) -> bool:
        """Check if key exists. A key the view cannot hold is not in it."""
        return is_view_key(key) and self.container.exists_child(key, strict=True)

    def __iter__(self) -> Generator[str | int, None, None]:
        """Iterate over keys."""
        yield from self.container.iter_child_keys(validate=False, strict=True)

    def __reversed__(self) -> Generator[str | int, None, None]:
        """Keys last first: one reverse scan."""
        yield from self.container.iter_child_keys(validate=False, strict=True, reverse=True)

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
        return ReversibleKeysView(self)

    def values(self) -> ValuesView[Value]:
        """Get all values as a collection view."""
        return _FlatValuesView(self)

    def items(self) -> ItemsView[str | int, Value]:
        """Get all items as a set-like view."""
        return _FlatItemsView(self)

    def clear(self) -> None:
        """Remove all items."""
        if not self.container.exists():
            return  # clear of nothing is not a write: the view stays absent
        self.ensure_created()
        self.container.clear_children(validate=False)

    def update(self, other: PyMapping[str | int, Value] | None = None, **kwargs: Value) -> None:
        """Update from dict or kwargs."""
        self._reject_empty_in(other or {})
        self._reject_empty_in(kwargs)
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
        value = self._reject_empty_in(value)  # type: ignore[assignment]
        self.ensure_created()
        if replace:
            self.clear()
        for key, val in value.items():
            self._put_child_primitive(key, val)

    def _rebuild_bookkeeping(self) -> None:
        """Drop the stored counter older versions kept; length is live now."""
        self.container.delete_metadata("__len__")


class _FlatValuesView(ValuesView):
    """Efficient ValuesView for FlatDictView — single-pass iteration."""

    _mapping: FlatDictView

    def __iter__(self) -> Generator[Value, None, None]:  # type: ignore[override]
        for _k, v in self._mapping.container.iter_children(validate=False, strict=True):
            yield cast("Value", v.primitive_value)

    def __reversed__(self) -> Generator[Value, None, None]:
        for _k, v in self._mapping.container.iter_children(
            validate=False, strict=True, reverse=True
        ):
            yield cast("Value", v.primitive_value)


class _FlatItemsView(ItemsView):
    """Efficient ItemsView for FlatDictView — single-pass iteration."""

    _mapping: FlatDictView

    def __iter__(self) -> Generator[tuple[str | int, Value], None, None]:  # type: ignore[override]
        for k, v in self._mapping.container.iter_children(validate=False, strict=True):
            yield k, cast("Value", v.primitive_value)

    def __reversed__(self) -> Generator[tuple[str | int, Value], None, None]:
        for k, v in self._mapping.container.iter_children(
            validate=False, strict=True, reverse=True
        ):
            yield k, cast("Value", v.primitive_value)


MutableMapping.register(FlatDictView)
