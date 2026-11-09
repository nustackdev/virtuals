"""LightDictView - Minimal dict view stripped of all overhead.

Ultra-lightweight dict view with no:
- Length tracking
- Nested views support (primitives only)
- Functional operations
- Child navigation

Observable like ``DictView``: the whole view, one key, the keys, a pattern.
"""

from __future__ import annotations

from collections.abc import ItemsView, KeysView, ValuesView
from typing import TYPE_CHECKING, ClassVar, cast

from virtuals.collections import DictMethods, ReactiveMappingBase
from virtuals.container import ContainerProtocol, ContainerStructure
from virtuals.types import Empty, Value
from virtuals.view import (
    ChildObservableBase,
    ChildPrimitiveSetBase,
    DescendantsObservableBase,
    KeyNotFoundError,
    ObservableBase,
    UnsafePrimitiveOpsBase,
    ViewBase,
    is_view_key,
)

from ._keys_view import ReversibleKeysView


if TYPE_CHECKING:
    from collections.abc import Generator
    from collections.abc import Mapping as PyMapping


__all__ = [
    "LightDictView",
]


class LightDictView(
    ObservableBase,
    ChildObservableBase[str | int],
    DescendantsObservableBase,
    ChildPrimitiveSetBase,
    UnsafePrimitiveOpsBase,
    ViewBase,
    ReactiveMappingBase[str | int, Value],
    DictMethods[str | int, Value],
):
    """Ultra-lightweight dict view for maximum performance.

    Stores only primitive values (int, float, str, bool, bytes, None).
    No nested containers, no length tracking.

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
            KeyError: If key not found, or a key the view cannot hold
        """
        if not is_view_key(key):
            raise KeyNotFoundError(key)
        value = self.container.get_child_primitive(key)
        if isinstance(value, Empty):
            raise KeyNotFoundError(key)
        return value

    def __setitem__(self, key: str | int, value: Value) -> None:
        """Set primitive value for key.

        Args:
            key: Key to set
            value: Primitive value to store
        """
        self._reject_empty(key, value)
        self.ensure_created()
        self.container.put_child_primitive(key, value)

    def __delitem__(self, key: str | int) -> None:
        """Delete key. KeyError if the key is missing, like a dict; an absent view stays absent.

        Args:
            key: Key to delete
        """
        if not is_view_key(key) or not self.container.exists_child(key):
            raise KeyNotFoundError(key)
        self.container.delete_child(key)

    def __contains__(self, key: str | int) -> bool:
        """Check if key exists. A key the view cannot hold is not in it."""
        return is_view_key(key) and self.container.exists_child(key, strict=True)

    def __len__(self) -> int:
        """Count number of keys."""
        return self.container.count_children(strict=True)

    def __bool__(self) -> bool:
        """True if any key exists. One row, so ``if view:`` never counts."""
        return self.container.has_children(strict=True)

    def __iter__(self) -> Generator[str | int, None, None]:
        """Iterate over keys."""
        yield from self.container.iter_child_keys(validate=False, strict=True)

    def __reversed__(self) -> Generator[str | int, None, None]:
        """Keys last first: one reverse scan."""
        yield from self.container.iter_child_keys(validate=False, strict=True, reverse=True)

    def keys(
        self, *, after: str | int | None = None, limit: int | None = None
    ) -> KeysView[str | int] | Generator[str | int, None, None]:
        """Keys in key order, or one page of them, as ``DictView.keys``.

        With no arguments: a set-like KeysView. With ``after`` or ``limit``:
        a generator over the keys strictly after ``after`` (a position in
        key order, it need not exist), at most ``limit`` of them. Cost: one
        seek plus O(limit) rows.
        """
        if after is None and limit is None:
            return ReversibleKeysView(self)
        return self._keys_page(cursor=after, limit=limit, reverse=False)

    def reversed_keys(
        self, *, before: str | int | None = None, limit: int | None = None
    ) -> Generator[str | int, None, None]:
        """Keys in reverse key order, strictly before ``before``, at most ``limit``."""
        return self._keys_page(cursor=before, limit=limit, reverse=True)

    def _keys_page(
        self, *, cursor: str | int | None, limit: int | None, reverse: bool
    ) -> Generator[str | int, None, None]:
        """One page of keys. A bad cursor raises at the call, not on the first read."""
        if cursor is not None and not is_view_key(cursor):
            raise TypeError(f"view keys must be str or int, got {type(cursor).__name__}")
        return self.container.iter_child_keys(
            validate=False, strict=True, reverse=reverse, cursor=cursor, limit=limit
        )

    def values(self) -> ValuesView[Value]:
        """Get all values as a collection view."""
        return _LightValuesView(self)

    def items(self) -> ItemsView[str | int, Value]:
        """Get all items as a set-like view."""
        return _LightItemsView(self)

    def clear(self) -> None:
        """Remove all items."""
        if not self.container.exists():
            return  # clear of nothing is not a write: the view stays absent
        self.ensure_created()
        self.container.clear_children(validate=False)

    def update(self, other: object = (), /, **kwargs: Value) -> None:
        """Update as ``dict.update``. The whole batch is checked for EMPTY before the first write."""
        items = self._update_items(other, kwargs)
        self._reject_empty_in(items)
        for key, value in items.items():
            self[key] = value

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
            self[key] = val


class _LightValuesView(ValuesView):
    """Efficient ValuesView for LightDictView — single-pass iteration."""

    _mapping: LightDictView

    def __iter__(self) -> Generator[Value, None, None]:  # type: ignore[override]
        for _k, v in self._mapping.container.iter_children(validate=False, strict=True):
            yield cast("Value", v.primitive_value)

    def __reversed__(self) -> Generator[Value, None, None]:
        for _k, v in self._mapping.container.iter_children(
            validate=False, strict=True, reverse=True
        ):
            yield cast("Value", v.primitive_value)


class _LightItemsView(ItemsView):
    """Efficient ItemsView for LightDictView — single-pass iteration."""

    _mapping: LightDictView

    def __iter__(self) -> Generator[tuple[str | int, Value], None, None]:  # type: ignore[override]
        for k, v in self._mapping.container.iter_children(validate=False, strict=True):
            yield k, cast("Value", v.primitive_value)

    def __reversed__(self) -> Generator[tuple[str | int, Value], None, None]:
        for k, v in self._mapping.container.iter_children(
            validate=False, strict=True, reverse=True
        ):
            yield k, cast("Value", v.primitive_value)
