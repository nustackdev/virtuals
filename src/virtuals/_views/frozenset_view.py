"""FrozenSetView - Frozenset-like view over container (immutable set)."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from virtuals._views._member_key import member_key
from virtuals.collections import FrozenSetMethods
from virtuals.container import ContainerProtocol, ContainerStructure, ContainerTypeError
from virtuals.types import is_empty
from virtuals.view import (
    ChildNestedSetBase,
    ChildPrimitiveSetBase,
    KeyedChildrenCountBase,
    UnsafePrimitiveOpsBase,
    ViewBase,
)


if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

    from virtuals.collections import (
        Containable,
        Convertible,
        Initializable,
        Sizeable,
    )
    from virtuals.loc import key as key_


__all__ = ["FrozenSetView"]


class FrozenSetView(
    KeyedChildrenCountBase,
    ChildNestedSetBase,
    ChildPrimitiveSetBase,
    UnsafePrimitiveOpsBase,
    ViewBase,
    FrozenSetMethods[object],
):
    """Frozenset-like view over container (immutable set).

    Every ``frozenset`` method with the same semantics; new values
    (operators, ``union``, ``copy``, ...) come back as a plain frozenset.
    Not hashable, unlike ``frozenset``: the stored members can be replaced
    under the view, so a hash would not hold for the view's lifetime.

    Type Parameters:
        V: Type of values (default: Value)

    Example:
        >>> perms: FrozenSetView[str] = FrozenSetView(container, registry)
        >>> # Must be initialized via store()
        >>> perms.store({"read", "write", "execute"})
        >>> print("read" in perms)  # True
        >>> print(len(perms))  # 3
    """

    STRUCTURE: ClassVar[ContainerStructure] = ContainerStructure(5)
    PROTOCOL: ClassVar[ContainerProtocol] = ContainerProtocol.SET
    CONTAINER_CLS: ClassVar[type] = frozenset

    @classmethod
    def _from_iterable(cls, it: Iterable[object]) -> frozenset[object]:
        """Operators return a plain frozenset, as ``frozenset`` operators do."""
        return frozenset(it)

    def _make_key(self, value: object) -> key_.KeySegment:
        """Convert value to storage key.

        Deterministic hash for any hashable object.

        Args:
            value: Value to store in set

        Returns:
            Key for storage
        """
        return member_key(value)

    @classmethod
    def is_address_static(cls, address: object) -> bool:
        """Frozenset addresses are hashed — never passthrough."""
        return False

    def normalize_address(self, address: object) -> key_.KeySegment:
        """Normalize value address to an internal storage key."""
        return self._make_key(address)

    def __contains__(self, obj: object) -> bool:
        """Check if value in set.

        Args:
            obj: Value to check

        Returns:
            True if value in set
        """
        hash(obj)  # TypeError for an unhashable value, as a Python set raises
        key = self._make_key(obj)
        return self.container.exists_child(key, strict=True)

    def __iter__(self) -> Generator[object, None, None]:
        """Iterate over values.

        Yields:
            Values in set
        """
        for key in self.container.iter_child_keys(strict=True):
            stored_value = self.container.get_child_primitive(key)
            if not is_empty(stored_value):
                yield stored_value

    def extract(self) -> frozenset[object]:
        """Extract all values as frozenset.

        Returns:
            Frozenset of all values
        """
        return frozenset(self)

    def store(self, value: Iterable[object], *, replace: bool = True) -> None:
        """Store frozenset contents.

        Args:
            value: Iterable to store
            replace: If True, clear existing content first; if False, union
        """
        keyed = []
        for item in value:  # every member checked and keyed before any write
            self._reject_empty_in(item)
            hash(item)  # TypeError for list, dict, set: unhashable type
            keyed.append((self._make_key(item), item))  # pickles: raises on a lambda or a lock
        self.ensure_created()
        if replace:
            self.container.clear_children()
            self._set_length(0)

        # One opaque row per member, never a child container (a tuple member
        # reads back as a tuple). A new member is counted via _on_child_created.
        # A member that an older write left as a child container is replaced.
        for key, item in keyed:
            try:
                self._put_child_primitive(key, item)
            except ContainerTypeError:
                self.container.delete_child(key)
                self._decrement_length()
                self._put_child_primitive(key, item)


if TYPE_CHECKING:
    # Verify protocol implementations
    _convertible: type[Convertible[object]] = FrozenSetView
    _initializable: type[Initializable[Iterable[object]]] = FrozenSetView
    _containable: type[Containable[object]] = FrozenSetView
    _sizeable: type[Sizeable] = FrozenSetView
    pass
