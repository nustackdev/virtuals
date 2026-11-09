"""SetView - Set-like view over container."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from virtuals._views._member_key import member_key
from virtuals.collections import ReactiveSetBase
from virtuals.container import ContainerProtocol, ContainerStructure, ContainerTypeError
from virtuals.types import is_empty
from virtuals.view import (
    ChildNestedSetBase,
    ChildObservableBase,
    ChildPrimitiveSetBase,
    KeyedChildrenCountBase,
    ObservableBase,
    UnsafePrimitiveOpsBase,
    ViewBase,
)


if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

    from virtuals.collections import (
        ChildObservable,
        Clearable,
        Containable,
        Convertible,
        Initializable,
        Observable,
        ReactiveSetProtocol,
        Sizeable,
    )


__all__ = ["SetView"]


class SetView(
    ObservableBase,
    ChildObservableBase[object],
    KeyedChildrenCountBase,
    ChildNestedSetBase,
    ChildPrimitiveSetBase,
    UnsafePrimitiveOpsBase,
    ViewBase,
    ReactiveSetBase[object],
):
    """Set-like view over container.

    A ``MutableSet`` using values as keys: ``add``, ``discard``,
    ``remove``, ``pop``, ``clear``, ``in``, ``len``, iteration, comparisons,
    the ``& | - ^`` operators (returning a plain set) and their in-place
    forms, ``isdisjoint``, plus ``issubset`` and ``issuperset`` as on ``set``.

    Implementation:
    - Keys are a hash of the member's canonical pickle (see ``_member_key``)
    - Stores actual values for extraction

    Type Parameters:
        V: Type of values (default: Value)

    Example:
        >>> tags: SetView[str] = SetView(container, registry)
        >>> tags.add("python")
        >>> tags.add("ai")
        >>> print("python" in tags)  # True
        >>> print(len(tags))  # 2
    """

    STRUCTURE: ClassVar[ContainerStructure] = ContainerStructure(4)
    PROTOCOL: ClassVar[ContainerProtocol] = ContainerProtocol.SET | ContainerProtocol.MUTABLE
    CONTAINER_CLS: ClassVar[type] = set

    def _make_key(self, value: object) -> str:
        """Convert value to storage key.

        Args:
            value: Value to store in set

        Returns:
            Key for storage
        """
        return member_key(value)

    @classmethod
    def is_address_static(cls, address: object) -> bool:
        """Set addresses are hashed — never passthrough."""
        return False

    def normalize_address(self, address: object) -> str:
        """Normalize value address to an internal storage key."""
        return self._make_key(address)

    def _check_member(self, value: object) -> str:
        """Reject what a Python set would, and return the member's key.

        EMPTY anywhere, an unhashable value (list, dict, set) and a value
        that does not pickle all raise here, before anything is written,
        so a rejected add leaves nothing behind. A tuple or frozenset is a
        member like any other.
        """
        self._reject_empty_in(value)
        hash(value)  # TypeError for list, dict, set: unhashable type
        return self._make_key(value)  # pickles: raises on a lambda, a lock, a generator

    def _put_member(self, key: str, value: object) -> None:
        """Write one member as a single opaque row under its key.

        Members never become child containers, whatever their Python type:
        a tuple member reads back as a tuple, and iteration stays a
        primitive read per row. Caller must have checked the member and
        ensured the view exists. A new member is counted via
        ``_on_child_created``. A set written before members were opaque
        rows may hold this member as a child container; it is replaced.
        """
        try:
            self._put_child_primitive(key, value)
        except ContainerTypeError:
            self.container.delete_child(key)
            self._decrement_length()  # the row written next counts it again
            self._put_child_primitive(key, value)

    def add(self, value: object) -> None:
        """Add value to set.

        Args:
            value: Value to add. Must be hashable, like a Python set member;
                a list, dict or set raises ``TypeError`` before any write.
        """
        key = self._check_member(value)
        self.ensure_created()
        self._put_member(key, value)

    def remove(self, value: object) -> None:
        """Remove value from set.

        Args:
            value: Value to remove

        Raises:
            KeyError: If value not in set
        """
        hash(value)  # TypeError for an unhashable value, as a Python set raises
        key = self._make_key(value)
        if not self.container.exists_child(key):  # False under an absent set
            raise KeyError(value)
        self.ensure_created()
        self.container.delete_child(key)
        self._decrement_length()

    def discard(self, value: object) -> None:
        """Remove value from set if present.

        Args:
            value: Value to remove
        """
        hash(value)  # TypeError for an unhashable value, as a Python set raises
        key = self._make_key(value)
        if self.container.exists_child(key):  # False under an absent set
            self.ensure_created()
            self.container.delete_child(key)
            self._decrement_length()

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

    def clear(self) -> None:
        """Remove all values."""
        if not self.container.exists():
            return  # clear of nothing is not a write: the view stays absent
        self.ensure_created()
        self.container.clear_children()
        # Reset length metadata
        self._set_length(0)

    def issubset(self, other: Iterable[object]) -> bool:
        """Check if all elements are in other.

        Args:
            other: Set to compare with

        Returns:
            True if subset
        """
        return all(value in other for value in self)

    def issuperset(self, other: Iterable[object]) -> bool:
        """Check if all elements of other are in this set.

        Args:
            other: Set to compare with

        Returns:
            True if superset
        """
        return all(value in self for value in other)

    # =========================================================================
    # VIEW INTERFACE
    # =========================================================================

    def extract(self) -> set[object]:
        """Extract all values as set.

        Returns:
            Set of all values
        """
        return set(self)

    def store(self, value: Iterable[object], *, replace: bool = True) -> None:
        """Store set contents.

        Args:
            value: Iterable to store
            replace: If True, clear existing content first; if False, union
        """
        keyed = [(self._check_member(item), item) for item in value]  # all checked before any write
        self.ensure_created()
        if replace:
            self.clear()

        for key, item in keyed:
            self._put_member(key, item)


if TYPE_CHECKING:
    # Verify protocol implementations
    _convertible: type[Convertible[set[object]]] = SetView
    _initializable: type[Initializable[Iterable[object]]] = SetView
    _containable: type[Containable[object]] = SetView
    _sizeable: type[Sizeable] = SetView
    _clearable: type[Clearable] = SetView
    _reactive_set: type[ReactiveSetProtocol[object]] = SetView
    _Observable: type[Observable] = SetView
    _Observable_children: type[ChildObservable] = SetView
