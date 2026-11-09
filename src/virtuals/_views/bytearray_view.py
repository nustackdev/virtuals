"""ByteArrayView - ByteArray-like view over container."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, cast

from virtuals.collections import ReactiveSequenceBase
from virtuals.container import ContainerProtocol, ContainerStructure
from virtuals.types import is_empty
from virtuals.view import (
    ChildObservableBase,
    ChildPrimitiveSetBase,
    IndexNotFoundError,
    MetadataBasedChildrenCountBase,
    ObservableBase,
    UnsafePrimitiveOpsBase,
    ViewBase,
)

from ._index import refuse_slice, require_int_index


if TYPE_CHECKING:
    from collections.abc import Iterable

    from virtuals.collections import (
        Assignable,
        Convertible,
        Initializable,
        Subscriptable,
    )


__all__ = ["ByteArrayView"]


class ByteArrayView(
    ObservableBase,
    ChildObservableBase[int],
    MetadataBasedChildrenCountBase,
    ChildPrimitiveSetBase,
    UnsafePrimitiveOpsBase,
    ViewBase,
    ReactiveSequenceBase[int],
):
    """ByteArray-like view over container.

    Stores bytes as individual integer children for efficient access.
    Provides bytearray interface with indexing and mutation.

    Example:
        >>> data = ByteArrayView(container, registry)
        >>> data.store(bytearray(b"hello"))
        >>> print(data[0])  # 104 (ord('h'))
        >>> data[0] = 72
        >>> print(data.extract())  # bytearray(b'Hello')
    """

    STRUCTURE: ClassVar[ContainerStructure] = ContainerStructure(6)
    PROTOCOL: ClassVar[ContainerProtocol] = ContainerProtocol.MUTABLE | ContainerProtocol.INDEXED
    CONTAINER_CLS: ClassVar[type] = bytearray

    @classmethod
    def is_address_static(cls, address: object) -> bool:
        """Positive int indices are passthrough; negative need len() lookup."""
        return isinstance(address, int) and address >= 0

    def normalize_address(self, address: int) -> int:
        """Normalize index, handling negative indices.

        Args:
            address: Index to normalize

        Returns:
            Normalized positive index

        Raises:
            TypeError: If ``address`` is not an int, as ``bytearray`` does
            IndexError: If index out of bounds (also on an absent bytearray)
        """
        require_int_index(address, "bytearray")
        length = self._len()

        if address < 0:
            address = length + address

        if address < 0 or address >= length:
            raise IndexNotFoundError("bytearray address (index) out of range")

        return address

    def __getitem__(self, address: int | slice) -> int | bytearray:
        """Get byte at index; a slice gives a plain bytearray, as ``bytearray`` does.

        Args:
            address: Index (supports negative)

        Returns:
            Byte value (0-255)

        Raises:
            IndexError: If index out of bounds
        """
        if isinstance(address, slice):
            return bytearray(self[i] for i in range(*address.indices(len(self))))
        normalized = self.normalize_address(address)
        value = self.container.get_child_primitive(normalized)
        if is_empty(value):
            raise IndexNotFoundError("bytearray index out of range")
        return cast("int", value)

    def __setitem__(self, address: int, value: int) -> None:
        """Set byte at index.

        Args:
            address: Index (supports negative)
            value: Byte value (0-255)

        Raises:
            IndexError: If index out of bounds
            ValueError: If value not in range 0-255
        """
        refuse_slice(address, "assignment", "bytearray")
        _check_byte(value)
        normalized = self.normalize_address(address)  # IndexError before any write
        self.ensure_created()
        self.container.put_child_primitive(normalized, value)

    def __delitem__(self, address: int) -> None:
        """Delete byte at index and shift remaining bytes down.

        O(n) in the number of bytes after ``address``. A slice raises TypeError.
        """
        refuse_slice(address, "deletion", "bytearray")
        normalized = self.normalize_address(address)  # IndexError before any write
        self.ensure_created()
        length = self._len()

        self.container.delete_child(normalized)

        for i in range(normalized + 1, length):
            value = self.container.get_child_primitive(i)
            self.container.put_child_primitive(i - 1, value)
            self.container.delete_child(i)

        self._set_length(length - 1)

    def append(self, value: int) -> None:
        """Append byte to end.

        Args:
            value: Byte value (0-255)

        Raises:
            ValueError: If value not in range 0-255
        """
        _check_byte(value)

        self.ensure_created()
        index = self._len()
        self.container.put_child_primitive(index, value)
        self._set_length(index + 1)

    def insert(self, index: int, value: int) -> None:
        """Insert byte at index, shifting later bytes up.

        O(n) in the number of bytes after ``index``.
        """
        require_int_index(index, "bytearray")
        _check_byte(value)

        self.ensure_created()
        length = self._len()

        if index < 0:
            index = max(0, length + index)
        else:
            index = min(index, length)

        for i in range(length - 1, index - 1, -1):
            byte = self.container.get_child_primitive(i)
            self.container.put_child_primitive(i + 1, byte)

        self.container.put_child_primitive(index, value)
        self._set_length(length + 1)

    def pop(self, index: int = -1) -> int:
        """Remove and return byte at index."""
        if self._len() == 0:
            raise IndexError("pop from empty bytearray")

        value = self[index]
        del self[index]
        return value

    def extend(self, values: Iterable[int]) -> None:
        """Extend with bytes from iterable. Every byte is checked before the first write."""
        batch = list(values)
        for value in batch:
            _check_byte(value)
        for value in batch:
            self.append(value)

    def __contains__(self, value: object) -> bool:
        """``value in view`` as on a bytearray: a byte, or a bytes-like run of bytes.

        Raises TypeError for anything else and ValueError for an int past
        range(0, 256), as ``bytearray`` does.
        """
        return value in bytes(self)

    def index(self, value: object, start: int = 0, stop: int | None = None) -> int:
        """Position of the first byte (or run of bytes) equal to ``value``, as ``bytearray.index``."""
        return bytes(self).index(value, start, stop)  # type: ignore[arg-type]

    def count(self, value: object) -> int:
        """Occurrences of a byte (or non-overlapping runs of bytes), as ``bytearray.count``."""
        return bytes(self).count(value)  # type: ignore[arg-type]

    def remove(self, value: int) -> None:
        """Remove the first byte equal to ``value``. On an absent bytearray, as on an empty one."""
        _check_byte(value)
        if self._len() == 0:
            raise ValueError(f"{value!r} is not in bytearray")
        super().remove(value)

    def clear(self) -> None:
        """Remove all bytes."""
        if not self.container.exists():
            return  # clear of nothing is not a write: the view stays absent
        self.ensure_created()
        self.container.clear_children()
        self._set_length(0)

    def __eq__(self, other: object) -> bool:
        """Equal to bytes, a bytearray or a bytearray view with the same bytes, as ``bytearray ==``."""
        if not isinstance(other, bytes | bytearray | ByteArrayView):
            return NotImplemented
        return bytes(self) == bytes(other)

    __hash__ = None  # type: ignore[assignment]

    def extract(self) -> bytearray:
        """Extract all bytes as bytearray.

        Returns:
            Bytearray of all bytes
        """
        return bytearray(self)

    def store(self, value: Iterable[int], *, replace: bool = True) -> None:
        """Store bytearray contents.

        Args:
            value: Bytes or bytearray to store
            replace: If True, clear existing content first; if False, append
        """
        self.ensure_created()
        if replace:
            self.clear()
            count = 0
        else:
            count = self._len()
        for byte in value:
            self.container.put_child_primitive(count, byte)
            count += 1
        self._set_length(count)


def _check_byte(value: object) -> None:
    """Refuse what ``bytearray`` refuses: TypeError for a non-int, ValueError past range(0, 256)."""
    if not isinstance(value, int):
        raise TypeError(f"'{type(value).__name__}' object cannot be interpreted as an integer")
    if not 0 <= value <= 255:
        raise ValueError("byte must be in range(0, 256)")


if TYPE_CHECKING:
    _subscriptable: type[Subscriptable[int, int]] = ByteArrayView
    _convertible: type[Convertible[bytearray]] = ByteArrayView
    _initializable: type[Initializable[Iterable[int]]] = ByteArrayView
    _assignable: type[Assignable[int, int]] = ByteArrayView
    _watchable: type[ObservableBase] = ByteArrayView
    _watchable_children: type[ChildObservableBase] = ByteArrayView
