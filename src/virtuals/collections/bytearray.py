"""What ``bytearray`` has beyond ``MutableSequence``.

``ByteArrayMethods`` is the ``bytearray`` counterpart of ``ListMethods``:
``copy``, ``+``, ``*``, ``%``, ordering, the slice writes, and the bytes
methods (``find``, ``split``, ``upper``, ``decode``, ...). A method that
returns a new value reads the bytes whole and answers exactly as
``bytearray`` does, with a plain ``bytearray`` (or ``bytes``, ``str``,
``list``, ``tuple``, as the method on ``bytearray`` gives).

Not here, because a Python class cannot provide them honestly:

- the buffer protocol (``memoryview(view)``, a view handed to a C API that
  takes a bytes-like object, such as ``hashlib`` or ``file.write``). Below
  Python 3.12 a Python class cannot expose a buffer at all, and a buffer
  over a copy would let writes through a ``memoryview`` vanish. ``bytes(view)``
  gives a plain copy to hand on; ``__bytes__`` makes ``b"%b" % view`` work.
- ``fromhex``: it builds a new bytearray from nothing, and a view cannot
  exist without a place in storage to stand at.
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, SupportsIndex

from .sequence import MutableSequenceBase


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


__all__ = ["ByteArrayMethods"]

BytesLike = bytes | bytearray | memoryview


class ByteArrayMethods(MutableSequenceBase[int]):
    """What ``bytearray`` has beyond ``MutableSequence``: a view built on it is a bytearray.

    ``+``, ``+=`` and the comparisons take bytes, a bytearray, a memoryview
    or another bytearray view, as ``bytearray`` does; a list, say, gives
    ``TypeError``. ``+=`` and the slice writes change the view in place;
    every other method leaves it as is and returns a plain value.
    """

    def _bytes(self) -> bytearray:
        """The bytes as a plain bytearray, read whole."""
        return bytearray(self)

    def _plain_values(self) -> bytearray:  # type: ignore[override]
        """The bytes as a plain bytearray: slice writes follow ``bytearray`` rules."""
        return self._bytes()

    @staticmethod
    def _operand(other: object) -> BytesLike | None:
        """``other`` as something ``bytearray`` takes in ``+`` or a comparison, or None."""
        if isinstance(other, ByteArrayMethods):
            return other._bytes()
        if isinstance(other, bytes | bytearray | memoryview):
            return other
        return None

    # -- operators ----------------------------------------------------------

    def __add__(self, other: object) -> bytearray:
        operand = self._operand(other)
        return NotImplemented if operand is None else self._bytes() + operand

    def __radd__(self, other: object) -> bytes | bytearray:
        if not isinstance(other, bytes | bytearray):
            return NotImplemented
        return other + self._bytes()  # bytes + bytearray is bytes, as on the built-ins

    def __iadd__(self, other: object) -> ByteArrayMethods:  # type: ignore[override]
        operand = self._operand(other)
        if operand is None:
            return NotImplemented
        self.extend(bytes(operand))
        return self

    def __mul__(self, count: object) -> bytearray:
        try:
            n = operator.index(count)  # type: ignore[arg-type]
        except TypeError:
            return NotImplemented
        return self._bytes() * n

    __rmul__ = __mul__

    def __imul__(self, count: object) -> ByteArrayMethods:
        try:
            n = operator.index(count)  # type: ignore[arg-type]
        except TypeError:
            return NotImplemented
        if n <= 0:
            self.clear()
        elif n > 1:
            self.extend(self._bytes() * (n - 1))
        return self

    def __mod__(self, args: object) -> bytearray:
        return self._bytes() % args

    def __rmod__(self, other: object) -> object:
        return self._bytes().__rmod__(other)

    def __bytes__(self) -> bytes:
        return bytes(self._bytes())

    def _cmp(self, other: object, op: Callable[[object, object], bool]) -> bool:
        operand = self._operand(other)
        return NotImplemented if operand is None else op(self._bytes(), operand)

    def __eq__(self, other: object) -> bool:
        return self._cmp(other, operator.eq)

    def __lt__(self, other: object) -> bool:
        return self._cmp(other, operator.lt)

    def __le__(self, other: object) -> bool:
        return self._cmp(other, operator.le)

    def __gt__(self, other: object) -> bool:
        return self._cmp(other, operator.gt)

    def __ge__(self, other: object) -> bool:
        return self._cmp(other, operator.ge)

    __hash__ = None  # type: ignore[assignment]

    # -- copy and search ----------------------------------------------------

    def copy(self) -> bytearray:
        """A plain bytearray of the bytes, as ``bytearray.copy``."""
        return self._bytes()

    def find(
        self,
        sub: BytesLike | int,
        start: SupportsIndex | None = None,
        end: SupportsIndex | None = None,
    ) -> int:
        """As ``bytearray.find``."""
        return self._bytes().find(sub, start, end)

    def rfind(
        self,
        sub: BytesLike | int,
        start: SupportsIndex | None = None,
        end: SupportsIndex | None = None,
    ) -> int:
        """As ``bytearray.rfind``."""
        return self._bytes().rfind(sub, start, end)

    def rindex(
        self,
        sub: BytesLike | int,
        start: SupportsIndex | None = None,
        end: SupportsIndex | None = None,
    ) -> int:
        """As ``bytearray.rindex``."""
        return self._bytes().rindex(sub, start, end)

    def startswith(
        self,
        prefix: BytesLike | tuple[BytesLike, ...],
        start: SupportsIndex | None = None,
        end: SupportsIndex | None = None,
    ) -> bool:
        """As ``bytearray.startswith``."""
        return self._bytes().startswith(prefix, start, end)

    def endswith(
        self,
        suffix: BytesLike | tuple[BytesLike, ...],
        start: SupportsIndex | None = None,
        end: SupportsIndex | None = None,
    ) -> bool:
        """As ``bytearray.endswith``."""
        return self._bytes().endswith(suffix, start, end)

    # -- conversion ---------------------------------------------------------

    def decode(self, encoding: str = "utf-8", errors: str = "strict") -> str:
        """As ``bytearray.decode``."""
        return self._bytes().decode(encoding, errors)

    def hex(self, *args: str | bytes | int) -> str:
        """As ``bytearray.hex``: ``hex()``, ``hex(sep)``, ``hex(sep, bytes_per_sep)``."""
        return self._bytes().hex(*args)  # type: ignore[arg-type]

    maketrans = staticmethod(bytearray.maketrans)

    def translate(self, table: BytesLike | None, /, delete: bytes = b"") -> bytearray:
        """As ``bytearray.translate``."""
        return self._bytes().translate(table, delete)

    # -- split and join -----------------------------------------------------

    def join(self, iterable_of_bytes: Iterable[BytesLike], /) -> bytearray:
        """As ``bytearray.join``."""
        return self._bytes().join(iterable_of_bytes)

    def split(self, sep: BytesLike | None = None, maxsplit: SupportsIndex = -1) -> list[bytearray]:
        """As ``bytearray.split``."""
        return self._bytes().split(sep, maxsplit)

    def rsplit(self, sep: BytesLike | None = None, maxsplit: SupportsIndex = -1) -> list[bytearray]:
        """As ``bytearray.rsplit``."""
        return self._bytes().rsplit(sep, maxsplit)

    def splitlines(self, keepends: bool = False) -> list[bytearray]:
        """As ``bytearray.splitlines``."""
        return self._bytes().splitlines(keepends)

    def partition(self, sep: BytesLike, /) -> tuple[bytearray, bytearray, bytearray]:
        """As ``bytearray.partition``."""
        return self._bytes().partition(sep)

    def rpartition(self, sep: BytesLike, /) -> tuple[bytearray, bytearray, bytearray]:
        """As ``bytearray.rpartition``."""
        return self._bytes().rpartition(sep)

    # -- new bytearrays -----------------------------------------------------

    def replace(self, old: BytesLike, new: BytesLike, count: SupportsIndex = -1, /) -> bytearray:
        """As ``bytearray.replace``."""
        return self._bytes().replace(old, new, count)

    def removeprefix(self, prefix: BytesLike, /) -> bytearray:
        """As ``bytearray.removeprefix``."""
        return self._bytes().removeprefix(prefix)

    def removesuffix(self, suffix: BytesLike, /) -> bytearray:
        """As ``bytearray.removesuffix``."""
        return self._bytes().removesuffix(suffix)

    def strip(self, chars: BytesLike | None = None, /) -> bytearray:
        """As ``bytearray.strip``."""
        return self._bytes().strip(chars)

    def lstrip(self, chars: BytesLike | None = None, /) -> bytearray:
        """As ``bytearray.lstrip``."""
        return self._bytes().lstrip(chars)

    def rstrip(self, chars: BytesLike | None = None, /) -> bytearray:
        """As ``bytearray.rstrip``."""
        return self._bytes().rstrip(chars)

    def center(self, width: SupportsIndex, fillchar: bytes = b" ", /) -> bytearray:
        """As ``bytearray.center``."""
        return self._bytes().center(width, fillchar)

    def ljust(self, width: SupportsIndex, fillchar: bytes = b" ", /) -> bytearray:
        """As ``bytearray.ljust``."""
        return self._bytes().ljust(width, fillchar)

    def rjust(self, width: SupportsIndex, fillchar: bytes = b" ", /) -> bytearray:
        """As ``bytearray.rjust``."""
        return self._bytes().rjust(width, fillchar)

    def zfill(self, width: SupportsIndex, /) -> bytearray:
        """As ``bytearray.zfill``."""
        return self._bytes().zfill(width)

    def expandtabs(self, tabsize: SupportsIndex = 8) -> bytearray:
        """As ``bytearray.expandtabs``."""
        return self._bytes().expandtabs(tabsize)

    def capitalize(self) -> bytearray:
        """As ``bytearray.capitalize``."""
        return self._bytes().capitalize()

    def lower(self) -> bytearray:
        """As ``bytearray.lower``."""
        return self._bytes().lower()

    def upper(self) -> bytearray:
        """As ``bytearray.upper``."""
        return self._bytes().upper()

    def swapcase(self) -> bytearray:
        """As ``bytearray.swapcase``."""
        return self._bytes().swapcase()

    def title(self) -> bytearray:
        """As ``bytearray.title``."""
        return self._bytes().title()

    # -- predicates ---------------------------------------------------------

    def isalnum(self) -> bool:
        """As ``bytearray.isalnum``."""
        return self._bytes().isalnum()

    def isalpha(self) -> bool:
        """As ``bytearray.isalpha``."""
        return self._bytes().isalpha()

    def isascii(self) -> bool:
        """As ``bytearray.isascii``."""
        return self._bytes().isascii()

    def isdigit(self) -> bool:
        """As ``bytearray.isdigit``."""
        return self._bytes().isdigit()

    def islower(self) -> bool:
        """As ``bytearray.islower``."""
        return self._bytes().islower()

    def isspace(self) -> bool:
        """As ``bytearray.isspace``."""
        return self._bytes().isspace()

    def istitle(self) -> bool:
        """As ``bytearray.istitle``."""
        return self._bytes().istitle()

    def isupper(self) -> bool:
        """As ``bytearray.isupper``."""
        return self._bytes().isupper()
