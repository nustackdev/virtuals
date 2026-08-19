"""Cython-optimized binary key codec."""

from __future__ import annotations

from typing import TYPE_CHECKING

from virtuals_binary_codec import BinaryKeyCodec as _CythonBinaryKeyCodec


if TYPE_CHECKING:
    from ..types import EncodedBinaryKey, Key


class BinaryKeyCodec(_CythonBinaryKeyCodec):  # type: ignore[misc, valid-type]
    """Cython BinaryKeyCodec with a Python-side ``upper_bound_of_prefix``.

    The compiled extension provides ``encode`` / ``decode`` only; the upper-
    bound primitive is added here so both binary variants (Cython and pure
    Python) present the same key-codec surface.
    """

    def upper_bound_of_prefix(self, key: Key) -> EncodedBinaryKey:
        """See :meth:`PyBinaryKeyCodec.upper_bound_of_prefix` — same shape.

        Appends ``0xFF`` to the encoded site: strictly greater than every
        valid segment type marker (``TYPE_INT=0x01``, ``TYPE_STR=0x02``).
        """
        return self.encode(key) + b"\xff"


__all__ = ["BinaryKeyCodec"]
