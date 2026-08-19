"""Type stubs for binary_codec Cython extension."""

__all__ = ["BinaryKeyCodec"]

class BinaryKeyCodec:
    """Binary key codec that preserves lexicographic ordering."""

    def encode(self, key: tuple[str | int, ...]) -> bytes:
        """Encode tuple key into binary format preserving lexicographic order."""
        ...

    def decode(self, encoded: bytes) -> tuple[str | int, ...]:
        """Decode binary data back to original tuple key."""
        ...

    def upper_bound_of_prefix(self, key: tuple[str | int, ...]) -> bytes:
        r"""Bytes strictly greater than every child key of ``key`` (``encode(key) + b"\xff"``)."""
        ...
