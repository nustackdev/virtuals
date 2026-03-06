"""Exception classes for binary key codec."""

from __future__ import annotations


__all__ = [
    "DecodeError",
    "EncodeError",
    "IntegerOverflowError",
    "KeyCodecError",
]


class KeyCodecError(Exception):
    """Base exception for all key codec errors."""

    pass


class EncodeError(KeyCodecError):
    """Raised when encoding a key fails."""

    pass


class DecodeError(KeyCodecError):
    """Raised when decoding an encoded key fails."""

    pass


class IntegerOverflowError(EncodeError):
    """Raised when an integer value exceeds the codec's supported range."""

    def __init__(  # noqa: D107
        self,
        value: int | None = None,
        min_value: int | None = None,
        max_value: int | None = None,
    ) -> None:
        if value is not None:
            self.value = value
            self.min_value = min_value
            self.max_value = max_value
            super().__init__(
                f"Integer {value} out of supported range"
                f"{f' [{min_value}, {max_value}]' if min_value is not None and max_value is not None else ''}"
            )
        else:
            super().__init__("Integer value out of supported range")
