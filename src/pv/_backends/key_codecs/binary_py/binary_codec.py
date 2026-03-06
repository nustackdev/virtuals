"""Binary key codec implementation for lexicographic ordering preservation."""

from __future__ import annotations

import re
from typing import Final

from tkv.tkv.codec import KeyCodecProtocol

from ..exceptions import DecodeError, EncodeError, IntegerOverflowError, StringConstraintError
from ..types import EncodedBinaryKey, Key, KeySegment


# Type markers for lexicographic ordering (int < str)
_TYPE_INT: Final[bytes] = b"\x01"
_TYPE_STR: Final[bytes] = b"\x02"

# Component terminator - using null byte (0x00) to ensure shorter strings sort before
# longer strings with the same prefix. This is critical for lexicographic ordering.
# Example: ('0',) must sort before ('00',) - with 0x00 terminator this works because
# 0x00 < any printable character.
_TERMINATOR: Final[bytes] = b"\x00"

# Escape byte for null bytes within string content
# String encoding: \x00 in content -> \x00\xff (escaped), terminator is bare \x00
_ESCAPE_BYTE: Final[bytes] = b"\xff"

# Integer range constraints
_INT64_MIN: Final[int] = -(2**63)
_INT64_MAX: Final[int] = 2**63 - 1
_INT64_BIAS: Final[int] = 2**63  # Bias for offset binary encoding


# Constants for validation
MAX_STRING_LENGTH: Final[int] = 10 * 1024 * 1024  # 10MB
MIN_STRING_LENGTH: Final[int] = 1  # No empty strings allowed

# Pattern for valid string components (printable ASCII + common Unicode)
# This ensures human readability and avoids problematic characters
VALID_STRING_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[\w\s\-_./:@#$%&+=<>?!()[\]{}|~`^*]+$", re.UNICODE
)


def validate_key(key: Key) -> None:
    """Validate that a key tuple meets basic requirements.

    Args:
        key: Tuple to validate

    Raises:
        EncodeError: If key structure is invalid
        StringConstraintError: If string components violate constraints
    """
    from ..exceptions import EncodeError

    if not isinstance(key, tuple):
        raise EncodeError(f"Key must be a tuple, got {type(key).__name__}")

    if not key:
        raise EncodeError("Empty tuple not allowed as key")

    for i, component in enumerate(key):
        validate_key_component(component, i)


def validate_key_component(component: KeySegment, index: int) -> None:
    """Validate a single key component.

    Args:
        component: Component to validate
        index: Position in the key tuple (for error messages)

    Raises:
        EncodeError: If component type is invalid
        StringConstraintError: If string component violates constraints
    """
    from ..exceptions import EncodeError

    if isinstance(component, str):
        validate_string_component(component, index)
    elif isinstance(component, int):
        # Integer validation is codec-specific, done in individual codecs
        pass
    else:
        raise EncodeError(
            f"Component at index {index} must be str or int, got {type(component).__name__}"
        )


def validate_string_component(value: str, index: int) -> None:
    """Validate a string component meets general constraints.

    Args:
        value: String to validate
        index: Position in the key tuple (for error messages)

    Raises:
        StringConstraintError: If string violates constraints
    """
    if len(value) < MIN_STRING_LENGTH:
        raise StringConstraintError(f"Empty string at index {index} not allowed")

    if len(value) > MAX_STRING_LENGTH:
        raise StringConstraintError(
            f"String at index {index} too long: {len(value)} chars (max {MAX_STRING_LENGTH})"
        )

    # Check for valid characters (human-readable constraint)
    if not VALID_STRING_PATTERN.match(value):
        raise StringConstraintError(
            f"String at index {index} contains invalid characters. "
            f"Only printable ASCII and common Unicode characters are allowed."
        )


def _encode_integer(value: int) -> bytes:
    """Encode integer preserving lexicographic order using bias/offset encoding.

    Uses offset binary (also called excess-K or biased representation) to map
    the entire signed integer range to unsigned values that maintain numeric
    ordering when compared lexicographically as bytes.

    Strategy:
    - Add bias of 2^63 to shift entire range to non-negative
    - More negative → smaller biased value → smaller bytes
    - More positive → larger biased value → larger bytes
    - Encode as unsigned 64-bit big-endian

    This naturally preserves ordering without bit manipulation.

    Examples (conceptual):
        -2^63     → 0x0000000000000000 (smallest)
        -2        → 0x7FFFFFFFFFFFFFFE
        -1        → 0x7FFFFFFFFFFFFFFF
        0         → 0x8000000000000000
        1         → 0x8000000000000001
        2^63-1    → 0xFFFFFFFFFFFFFFFF (largest)

    Args:
        value: Integer to encode

    Returns:
        Encoded bytes (8 bytes total, unsigned big-endian)

    Raises:
        IntegerOverflowError: If integer is outside int64 range
    """
    if not (_INT64_MIN <= value <= _INT64_MAX):
        raise IntegerOverflowError(value, _INT64_MIN, _INT64_MAX)

    # Bias encoding: shift entire range to non-negative
    # This naturally preserves ordering: smaller values → smaller biased values
    biased_value = value + _INT64_BIAS

    # Encode as unsigned 64-bit big-endian
    return biased_value.to_bytes(8, byteorder="big", signed=False)


def _decode_integer(data: bytes, offset: int) -> tuple[int, int]:
    """Decode integer from binary data using bias/offset encoding.

    Args:
        data: Binary data containing encoded integer
        offset: Starting position in data

    Returns:
        Tuple of (decoded_value, bytes_consumed)

    Raises:
        DecodeError: If data is insufficient or invalid
    """
    if len(data) - offset < 8:  # Need 8 bytes for uint64
        raise DecodeError(f"Insufficient bytes for integer at offset {offset}")

    int_bytes = data[offset : offset + 8]

    # Decode as unsigned 64-bit big-endian
    biased_value = int.from_bytes(int_bytes, byteorder="big", signed=False)

    # Remove bias to get original signed value
    value = biased_value - _INT64_BIAS

    return value, 8


def _encode_string(value: str) -> bytes:
    r"""Encode string to UTF-8 bytes with null byte escaping.

    Escapes null bytes (0x00) in the content to allow using 0x00 as terminator.
    This is essential for correct lexicographic ordering where shorter strings
    must sort before longer strings with the same prefix.

    Escaping scheme:
    - \x00 in content -> \x00\xff (escaped null)
    - Bare \x00 is reserved for terminator (not in this function)

    Args:
        value: String to encode

    Returns:
        UTF-8 encoded bytes with null bytes escaped

    Raises:
        EncodeError: If string contains invalid UTF-8
    """
    try:
        encoded = value.encode("utf-8")
        # Escape null bytes: \x00 -> \x00\xff
        return encoded.replace(b"\x00", b"\x00\xff")
    except UnicodeEncodeError as e:
        raise EncodeError(f"Invalid UTF-8 in string: {e}") from e


def _decode_string(data: bytes) -> str:
    r"""Decode UTF-8 string from bytes with null byte unescaping.

    Unescapes null bytes that were escaped during encoding.

    Unescaping scheme:
    - \x00\xff -> \x00 (original null byte in content)

    Args:
        data: Encoded string bytes (with escaped null bytes)

    Returns:
        Decoded string

    Raises:
        DecodeError: If data contains invalid UTF-8
    """
    try:
        # Unescape null bytes: \x00\xff -> \x00
        unescaped = data.replace(b"\x00\xff", b"\x00")
        return unescaped.decode("utf-8")
    except UnicodeDecodeError as e:
        raise DecodeError(f"Invalid UTF-8 in encoded string: {e}") from e


class PyBinaryKeyCodec(KeyCodecProtocol[EncodedBinaryKey]):
    """Binary key codec that preserves lexicographic ordering for KV storage.

    This codec encodes tuple keys into binary format while maintaining the
    natural sort order of the original tuples. It supports mixed integer
    and string components.

    Key features:
    - Preserves lexicographic ordering for both integers and strings
    - Handles negative integers correctly using bias/offset encoding
    - Efficient binary encoding with type safety
    - Uses null byte (0x00) as terminator with escaping for correct ordering
    - Trailing terminator ensures prefix-based range queries work correctly

    Integer encoding:
    - Uses bias/offset encoding (excess-2^63 representation)
    - Maps signed int64 range to unsigned for natural byte ordering
    - 8 bytes per integer (unsigned big-endian)

    String encoding:
    - UTF-8 bytes with null byte escaping
    - Null bytes in content: 0x00 -> 0x00 0xFF (escaped)
    - Terminator is bare 0x00 (not escaped)
    - This ensures shorter strings sort before longer strings with same prefix:
      e.g., ('0',) < ('00',) because 0x00 < any printable character

    Encoding format:
    - Each component: TYPE_MARKER + ENCODED_VALUE + TERMINATOR
    - Integers: TYPE_INT (0x01) + 8_BYTES + TERMINATOR (0x00)
    - Strings: TYPE_STR (0x02) + ESCAPED_UTF8_BYTES + TERMINATOR (0x00)
    - Trailing terminator after last component

    Type ordering:
    - TYPE_INT (0x01) < TYPE_STR (0x02) ensures integers sort before strings

    Example:
        >>> codec = PyBinaryKeyCodec()
        >>> key = ("users", 42, "profile")
        >>> encoded = codec.encode(key)
        >>> decoded = codec.decode(encoded)
        >>> assert decoded == key
        >>>
        >>> # Negative integers work correctly
        >>> k1 = ("balance", -100)
        >>> k2 = ("balance", 50)
        >>> assert (k1 < k2) == (codec.encode(k1) < codec.encode(k2))
        >>>
        >>> # String prefix ordering works correctly
        >>> k1 = ("0",)
        >>> k2 = ("00",)
        >>> assert (k1 < k2) == (codec.encode(k1) < codec.encode(k2))
    """

    def encode(self, key: Key) -> EncodedBinaryKey:
        """Encode tuple key into binary format preserving lexicographic order.

        Args:
            key: Tuple containing strings and/or integers

        Returns:
            Binary encoded key with trailing separator

        Raises:
            EncodeError: If key structure is invalid
            IntegerOverflowError: If integer is outside supported range
        """
        # Basic validation (skip pattern validation for performance)
        if not isinstance(key, tuple):
            raise EncodeError(f"Key must be a tuple, got {type(key).__name__}")
        if not key:
            raise EncodeError("Empty tuple not allowed as key")

        parts: list[bytes] = []

        for i, component in enumerate(key):
            if isinstance(component, int):
                parts.extend([_TYPE_INT, _encode_integer(component), _TERMINATOR])
            elif isinstance(component, str):
                parts.extend([_TYPE_STR, _encode_string(component), _TERMINATOR])
            else:
                # This should never happen due to validate_key, but be defensive
                raise EncodeError(
                    f"Unsupported component type at index {i}: {type(component).__name__}"
                )

        # Parts already include terminator after each component
        return b"".join(parts)

    def decode(self, encoded: EncodedBinaryKey) -> Key:
        """Decode binary data back to original tuple key.

        Args:
            encoded: Previously encoded binary key

        Returns:
            Original tuple key

        Raises:
            DecodeError: If data is invalid or corrupted
        """
        if not isinstance(encoded, bytes):
            raise DecodeError(f"Expected bytes, got {type(encoded).__name__}")

        if not encoded:
            raise DecodeError("Empty encoded key")

        result: list[int | str] = []
        pos = 0

        while pos < len(encoded):
            # Read type marker
            if pos >= len(encoded):
                raise DecodeError("Unexpected end of data while reading type marker")

            type_marker = encoded[pos : pos + 1]
            pos += 1

            if type_marker == _TYPE_INT:
                # Decode integer (8 bytes with bias encoding)
                value, consumed = _decode_integer(encoded, pos)
                result.append(value)
                pos += consumed

                # Expect terminator
                if pos >= len(encoded) or encoded[pos : pos + 1] != _TERMINATOR:
                    raise DecodeError(f"Missing terminator after integer at position {pos}")
                pos += 1

            elif type_marker == _TYPE_STR:
                # Find next UNESCAPED terminator (bare \x00, not \x00\xff)
                # We need to scan for \x00 that is NOT followed by \xff
                term_pos = -1
                search_pos = pos
                while search_pos < len(encoded):
                    next_null = encoded.find(b"\x00", search_pos)
                    if next_null == -1:
                        break
                    # Check if this is an escaped null (\x00\xff) or real terminator
                    if next_null + 1 < len(encoded) and encoded[next_null + 1] == 0xFF:
                        # This is an escaped null byte, skip it
                        search_pos = next_null + 2
                    else:
                        # This is the real terminator
                        term_pos = next_null
                        break

                if term_pos == -1:
                    raise DecodeError(f"Missing terminator after string at position {pos}")

                str_data = encoded[pos:term_pos]
                result.append(_decode_string(str_data))
                pos = term_pos + 1

            else:
                raise DecodeError(f"Invalid type marker at position {pos - 1}: {type_marker!r}")

        return tuple(result)


__all__ = [
    "PyBinaryKeyCodec",
]
