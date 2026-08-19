# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: overflowcheck=False
"""Optimized Cython implementation of binary key codec."""

from libc.stdint cimport int64_t, uint64_t, INT64_MIN, INT64_MAX
from libc.string cimport memcpy
from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AS_STRING, PyBytes_GET_SIZE
from cpython.unicode cimport PyUnicode_AsUTF8String, PyUnicode_DecodeUTF8

from .exceptions import (
    IntegerOverflowError,
    EncodeError,
    DecodeError,
)

# Type markers for lexicographic ordering (int < str)
cdef unsigned char TYPE_INT = 0x01
cdef unsigned char TYPE_STR = 0x02

# Component terminator - using null byte (0x00) to ensure shorter strings sort before
# longer strings with the same prefix. This is critical for lexicographic ordering.
cdef unsigned char TERMINATOR = 0x00

# Escape byte for null bytes within string content
# String encoding: \x00 in content -> \x00\xff (escaped), terminator is bare \x00
cdef unsigned char ESCAPE_BYTE = 0xFF

# Integer range constants
cdef int64_t INT64_MIN_VAL = INT64_MIN
cdef int64_t INT64_MAX_VAL = INT64_MAX
cdef uint64_t INT64_BIAS = 0x8000000000000000ULL  # 2^63

# String constraints
cdef size_t MAX_STRING_LENGTH = 10 * 1024 * 1024  # 10MB


cdef inline void encode_integer_inline(int64_t value, unsigned char* buffer) noexcept nogil:
    """Encode integer using bias/offset encoding directly into buffer.

    Args:
        value: Integer to encode (must be in int64 range)
        buffer: Output buffer (must have at least 8 bytes available)
    """
    cdef uint64_t biased_value = <uint64_t>(value + <int64_t>INT64_BIAS)

    # Write as big-endian (most significant byte first)
    buffer[0] = <unsigned char>((biased_value >> 56) & 0xFF)
    buffer[1] = <unsigned char>((biased_value >> 48) & 0xFF)
    buffer[2] = <unsigned char>((biased_value >> 40) & 0xFF)
    buffer[3] = <unsigned char>((biased_value >> 32) & 0xFF)
    buffer[4] = <unsigned char>((biased_value >> 24) & 0xFF)
    buffer[5] = <unsigned char>((biased_value >> 16) & 0xFF)
    buffer[6] = <unsigned char>((biased_value >> 8) & 0xFF)
    buffer[7] = <unsigned char>(biased_value & 0xFF)


cdef inline int64_t decode_integer_inline(const unsigned char* buffer) noexcept nogil:
    """Decode integer from buffer using bias/offset encoding.

    Args:
        buffer: Input buffer (must have at least 8 bytes)

    Returns:
        Decoded integer value
    """
    cdef uint64_t biased_value = (
        (<uint64_t>buffer[0] << 56) |
        (<uint64_t>buffer[1] << 48) |
        (<uint64_t>buffer[2] << 40) |
        (<uint64_t>buffer[3] << 32) |
        (<uint64_t>buffer[4] << 24) |
        (<uint64_t>buffer[5] << 16) |
        (<uint64_t>buffer[6] << 8) |
        <uint64_t>buffer[7]
    )

    # Remove bias to get original signed value
    return <int64_t>(biased_value - INT64_BIAS)


cdef inline Py_ssize_t count_null_bytes(const unsigned char* data, Py_ssize_t length) noexcept nogil:
    """Count null bytes in data for calculating escaped size."""
    cdef Py_ssize_t count = 0
    cdef Py_ssize_t i
    for i in range(length):
        if data[i] == 0x00:
            count += 1
    return count


cdef inline Py_ssize_t copy_with_escape(unsigned char* dest, const unsigned char* src, Py_ssize_t length) noexcept nogil:
    r"""Copy source to dest, escaping null bytes (\x00 -> \x00\xff).

    Returns the number of bytes written to dest.
    """
    cdef Py_ssize_t dest_pos = 0
    cdef Py_ssize_t i
    for i in range(length):
        dest[dest_pos] = src[i]
        dest_pos += 1
        if src[i] == 0x00:
            dest[dest_pos] = ESCAPE_BYTE
            dest_pos += 1
    return dest_pos


cdef class BinaryKeyCodec:
    """Optimized binary key codec that preserves lexicographic ordering.

    This Cython implementation provides maximum performance for encoding and
    decoding tuple keys into binary format while maintaining sort order.

    Features:
    - Zero Python overhead in hot paths
    - Direct memory operations
    - Inline integer encoding/decoding
    - Efficient buffer management
    - Preserves lexicographic ordering
    - Uses null byte (0x00) terminator with escaping for correct string ordering

    Encoding format:
    - Each component: TYPE_MARKER + ENCODED_VALUE + TERMINATOR
    - Integers: TYPE_INT (0x01) + 8_BYTES + TERMINATOR (0x00)
    - Strings: TYPE_STR (0x02) + ESCAPED_UTF8_BYTES + TERMINATOR (0x00)

    String escaping:
    - Null bytes in content: 0x00 -> 0x00 0xFF (escaped)
    - Terminator is bare 0x00 (not escaped)
    - This ensures shorter strings sort before longer strings with same prefix
    """

    def encode(self, tuple key):
        """Encode tuple key into binary format preserving lexicographic order.

        Args:
            key: Tuple containing strings and/or integers

        Returns:
            Binary encoded key (bytes)

        Raises:
            IntegerOverflowError: If integer is outside int64 range
            EncodeError: If encoding fails
        """
        if not key:
            raise EncodeError("Empty tuple not allowed as key")

        cdef Py_ssize_t num_components = len(key)
        cdef Py_ssize_t i
        cdef object component
        cdef int64_t int_val
        cdef bytes str_bytes
        cdef const unsigned char* str_data
        cdef Py_ssize_t str_len
        cdef Py_ssize_t null_count

        # First pass: calculate exact size (accounting for null byte escaping)
        cdef Py_ssize_t total_size = 0
        for i in range(num_components):
            component = key[i]

            if isinstance(component, int):
                # Check bounds
                try:
                    int_val = <int64_t>component
                except OverflowError:
                    raise IntegerOverflowError(component)

                if int_val < INT64_MIN_VAL or int_val > INT64_MAX_VAL:
                    raise IntegerOverflowError(int_val, INT64_MIN_VAL, INT64_MAX_VAL)
                total_size += 1 + 8 + 1  # type + 8 bytes + terminator

            elif isinstance(component, str):
                str_bytes = PyUnicode_AsUTF8String(component)
                str_data = <const unsigned char*>PyBytes_AS_STRING(str_bytes)
                str_len = PyBytes_GET_SIZE(str_bytes)

                # Check string length bounds
                if str_len == 0:
                    raise EncodeError(f"Empty string at index {i} not allowed")
                if str_len > MAX_STRING_LENGTH:
                    raise EncodeError(
                        f"String at index {i} too long: {str_len} bytes "
                        f"(max {MAX_STRING_LENGTH})"
                    )

                # Count null bytes for escape sizing
                null_count = count_null_bytes(str_data, str_len)
                # type + utf8 bytes + escape bytes for nulls + terminator
                total_size += 1 + str_len + null_count + 1
            else:
                raise EncodeError(
                    f"Component at index {i} must be str or int, "
                    f"got {type(component).__name__}"
                )

        # Allocate output buffer
        cdef bytes result = PyBytes_FromStringAndSize(NULL, total_size)
        cdef unsigned char* buffer = <unsigned char*>PyBytes_AS_STRING(result)
        cdef Py_ssize_t pos = 0

        # Second pass: encode into buffer
        for i in range(num_components):
            component = key[i]

            if isinstance(component, int):
                int_val = <int64_t>component

                # Write type marker
                buffer[pos] = TYPE_INT
                pos += 1

                # Encode integer
                encode_integer_inline(int_val, &buffer[pos])
                pos += 8

                # Write terminator
                buffer[pos] = TERMINATOR
                pos += 1

            else:  # isinstance(component, str)
                str_bytes = PyUnicode_AsUTF8String(component)
                str_data = <const unsigned char*>PyBytes_AS_STRING(str_bytes)
                str_len = PyBytes_GET_SIZE(str_bytes)

                # Write type marker
                buffer[pos] = TYPE_STR
                pos += 1

                # Copy string bytes with null byte escaping
                pos += copy_with_escape(&buffer[pos], str_data, str_len)

                # Write terminator
                buffer[pos] = TERMINATOR
                pos += 1

        return result

    def upper_bound_of_prefix(self, tuple key):
        """Encoded bytes strictly greater than every child key of ``key``.

        Appends ``0xFF`` to ``encode(key)``. The trailing 0xFF sorts strictly
        above every valid segment type marker (``TYPE_INT=0x01``, ``TYPE_STR=0x02``),
        so any child key of the form ``key + (segment,)`` compares less than
        the returned bytes. Used by container-side reverse scans to seed
        ``StorageScanOptions.start`` with an inclusive upper bound of the
        child prefix range.

        Args:
            key: Tuple containing strings and/or integers

        Returns:
            Binary bytes that sort strictly greater than every child key of ``key``
        """
        return self.encode(key) + b"\xff"

    def decode(self, bytes encoded):
        """Decode binary data back to original tuple key.

        Args:
            encoded: Previously encoded binary key

        Returns:
            Original tuple key

        Raises:
            DecodeError: If data is invalid or corrupted
        """
        if not encoded:
            raise DecodeError("Empty encoded key")

        cdef const unsigned char* data = <const unsigned char*>PyBytes_AS_STRING(encoded)
        cdef Py_ssize_t data_len = PyBytes_GET_SIZE(encoded)
        cdef Py_ssize_t pos = 0

        cdef list result = []
        cdef unsigned char type_marker
        cdef int64_t int_val
        cdef Py_ssize_t str_start, str_end
        cdef bytes str_bytes
        cdef str decoded_str

        while pos < data_len:
            # Read type marker
            if pos >= data_len:
                raise DecodeError("Unexpected end of data while reading type marker")

            type_marker = data[pos]
            pos += 1

            if type_marker == TYPE_INT:
                # Check we have enough bytes for integer
                if pos + 8 > data_len:
                    raise DecodeError(
                        f"Insufficient bytes for integer at offset {pos - 1}"
                    )

                # Decode integer
                int_val = decode_integer_inline(&data[pos])
                result.append(int_val)
                pos += 8

                # Check terminator
                if pos >= data_len or data[pos] != TERMINATOR:
                    raise DecodeError(
                        f"Missing terminator after integer at position {pos}"
                    )
                pos += 1

            elif type_marker == TYPE_STR:
                # Find next UNESCAPED terminator (bare \x00, not \x00\xff)
                str_start = pos
                str_end = -1

                while pos < data_len:
                    if data[pos] == TERMINATOR:
                        # Check if this is an escaped null (\x00\xff) or real terminator
                        if pos + 1 < data_len and data[pos + 1] == ESCAPE_BYTE:
                            # This is an escaped null byte, skip both bytes
                            pos += 2
                        else:
                            # This is the real terminator
                            str_end = pos
                            break
                    else:
                        pos += 1

                if str_end == -1:
                    raise DecodeError(
                        f"Missing terminator after string at position {str_start - 1}"
                    )

                # Get escaped string bytes and unescape them
                str_bytes = encoded[str_start:str_end]
                # Unescape: \x00\xff -> \x00
                str_bytes = str_bytes.replace(b"\x00\xff", b"\x00")
                try:
                    decoded_str = PyUnicode_DecodeUTF8(
                        PyBytes_AS_STRING(str_bytes),
                        PyBytes_GET_SIZE(str_bytes),
                        NULL  # Use default error handling (strict)
                    )
                    result.append(decoded_str)
                except UnicodeDecodeError as e:
                    raise DecodeError(f"Invalid UTF-8 in encoded string: {e}")

                pos = str_end + 1

            else:
                raise DecodeError(
                    f"Invalid type marker at position {pos - 1}: 0x{type_marker:02x}"
                )

        return tuple(result)


__all__ = ['BinaryKeyCodec']
