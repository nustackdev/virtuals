"""Functional tests for key codecs.

Tests run against all codec implementations (BinaryKeyCodec, PyBinaryKeyCodec, StringKeyCodec)
using pytest parametrization.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from virtuals_binary_codec import exceptions as _cython_exc

from virtuals._backends.key_codecs import BinaryKeyCodec, PyBinaryKeyCodec, StringKeyCodec
from virtuals._backends.key_codecs.exceptions import (
    DecodeError,
    EncodeError,
    IntegerOverflowError,
)


# Cython binary codec has its own exception hierarchy.
# Combine both so pytest.raises catches either.
AnyEncodeError = (EncodeError, _cython_exc.EncodeError)
AnyDecodeError = (DecodeError, _cython_exc.DecodeError)
AnyIntegerOverflowError = (IntegerOverflowError, _cython_exc.IntegerOverflowError)


if TYPE_CHECKING:
    from virtuals._backends.key_codecs.types import Key
    from virtuals.tkv.codec import KeyCodecProtocol


# ============================================================================
# Test Data Strategies
# ============================================================================


@st.composite
def safe_key(draw: st.DrawFn) -> Key:
    """Generate keys safe for all codecs (intersection of all constraints)."""
    components = draw(
        st.lists(
            st.one_of(
                # Strings: no forbidden chars for StringKeyCodec
                st.text(
                    alphabet=st.characters(
                        whitelist_categories=("Lu", "Ll", "Nd"),
                        blacklist_characters=".[]",
                    ),
                    min_size=0,
                    max_size=50,
                ),
                # Integers: StringKeyCodec has smallest range
                st.integers(min_value=-49999, max_value=49999),
            ),
            min_size=1,
            max_size=5,
        )
    )
    return tuple(components)


@st.composite
def binary_key(draw: st.DrawFn) -> Key:
    """Generate any key the binary codecs accept: any string (empty, NUL, astral), int64."""
    components = draw(
        st.lists(
            st.one_of(
                st.text(
                    alphabet=st.one_of(
                        st.sampled_from(["\x00", "\x01", "\x7f", "a", "b", "\uffff"]),
                        st.characters(blacklist_categories=("Cs",)),
                    ),
                    max_size=6,
                ),
                st.integers(min_value=-(2**63), max_value=2**63 - 1),
            ),
            min_size=1,
            max_size=5,
        )
    )
    return tuple(components)


BINARY_CODECS = [
    pytest.param(BinaryKeyCodec(), id="binary"),
    pytest.param(PyBinaryKeyCodec(), id="pybinary"),
]


def _python_order(k1: Key, k2: Key) -> int | None:
    """-1/0/1 per Python tuple ordering, None when the tuples are incomparable."""
    try:
        return (k1 > k2) - (k1 < k2)
    except TypeError:
        return None


def _byte_order(e1: bytes, e2: bytes) -> int:
    return (e1 > e2) - (e1 < e2)


# ============================================================================
# Core Functional Tests
# ============================================================================


class TestCodecRoundtrip:
    """Test encode/decode round-trip for all codecs."""

    @given(key=safe_key())
    def test_roundtrip(self, codec: KeyCodecProtocol, key: Key) -> None:
        """Encode then decode returns original key."""
        assert codec.decode(codec.encode(key)) == key

    def test_simple_string_key(self, codec: KeyCodecProtocol) -> None:
        """Simple string-only key."""
        key = ("users", "alice")
        assert codec.decode(codec.encode(key)) == key

    def test_simple_int_key(self, codec: KeyCodecProtocol) -> None:
        """Simple integer-only key."""
        key = (42, 100)
        assert codec.decode(codec.encode(key)) == key

    def test_mixed_key(self, codec: KeyCodecProtocol) -> None:
        """Mixed string/int key."""
        key = ("users", 42, "profile")
        assert codec.decode(codec.encode(key)) == key


class TestLexicographicOrdering:
    """Test lexicographic ordering preservation for all codecs."""

    @given(k1=safe_key(), k2=safe_key())
    def test_ordering_preserved(self, codec: KeyCodecProtocol, k1: Key, k2: Key) -> None:
        """Lexicographic ordering preserved: k1 < k2 ⟺ encode(k1) < encode(k2)."""
        e1, e2 = codec.encode(k1), codec.encode(k2)

        # Python 3 can't compare tuples with incompatible types (e.g., ('0',) vs (0,))
        # In such cases, the codec still orders them deterministically (int < str via type markers)
        try:
            if k1 < k2:
                assert e1 < e2
            elif k1 > k2:
                assert e1 > e2
            else:
                assert e1 == e2
        except TypeError:
            # Types incomparable - codec still produces deterministic ordering
            # Just verify encoding succeeded
            assert isinstance(e1, (bytes, str))
            assert isinstance(e2, (bytes, str))

    def test_prefix_ordering(self, codec: KeyCodecProtocol) -> None:
        """Shorter key orders before longer key with same prefix."""
        k1 = ("users", 42)
        k2 = ("users", 42, "profile")
        e1, e2 = codec.encode(k1), codec.encode(k2)
        assert e1 < e2

    def test_negative_integers(self, codec: KeyCodecProtocol) -> None:
        """Negative integers order correctly."""
        keys = [("x", -100), ("x", -10), ("x", 0), ("x", 10), ("x", 100)]
        encoded = [codec.encode(k) for k in keys]
        assert encoded == sorted(encoded)

    def test_string_ordering(self, codec: KeyCodecProtocol) -> None:
        """String components order lexicographically."""
        keys = [("a",), ("b",), ("c",)]
        encoded = [codec.encode(k) for k in keys]
        assert encoded == sorted(encoded)

    def test_sorted_keys_remain_sorted(self, codec: KeyCodecProtocol) -> None:
        """Sorting keys and sorting encoded keys produce same order."""
        keys = [
            ("users", 1),
            ("users", 10),
            ("users", 2),
            ("items", 5),
            ("admin", 0),
        ]

        sorted_keys = sorted(keys)
        encoded_pairs = [(codec.encode(k), k) for k in keys]
        sorted_by_encoded = [k for _, k in sorted(encoded_pairs)]

        assert sorted_keys == sorted_by_encoded

    def test_string_prefix_ordering_simple(self, codec: KeyCodecProtocol) -> None:
        """Shorter string sorts before longer string with same prefix."""
        # This is the exact bug that was fixed - '0' < '00' in Python
        k1 = ("0",)
        k2 = ("00",)
        e1, e2 = codec.encode(k1), codec.encode(k2)
        assert k1 < k2, "Python tuple ordering should have '0' < '00'"
        assert e1 < e2, f"Encoded ordering must match: {e1!r} should be < {e2!r}"

    def test_string_prefix_ordering_various(self, codec: KeyCodecProtocol) -> None:
        """Various string prefix orderings."""
        # All these should maintain prefix ordering
        test_cases = [
            (("a",), ("aa",)),
            (("a",), ("ab",)),
            (("x",), ("xy",)),
            (("foo",), ("foobar",)),
            (("test",), ("testing",)),
        ]
        for k1, k2 in test_cases:
            e1, e2 = codec.encode(k1), codec.encode(k2)
            assert k1 < k2
            assert e1 < e2, f"Failed for {k1} vs {k2}: {e1!r} should be < {e2!r}"

    def test_string_prefix_ordering_comprehensive(self, codec: KeyCodecProtocol) -> None:
        """Comprehensive string prefix ordering test."""
        # Verify that a sorted list of keys stays sorted after encoding
        keys = [
            ("a",),
            ("aa",),
            ("aaa",),
            ("ab",),
            ("b",),
            ("ba",),
            ("0",),
            ("00",),
            ("000",),
            ("01",),
            ("1",),
            ("10",),
        ]
        sorted_keys = sorted(keys)
        encoded = [codec.encode(k) for k in sorted_keys]
        assert encoded == sorted(encoded), "Encoding should preserve sorted order"

    def test_integer_before_string(self, codec: KeyCodecProtocol) -> None:
        """Integers sort before strings (type marker ordering)."""
        k_int = (0,)
        k_str = ("0",)
        e_int, e_str = codec.encode(k_int), codec.encode(k_str)
        # Can't compare tuples with different types in Python 3
        # But encodings should be comparable and ints should come first
        assert e_int < e_str, "Integers should sort before strings"

    def test_many_integers_before_strings(self, codec: KeyCodecProtocol) -> None:
        """All integers sort before all strings."""
        int_keys = [(i,) for i in [-100, -1, 0, 1, 100]]
        str_keys = [("a",), ("z",), ("0",)]

        int_encoded = [codec.encode(k) for k in int_keys]
        str_encoded = [codec.encode(k) for k in str_keys]

        # Every integer encoding should be less than every string encoding
        for ie in int_encoded:
            for se in str_encoded:
                assert ie < se, f"{ie!r} should be < {se!r}"


# ============================================================================
# Binary Codec Specific Tests
# ============================================================================


class TestBinaryCodecNullByteHandling:
    """Test null byte escaping in binary codecs.

    Binary codecs use 0x00 as the component terminator, so null bytes
    in string content must be escaped to preserve round-trip correctness.
    """

    @pytest.fixture(
        params=[
            pytest.param(BinaryKeyCodec(), id="binary"),
            pytest.param(PyBinaryKeyCodec(), id="pybinary"),
        ]
    )
    def binary_codec(self, request):
        """Binary codec instances only."""
        return request.param

    def test_string_with_null_byte_roundtrip(self, binary_codec) -> None:
        """String containing null byte round-trips correctly."""
        # Create a string with an embedded null byte
        key = ("hello\x00world",)
        encoded = binary_codec.encode(key)
        decoded = binary_codec.decode(encoded)
        assert decoded == key

    def test_string_with_multiple_null_bytes(self, binary_codec) -> None:
        """String with multiple null bytes round-trips correctly."""
        key = ("\x00start\x00middle\x00end\x00",)
        encoded = binary_codec.encode(key)
        decoded = binary_codec.decode(encoded)
        assert decoded == key

    def test_null_byte_escaped_correctly(self, binary_codec) -> None:
        """Verify null bytes are escaped in encoding."""
        key = ("a\x00b",)
        encoded = binary_codec.encode(key)
        # The encoded form should contain \x00\xff (escaped null)
        # not a bare \x00 in the string content
        # Type marker is \x02, then 'a' (0x61), then escaped null \x00\xff,
        # then 'b' (0x62), then terminator \x00
        assert b"\x00\xff" in encoded, "Null byte should be escaped"

    def test_null_byte_ordering_preserved(self, binary_codec) -> None:
        """Strings with null bytes maintain correct ordering."""
        # "a" < "a\x00" < "a\x00b" < "ab" in Python string ordering
        keys = [
            ("a",),
            ("a\x00",),
            ("a\x00b",),
            ("ab",),
        ]
        sorted_keys = sorted(keys)
        encoded = [binary_codec.encode(k) for k in sorted_keys]
        assert encoded == sorted(encoded), "Null byte ordering should be preserved"

    def test_complex_key_with_null_bytes(self, binary_codec) -> None:
        """Complex key with multiple components including null bytes."""
        key = ("prefix", 42, "data\x00with\x00nulls", -1)
        encoded = binary_codec.encode(key)
        decoded = binary_codec.decode(encoded)
        assert decoded == key


class TestIntegerEdgeCases:
    """Test integer encoding edge cases for all codecs."""

    def test_zero(self, codec: KeyCodecProtocol) -> None:
        """Zero encodes and decodes correctly."""
        key = (0,)
        assert codec.decode(codec.encode(key)) == key

    def test_one(self, codec: KeyCodecProtocol) -> None:
        """One encodes and decodes correctly."""
        key = (1,)
        assert codec.decode(codec.encode(key)) == key

    def test_negative_one(self, codec: KeyCodecProtocol) -> None:
        """Negative one encodes and decodes correctly."""
        key = (-1,)
        assert codec.decode(codec.encode(key)) == key

    def test_boundary_transitions(self, codec: KeyCodecProtocol) -> None:
        """Test ordering around zero boundary."""
        keys = [(-2,), (-1,), (0,), (1,), (2,)]
        encoded = [codec.encode(k) for k in keys]
        assert encoded == sorted(encoded), "Ordering around zero should be correct"


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestEncodeErrors:
    """Test that invalid inputs are properly rejected during encoding."""

    def test_empty_tuple_rejected(self, codec: KeyCodecProtocol) -> None:
        """Empty tuple is rejected."""
        with pytest.raises(AnyEncodeError):
            codec.encode(())

    def test_none_rejected(self, codec: KeyCodecProtocol) -> None:
        """None as key is rejected."""
        with pytest.raises((*AnyEncodeError, TypeError, AttributeError)):
            codec.encode(None)  # type: ignore

    def test_list_rejected(self, codec: KeyCodecProtocol) -> None:
        """List instead of tuple is rejected."""
        with pytest.raises((*AnyEncodeError, TypeError, AttributeError)):
            codec.encode(["a", "b"])  # type: ignore

    def test_float_component_rejected(self, codec: KeyCodecProtocol) -> None:
        """Float component is rejected."""
        with pytest.raises(AnyEncodeError):
            codec.encode((3.14,))  # type: ignore

    def test_none_component_rejected(self, codec: KeyCodecProtocol) -> None:
        """None component is rejected."""
        with pytest.raises(AnyEncodeError):
            codec.encode((None,))  # type: ignore

    def test_nested_tuple_rejected(self, codec: KeyCodecProtocol) -> None:
        """Nested tuple component is rejected."""
        with pytest.raises(AnyEncodeError):
            codec.encode((("nested",),))  # type: ignore

    def test_dict_component_rejected(self, codec: KeyCodecProtocol) -> None:
        """Dict component is rejected."""
        with pytest.raises(AnyEncodeError):
            codec.encode(({"key": "value"},))  # type: ignore

    def test_bytes_component_rejected(self, codec: KeyCodecProtocol) -> None:
        """Bytes component is rejected (strings only, not bytes)."""
        with pytest.raises(AnyEncodeError):
            codec.encode((b"bytes",))  # type: ignore


class TestStringCodecSpecificErrors:
    """Test string codec specific error handling."""

    @pytest.fixture
    def string_codec(self) -> StringKeyCodec:
        return StringKeyCodec()

    def test_forbidden_dot_rejected(self, string_codec: StringKeyCodec) -> None:
        """String containing dot is rejected."""
        with pytest.raises(EncodeError):
            string_codec.encode(("has.dot",))

    def test_forbidden_bracket_rejected(self, string_codec: StringKeyCodec) -> None:
        """String containing brackets is rejected."""
        with pytest.raises(EncodeError):
            string_codec.encode(("has[bracket",))
        with pytest.raises(EncodeError):
            string_codec.encode(("has]bracket",))

    def test_integer_overflow_rejected(self, string_codec: StringKeyCodec) -> None:
        """Integer outside string codec range is rejected."""
        # String codec has range -49999 to 49999
        with pytest.raises(IntegerOverflowError):
            string_codec.encode((50000,))
        with pytest.raises(IntegerOverflowError):
            string_codec.encode((-50000,))


class TestBinaryCodecIntegerBoundaries:
    """Test binary codec integer boundaries (int64 range)."""

    @pytest.fixture(
        params=[
            pytest.param(BinaryKeyCodec(), id="binary"),
            pytest.param(PyBinaryKeyCodec(), id="pybinary"),
        ]
    )
    def binary_codec(self, request):
        return request.param

    def test_int64_max(self, binary_codec) -> None:
        """Maximum int64 value encodes/decodes correctly."""
        max_int64 = 2**63 - 1
        key = (max_int64,)
        assert binary_codec.decode(binary_codec.encode(key)) == key

    def test_int64_min(self, binary_codec) -> None:
        """Minimum int64 value encodes/decodes correctly."""
        min_int64 = -(2**63)
        key = (min_int64,)
        assert binary_codec.decode(binary_codec.encode(key)) == key

    def test_int64_overflow_rejected(self, binary_codec) -> None:
        """Values outside int64 range are rejected."""
        with pytest.raises(AnyIntegerOverflowError):
            binary_codec.encode((2**63,))  # max + 1
        with pytest.raises(AnyIntegerOverflowError):
            binary_codec.encode((-(2**63) - 1,))  # min - 1

    def test_int64_ordering_at_boundaries(self, binary_codec) -> None:
        """Ordering is correct at int64 boundaries."""
        keys = [
            (-(2**63),),  # min
            (-(2**63) + 1,),
            (-1,),
            (0,),
            (1,),
            (2**63 - 2,),
            (2**63 - 1,),  # max
        ]
        encoded = [binary_codec.encode(k) for k in keys]
        assert encoded == sorted(encoded), "Boundary ordering must be correct"


# ============================================================================
# Decode Robustness Tests
# ============================================================================


class TestDecodeErrors:
    """Test that invalid encoded data is properly rejected during decoding."""

    @pytest.fixture(
        params=[
            pytest.param(BinaryKeyCodec(), id="binary"),
            pytest.param(PyBinaryKeyCodec(), id="pybinary"),
        ]
    )
    def binary_codec(self, request):
        return request.param

    def test_empty_bytes_rejected(self, binary_codec) -> None:
        """Empty bytes is rejected."""
        with pytest.raises(AnyDecodeError):
            binary_codec.decode(b"")

    def test_invalid_type_marker_rejected(self, binary_codec) -> None:
        """Invalid type marker is rejected."""
        # 0x03 is not a valid type marker (only 0x01 and 0x02 are)
        with pytest.raises(AnyDecodeError):
            binary_codec.decode(b"\x03test\x00")

    def test_truncated_integer_rejected(self, binary_codec) -> None:
        """Truncated integer (less than 8 bytes) is rejected."""
        # Type marker for int (0x01) followed by only 4 bytes
        with pytest.raises(AnyDecodeError):
            binary_codec.decode(b"\x01\x00\x00\x00\x00")

    def test_missing_terminator_rejected(self, binary_codec) -> None:
        """Missing terminator is rejected."""
        # Valid string but no terminator
        with pytest.raises(AnyDecodeError):
            binary_codec.decode(b"\x02test")

    def test_random_bytes_rejected(self, binary_codec) -> None:
        """Random garbage bytes are rejected or decoded (but don't crash)."""
        # Pre-generated garbage bytes for reproducibility (not crypto, just test data)
        garbage_samples = [
            b"\xff\x03\x10",
            b"\x00",
            b"\x01\x02\x03\x04",
            b"\x02\xff\xff\xff",
            b"\x01",
            b"\x03test\x00",
            b"\x01\x00\x00\x00",
            b"\x02\x00\x00",
            b"\xff" * 20,
            b"\x01\x80\x00\x00\x00\x00\x00\x00\x00\xff",  # Almost valid int but wrong terminator
        ]
        for garbage in garbage_samples:
            # Garbage should either fail to decode or produce some output (no crashes)
            try:
                binary_codec.decode(garbage)
            except AnyDecodeError:
                pass  # Expected - invalid data rejected


class TestStringCodecDecodeErrors:
    """Test string codec decode error handling."""

    @pytest.fixture
    def string_codec(self) -> StringKeyCodec:
        return StringKeyCodec()

    def test_empty_string_rejected(self, string_codec: StringKeyCodec) -> None:
        """Empty string is rejected."""
        with pytest.raises(DecodeError):
            string_codec.decode("")

    def test_missing_type_marker_rejected(self, string_codec: StringKeyCodec) -> None:
        """String without type marker is rejected."""
        with pytest.raises(DecodeError):
            string_codec.decode("nomarker.")

    def test_invalid_type_marker_rejected(self, string_codec: StringKeyCodec) -> None:
        """Invalid type marker is rejected."""
        with pytest.raises(DecodeError):
            string_codec.decode("[x]invalid.")


# ============================================================================
# Cross-Codec Isomorphism Tests
# ============================================================================


class TestBinaryCodecIsomorphism:
    """Test that BinaryKeyCodec and PyBinaryKeyCodec produce identical output."""

    @pytest.fixture
    def binary_codec(self) -> BinaryKeyCodec:
        return BinaryKeyCodec()

    @pytest.fixture
    def pybinary_codec(self) -> PyBinaryKeyCodec:
        return PyBinaryKeyCodec()

    def test_simple_key_identical(
        self, binary_codec: BinaryKeyCodec, pybinary_codec: PyBinaryKeyCodec
    ) -> None:
        """Simple keys produce identical encodings."""
        key = ("users", 42, "profile")
        assert binary_codec.encode(key) == pybinary_codec.encode(key)

    def test_negative_int_identical(
        self, binary_codec: BinaryKeyCodec, pybinary_codec: PyBinaryKeyCodec
    ) -> None:
        """Negative integers produce identical encodings."""
        key = ("balance", -12345)
        assert binary_codec.encode(key) == pybinary_codec.encode(key)

    def test_boundary_ints_identical(
        self, binary_codec: BinaryKeyCodec, pybinary_codec: PyBinaryKeyCodec
    ) -> None:
        """Boundary integers produce identical encodings."""
        keys = [
            (-(2**63),),
            (-(2**63) + 1,),
            (-1,),
            (0,),
            (1,),
            (2**63 - 2,),
            (2**63 - 1,),
        ]
        for key in keys:
            assert binary_codec.encode(key) == pybinary_codec.encode(key), f"Failed for {key}"

    def test_null_bytes_identical(
        self, binary_codec: BinaryKeyCodec, pybinary_codec: PyBinaryKeyCodec
    ) -> None:
        """Strings with null bytes produce identical encodings."""
        key = ("hello\x00world",)
        assert binary_codec.encode(key) == pybinary_codec.encode(key)

    @given(key=safe_key())
    @settings(max_examples=200)
    def test_random_keys_identical(self, key: Key) -> None:
        """Random keys produce identical encodings."""
        binary_codec = BinaryKeyCodec()
        pybinary_codec = PyBinaryKeyCodec()
        assert binary_codec.encode(key) == pybinary_codec.encode(key)

    def test_cross_decode_compatible(
        self, binary_codec: BinaryKeyCodec, pybinary_codec: PyBinaryKeyCodec
    ) -> None:
        """Encoded by one codec can be decoded by the other."""
        key = ("users", -999, "data\x00with\x00nulls", 12345)

        # Encode with binary, decode with pybinary
        encoded = binary_codec.encode(key)
        assert pybinary_codec.decode(encoded) == key

        # Encode with pybinary, decode with binary
        encoded = pybinary_codec.encode(key)
        assert binary_codec.decode(encoded) == key


# ============================================================================
# Encoding Format Verification Tests
# ============================================================================


class TestBinaryEncodingFormat:
    """Verify the actual byte structure of binary encodings."""

    @pytest.fixture(
        params=[
            pytest.param(BinaryKeyCodec(), id="binary"),
            pytest.param(PyBinaryKeyCodec(), id="pybinary"),
        ]
    )
    def binary_codec(self, request):
        return request.param

    def test_string_format(self, binary_codec) -> None:
        """Verify string encoding format: [0x02][UTF-8][0x00]."""
        key = ("abc",)
        encoded = binary_codec.encode(key)
        # TYPE_STR (0x02) + "abc" + TERMINATOR (0x00)
        assert encoded == b"\x02abc\x00"

    def test_integer_format(self, binary_codec) -> None:
        """Verify integer encoding format: [0x01][8-byte biased][0x00]."""
        key = (0,)
        encoded = binary_codec.encode(key)
        # TYPE_INT (0x01) + biased 0 (0x8000000000000000) + TERMINATOR (0x00)
        assert encoded == b"\x01\x80\x00\x00\x00\x00\x00\x00\x00\x00"

    def test_negative_one_format(self, binary_codec) -> None:
        """Verify -1 encoding: biased value should be 0x7FFFFFFFFFFFFFFF."""
        key = (-1,)
        encoded = binary_codec.encode(key)
        # TYPE_INT + biased -1 (2^63 - 1 = 0x7FFFFFFFFFFFFFFF) + TERMINATOR
        assert encoded == b"\x01\x7f\xff\xff\xff\xff\xff\xff\xff\x00"

    def test_one_format(self, binary_codec) -> None:
        """Verify 1 encoding: biased value should be 0x8000000000000001."""
        key = (1,)
        encoded = binary_codec.encode(key)
        # TYPE_INT + biased 1 (2^63 + 1 = 0x8000000000000001) + TERMINATOR
        assert encoded == b"\x01\x80\x00\x00\x00\x00\x00\x00\x01\x00"

    def test_null_byte_escaping_format(self, binary_codec) -> None:
        """Verify null bytes are escaped as 0x00 0xFF."""
        key = ("a\x00b",)
        encoded = binary_codec.encode(key)
        # TYPE_STR + 'a' + escaped_null (0x00 0xFF) + 'b' + TERMINATOR
        assert encoded == b"\x02a\x00\xffb\x00"

    def test_multi_component_format(self, binary_codec) -> None:
        """Verify multi-component encoding."""
        key = ("x", 1)
        encoded = binary_codec.encode(key)
        # String component + Integer component
        expected = (
            b"\x02x\x00"  # TYPE_STR + 'x' + TERMINATOR
            b"\x01\x80\x00\x00\x00\x00\x00\x00\x01\x00"  # TYPE_INT + biased 1 + TERMINATOR
        )
        assert encoded == expected


# ============================================================================
# Determinism and Stability Tests
# ============================================================================


class TestDeterminism:
    """Test that encoding is deterministic and stable."""

    def test_same_key_same_encoding(self, codec: KeyCodecProtocol) -> None:
        """Same key always produces same encoding."""
        key = ("users", 42, "profile")
        encodings = [codec.encode(key) for _ in range(100)]
        assert all(e == encodings[0] for e in encodings)

    def test_equivalent_keys_same_encoding(self, codec: KeyCodecProtocol) -> None:
        """Equivalent keys produce same encoding."""
        key1 = ("test", 123)
        key2 = ("test", 123)  # Same tuple, different construction
        assert codec.encode(key1) == codec.encode(key2)

    @given(key=safe_key())
    def test_encode_decode_identity(self, codec: KeyCodecProtocol, key: Key) -> None:
        """encode(decode(encode(k))) == encode(k) (idempotent after first encode)."""
        encoded1 = codec.encode(key)
        decoded = codec.decode(encoded1)
        encoded2 = codec.encode(decoded)
        assert encoded1 == encoded2


# ============================================================================
# Ordering Stress Tests
# ============================================================================


class TestOrderingStress:
    """Stress tests for ordering correctness."""

    @given(keys=st.lists(safe_key(), min_size=2, max_size=50))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.filter_too_much])
    def test_sorting_equivalence(self, codec: KeyCodecProtocol, keys: list[Key]) -> None:
        """Sorting by key equals sorting by encoded value for comparable keys."""
        # Filter to only keys that are mutually comparable
        try:
            sorted_keys = sorted(keys)
        except TypeError:
            # Keys contain incomparable types, skip this test case
            assume(False)

        encoded_pairs = [(codec.encode(k), k) for k in keys]
        sorted_by_encoded = [k for _, k in sorted(encoded_pairs)]
        assert sorted_keys == sorted_by_encoded

    def test_large_key_ordering(self, codec: KeyCodecProtocol) -> None:
        """Large keys with many components maintain ordering."""
        # Use only integers to ensure keys are comparable
        keys = [(0, *range(i), 999) for i in range(10)]
        sorted_keys = sorted(keys)
        encoded = [codec.encode(k) for k in sorted_keys]
        assert encoded == sorted(encoded)

    def test_unicode_ordering(self, codec: KeyCodecProtocol) -> None:
        """Unicode strings maintain lexicographic ordering."""
        # These should sort by their UTF-8 byte representation
        keys = [("a",), ("b",), ("z",), ("A",), ("Z",)]
        sorted_keys = sorted(keys)
        encoded = [codec.encode(k) for k in sorted_keys]
        assert encoded == sorted(encoded)


# ============================================================================
# Empty Strings and Backward Compatibility (binary codecs)
# ============================================================================


class TestBinaryCodecBackwardCompat:
    """Every key legal before empty strings were allowed encodes byte-for-byte as before.

    The golden file was generated with the released virtuals-binary-codec 0.1.1
    from a seeded corpus (NUL bytes, astral and BMP edge characters, int64
    bounds, mixed keys). Existing on-disk data must stay valid.
    """

    GOLDEN = json.loads((Path(__file__).parent / "binary_codec_golden.json").read_text())

    @pytest.mark.parametrize("codec", BINARY_CODECS)
    def test_encodings_unchanged(self, codec: KeyCodecProtocol) -> None:
        assert len(self.GOLDEN) > 400
        for entry in self.GOLDEN:
            key = tuple(entry["key"])
            assert codec.encode(key).hex() == entry["hex"], key
            assert codec.decode(bytes.fromhex(entry["hex"])) == key


class TestBinaryCodecEmptyStrings:
    """The empty string is a legal key component in both binary codecs."""

    @pytest.mark.parametrize("codec", BINARY_CODECS)
    def test_encoding_bytes(self, codec: KeyCodecProtocol) -> None:
        """The empty string is TYPE_STR + TERMINATOR."""
        assert codec.encode(("",)) == b"\x02\x00"
        assert codec.encode(("a", "")) == b"\x02a\x00\x02\x00"
        assert codec.encode(("", 0)) == b"\x02\x00\x01\x80" + b"\x00" * 7 + b"\x00"

    @pytest.mark.parametrize("codec", BINARY_CODECS)
    def test_roundtrip_every_position(self, codec: KeyCodecProtocol) -> None:
        """The empty string round-trips at every position, next to NULs, ints and strings."""
        neighbours = ["", "\x00", "\x00\x00", "a", "\xff", 0, -1, 2**63 - 1]
        for n in range(1, 4):
            for pos in range(n):
                for other in neighbours:
                    key = tuple("" if i == pos else other for i in range(n))
                    assert codec.decode(codec.encode(key)) == key

    @pytest.mark.parametrize("codec", BINARY_CODECS)
    def test_no_ambiguity_with_escapes(self, codec: KeyCodecProtocol) -> None:
        """Keys that differ only around empty strings and NULs encode differently."""
        keys = [
            ("",),
            ("", ""),
            ("\x00",),
            ("", "\x00"),
            ("\x00", ""),
            ("\x00\x00",),
            ("", 0),
            (0, ""),
            ("", "", ""),
        ]
        encoded = [codec.encode(k) for k in keys]
        assert len(set(encoded)) == len(keys)
        for k, e in zip(keys, encoded, strict=True):
            assert codec.decode(e) == k

    @pytest.mark.parametrize("codec", BINARY_CODECS)
    def test_ordering_mixed(self, codec: KeyCodecProtocol) -> None:
        """Encoded order matches Python tuple order with empty components."""
        keys = [
            ("",),
            ("", ""),
            ("", "", "a"),
            ("", "a"),
            ("", "\x00"),
            ("\x00",),
            ("\x00", ""),
            ("a",),
            ("a", ""),
            ("a", "", "x"),
            ("a", "\x00"),
            ("a", "b"),
            ("a\x00",),
            ("aa",),
            ("b",),
        ]
        assert sorted(keys, key=codec.encode) == sorted(keys)

    @pytest.mark.parametrize("codec", BINARY_CODECS)
    def test_int_before_empty_string(self, codec: KeyCodecProtocol) -> None:
        """Type markers still decide: every int sorts before the empty string."""
        assert codec.encode(("a", 2**63 - 1)) < codec.encode(("a", ""))
        assert codec.encode(("a", "")) < codec.encode(("a", "\x00"))

    @pytest.mark.parametrize("codec", BINARY_CODECS)
    @given(s=st.text(min_size=1, alphabet=st.characters(blacklist_categories=("Cs",))))
    @settings(max_examples=200)
    def test_empty_sorts_first(self, codec: KeyCodecProtocol, s: str) -> None:
        """The empty string sorts before every other string, with or without a tail."""
        assert codec.encode(("",)) < codec.encode((s,))
        assert codec.encode(("", s)) < codec.encode((s,))
        assert codec.encode(("x", "", "zzz")) < codec.encode(("x", s))

    @pytest.mark.parametrize("codec", BINARY_CODECS)
    @given(k1=binary_key(), k2=binary_key())
    @settings(max_examples=500)
    def test_order_matches_python(self, codec: KeyCodecProtocol, k1: Key, k2: Key) -> None:
        """For any comparable keys, byte order equals Python tuple order."""
        expected = _python_order(k1, k2)
        assume(expected is not None)
        assert _byte_order(codec.encode(k1), codec.encode(k2)) == expected

    @pytest.mark.parametrize("codec", BINARY_CODECS)
    @given(key=binary_key())
    @settings(max_examples=300)
    def test_roundtrip_any_key(self, codec: KeyCodecProtocol, key: Key) -> None:
        assert codec.decode(codec.encode(key)) == key

    @pytest.mark.parametrize("codec", BINARY_CODECS)
    def test_upper_bound_of_prefix_with_empty(self, codec: KeyCodecProtocol) -> None:
        """Prefix ("a",) includes ("a", "") and its children, excludes ("b",)."""
        lo, hi = codec.encode(("a",)), codec.upper_bound_of_prefix(("a",))
        for inside in [("a",), ("a", ""), ("a", "", ""), ("a", "", 5), ("a", "\x00"), ("a", 0)]:
            assert lo <= codec.encode(inside) < hi, inside
        for outside in [("b",), ("",), ("a\x00",), ("aa",), ("a\x00", "")]:
            assert not (lo <= codec.encode(outside) < hi), outside

        lo, hi = codec.encode(("",)), codec.upper_bound_of_prefix(("",))
        for inside in [("",), ("", ""), ("", "x"), ("", 0), ("", "\x00")]:
            assert lo <= codec.encode(inside) < hi, inside
        for outside in [("\x00",), ("\x00", ""), ("a",), (0,)]:
            assert not (lo <= codec.encode(outside) < hi), outside

    @pytest.mark.parametrize("codec", BINARY_CODECS)
    @given(prefix=binary_key(), key=binary_key())
    @settings(max_examples=500)
    def test_prefix_range_is_exact(self, codec: KeyCodecProtocol, prefix: Key, key: Key) -> None:
        """A key is in [encode(p), upper_bound_of_prefix(p)) exactly when p is a prefix of it."""
        lo, hi = codec.encode(prefix), codec.upper_bound_of_prefix(prefix)
        inside = lo <= codec.encode(key) < hi
        assert inside == (key[: len(prefix)] == prefix)

    @given(key=binary_key())
    @settings(max_examples=300)
    def test_cython_and_python_identical(self, key: Key) -> None:
        """Cython and Python codecs produce identical bytes and cross-decode."""
        cy, py = BinaryKeyCodec(), PyBinaryKeyCodec()
        encoded = cy.encode(key)
        assert encoded == py.encode(key)
        assert py.decode(encoded) == key
        assert cy.decode(encoded) == key
        assert cy.upper_bound_of_prefix(key) == py.upper_bound_of_prefix(key)


class TestEmptyStringAcrossCodecs:
    """Every codec, and the composite codecs built on them, accept empty strings."""

    @pytest.mark.parametrize("key", [("",), ("a", ""), ("", 1, ""), ("", "a")])
    def test_roundtrip(self, codec: KeyCodecProtocol, key: Key) -> None:
        assert codec.decode(codec.encode(key)) == key

    def test_composite_codecs_agree(self) -> None:
        from virtuals.codecs import BinaryCodec, NoOpCodec, TextCodec

        for composite in (NoOpCodec(), BinaryCodec(), TextCodec()):
            for key in [("",), ("/", ""), ("", 0, "x")]:
                assert composite.decode_key(composite.encode_key(key)) == key
