"""Functional tests for key codecs.

Tests run against all codec implementations (BinaryKeyCodec, PyBinaryKeyCodec, StringKeyCodec)
using pytest parametrization.
"""

from __future__ import annotations

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
                    min_size=1,
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
