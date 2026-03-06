"""Unit tests for empty sentinel module."""

from tkv.tkv.types import EMPTY, Empty, is_empty


# ============================================================================
# SINGLETON TESTS
# ============================================================================


def test_empty_singleton_exists() -> None:
    """Test that EMPTY singleton exists and is an Empty instance."""
    assert EMPTY is not None
    assert isinstance(EMPTY, Empty)


# ============================================================================
# EMPTY CLASS TESTS
# ============================================================================


def test_empty_repr() -> None:
    """Test Empty.__repr__ returns debug representation."""
    assert repr(EMPTY) == "<Empty>"


def test_empty_str() -> None:
    """Test Empty.__str__ returns display representation."""
    assert str(EMPTY) == "Empty"


def test_empty_bool() -> None:
    """Test Empty.__bool__ always returns False."""
    assert bool(EMPTY) is False
    assert not EMPTY


def test_empty_eq_with_same_instance() -> None:
    """Test Empty.__eq__ returns True for same instance."""
    assert EMPTY == EMPTY


def test_empty_eq_with_new_instance() -> None:
    """Test Empty.__eq__ returns True for different Empty instances."""
    empty2 = Empty()
    assert EMPTY == empty2


def test_empty_eq_with_other_types() -> None:
    """Test Empty.__eq__ returns False for non-Empty values."""
    assert EMPTY != None
    assert EMPTY != ""
    assert EMPTY != 0
    assert EMPTY != False


def test_empty_hash() -> None:
    """Test Empty.__hash__ returns consistent hash."""
    empty2 = Empty()
    assert hash(EMPTY) == hash(empty2)


def test_empty_hashable() -> None:
    """Test Empty can be used in sets and dicts."""
    empty_set = {EMPTY}
    assert EMPTY in empty_set

    empty_dict = {EMPTY: "value"}
    assert empty_dict[EMPTY] == "value"


# ============================================================================
# TYPE GUARD FUNCTION TESTS
# ============================================================================


def test_is_empty_with_empty_singleton() -> None:
    """Test is_empty() returns True for EMPTY singleton."""
    assert is_empty(EMPTY) is True


def test_is_empty_with_empty_instance() -> None:
    """Test is_empty() returns True for Empty instances."""
    empty = Empty()
    assert is_empty(empty) is True


def test_is_empty_with_non_empty() -> None:
    """Test is_empty() returns False for non-Empty values."""
    assert is_empty(None) is False
    assert is_empty("") is False
    assert is_empty(0) is False
    assert is_empty(False) is False
    assert is_empty([]) is False
