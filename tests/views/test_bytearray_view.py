"""End-to-end tests for ByteArrayView."""

import pytest


def test_bytearray_factory_creates_empty_view(bytearray_factory):
    data = bytearray_factory("data")
    assert data is not None
    assert len(data) == 0


def test_bytearray_factory_with_data(bytearray_factory):
    data = bytearray_factory("data", b"hello")
    extracted = data.extract()
    assert extracted == bytearray(b"hello")


# ============================================================================
# BASIC OPERATIONS
# ============================================================================


def test_bytearray_indexing(bytearray_factory):
    data = bytearray_factory("data", b"abc")
    assert data[0] == ord("a")
    assert data[1] == ord("b")
    assert data[2] == ord("c")


def test_bytearray_negative_indexing(bytearray_factory):
    data = bytearray_factory("data", b"abc")
    assert data[-1] == ord("c")
    assert data[-2] == ord("b")
    assert data[-3] == ord("a")


def test_bytearray_index_out_of_range(bytearray_factory):
    data = bytearray_factory("data", b"abc")
    with pytest.raises(IndexError):
        data[5]
    with pytest.raises(IndexError):
        data[-4]


def test_bytearray_setitem(bytearray_factory):
    data = bytearray_factory("data", b"abc")
    data[0] = ord("A")
    assert data[0] == ord("A")
    assert data.extract() == bytearray(b"Abc")


def test_bytearray_setitem_validation(bytearray_factory):
    data = bytearray_factory("data", b"abc")
    with pytest.raises(ValueError, match="byte must be in range"):
        data[0] = 256
    with pytest.raises(ValueError, match="byte must be in range"):
        data[0] = -1


def test_bytearray_len(bytearray_factory):
    empty = bytearray_factory("empty", b"")
    assert len(empty) == 0

    data = bytearray_factory("data", b"hello")
    assert len(data) == 5


def test_bytearray_iteration(bytearray_factory):
    data = bytearray_factory("data", b"abc")
    result = list(data)
    assert result == [ord("a"), ord("b"), ord("c")]


def test_bytearray_contains(bytearray_factory):
    data = bytearray_factory("data", b"abc")
    assert ord("a") in data
    assert ord("z") not in data


def test_bytearray_reversed(bytearray_factory):
    data = bytearray_factory("data", b"abc")
    result = list(reversed(data))
    assert result == [ord("c"), ord("b"), ord("a")]


# ============================================================================
# MUTATION
# ============================================================================


def test_bytearray_append(bytearray_factory):
    data = bytearray_factory("data", b"ab")
    data.append(ord("c"))
    assert len(data) == 3
    assert data.extract() == bytearray(b"abc")


def test_bytearray_append_validation(bytearray_factory):
    data = bytearray_factory("data", b"")
    with pytest.raises(ValueError, match="byte must be in range"):
        data.append(300)


def test_bytearray_extend(bytearray_factory):
    data = bytearray_factory("data", b"ab")
    data.extend([ord("c"), ord("d")])
    assert data.extract() == bytearray(b"abcd")


def test_bytearray_clear(bytearray_factory):
    data = bytearray_factory("data", b"hello")
    data.clear()
    assert len(data) == 0
    assert data.extract() == bytearray(b"")


# ============================================================================
# SHIFT-BASED MUTATIONS
# ============================================================================


def test_bytearray_delitem_middle(bytearray_factory):
    data = bytearray_factory("data", b"abcde")
    del data[2]
    assert data.extract() == bytearray(b"abde")
    assert len(data) == 4


def test_bytearray_delitem_head(bytearray_factory):
    data = bytearray_factory("data", b"abcde")
    del data[0]
    assert data.extract() == bytearray(b"bcde")


def test_bytearray_delitem_tail(bytearray_factory):
    data = bytearray_factory("data", b"abcde")
    del data[-1]
    assert data.extract() == bytearray(b"abcd")


def test_bytearray_delitem_out_of_range(bytearray_factory):
    data = bytearray_factory("data", b"abc")
    with pytest.raises(IndexError):
        del data[10]


def test_bytearray_insert_middle(bytearray_factory):
    data = bytearray_factory("data", b"ace")
    data.insert(1, ord("b"))
    assert data.extract() == bytearray(b"abce")


def test_bytearray_insert_head(bytearray_factory):
    data = bytearray_factory("data", b"bcd")
    data.insert(0, ord("a"))
    assert data.extract() == bytearray(b"abcd")


def test_bytearray_insert_beyond_end_clamps(bytearray_factory):
    data = bytearray_factory("data", b"abc")
    data.insert(100, ord("d"))
    assert data.extract() == bytearray(b"abcd")


def test_bytearray_insert_validation(bytearray_factory):
    data = bytearray_factory("data", b"abc")
    with pytest.raises(ValueError):
        data.insert(0, 999)


def test_bytearray_pop_default(bytearray_factory):
    data = bytearray_factory("data", b"abcde")
    value = data.pop()
    assert value == ord("e")
    assert data.extract() == bytearray(b"abcd")


def test_bytearray_pop_index(bytearray_factory):
    data = bytearray_factory("data", b"abcde")
    value = data.pop(1)
    assert value == ord("b")
    assert data.extract() == bytearray(b"acde")


def test_bytearray_pop_empty_raises(bytearray_factory):
    data = bytearray_factory("data")
    with pytest.raises(IndexError):
        data.pop()


def test_bytearray_remove_first_occurrence(bytearray_factory):
    data = bytearray_factory("data", b"abcba")
    data.remove(ord("b"))
    assert data.extract() == bytearray(b"acba")


def test_bytearray_remove_missing_raises(bytearray_factory):
    data = bytearray_factory("data", b"abc")
    with pytest.raises(ValueError):
        data.remove(ord("z"))


# ============================================================================
# SEARCH
# ============================================================================


def test_bytearray_index(bytearray_factory):
    data = bytearray_factory("data", b"abcabc")
    assert data.index(ord("a")) == 0
    assert data.index(ord("b")) == 1
    assert data.index(ord("c")) == 2


def test_bytearray_index_not_found(bytearray_factory):
    data = bytearray_factory("data", b"abc")
    with pytest.raises(ValueError, match="is not in bytearray"):
        data.index(ord("z"))


def test_bytearray_count(bytearray_factory):
    data = bytearray_factory("data", b"abcabc")
    assert data.count(ord("a")) == 2
    assert data.count(ord("z")) == 0


# ============================================================================
# STORE AND EXTRACT
# ============================================================================


def test_bytearray_store_replaces_content(bytearray_factory):
    data = bytearray_factory("data", b"hello")
    data.store(b"world")
    assert data.extract() == bytearray(b"world")


def test_bytearray_extract_returns_bytearray(bytearray_factory):
    data = bytearray_factory("data", b"test")
    extracted = data.extract()
    assert isinstance(extracted, bytearray)
    assert extracted == bytearray(b"test")
