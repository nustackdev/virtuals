"""End-to-end tests for FlatListView."""

import pytest


def test_flat_list_factory_creates_empty_view(flat_list_factory):
    items = flat_list_factory("items")
    assert items is not None
    assert len(items) == 0


def test_flat_list_factory_with_data(flat_list_factory):
    items = flat_list_factory("items", [1, 2, 3])
    assert items.extract() == [1, 2, 3]


# ============================================================================
# BASIC OPERATIONS
# ============================================================================


def test_flat_list_append(flat_list_factory):
    items = flat_list_factory("items")
    items.append(10)
    items.append(20)
    items.append(30)
    assert len(items) == 3
    assert items[0] == 10
    assert items[2] == 30


def test_flat_list_indexing(flat_list_factory):
    items = flat_list_factory("items", [10, 20, 30, 40, 50])
    assert items[0] == 10
    assert items[2] == 30
    assert items[4] == 50


def test_flat_list_negative_indexing(flat_list_factory):
    items = flat_list_factory("items", [10, 20, 30])
    assert items[-1] == 30
    assert items[-2] == 20
    assert items[-3] == 10


def test_flat_list_index_out_of_range(flat_list_factory):
    items = flat_list_factory("items", [1, 2, 3])
    with pytest.raises(IndexError):
        items[5]
    with pytest.raises(IndexError):
        items[-4]


def test_flat_list_setitem(flat_list_factory):
    items = flat_list_factory("items", [1, 2, 3])
    items[1] = 99
    assert items[0] == 1
    assert items[1] == 99
    assert items[2] == 3


def test_flat_list_len(flat_list_factory):
    items = flat_list_factory("items")
    assert len(items) == 0

    items.append(1)
    assert len(items) == 1

    items.append(2)
    items.append(3)
    assert len(items) == 3


def test_flat_list_iteration(flat_list_factory):
    items = flat_list_factory("items", [1, 2, 3, 4, 5])
    result = list(items)
    assert result == [1, 2, 3, 4, 5]


def test_flat_list_contains(flat_list_factory):
    items = flat_list_factory("items", [1, 2, 3])
    assert 2 in items
    assert 99 not in items


# ============================================================================
# MUTATION
# ============================================================================


def test_flat_list_extend(flat_list_factory):
    items = flat_list_factory("items", [1, 2])
    items.extend([3, 4, 5])
    assert items.extract() == [1, 2, 3, 4, 5]


def test_flat_list_clear(flat_list_factory):
    items = flat_list_factory("items", [1, 2, 3])
    items.clear()
    assert len(items) == 0
    assert items.extract() == []


# ============================================================================
# STORE AND EXTRACT
# ============================================================================


def test_flat_list_store_replaces_content(flat_list_factory):
    items = flat_list_factory("items", [1, 2, 3])
    items.store([4, 5, 6, 7])
    assert len(items) == 4
    assert items.extract() == [4, 5, 6, 7]


def test_flat_list_extract_returns_list(flat_list_factory):
    items = flat_list_factory("items", [10, 20, 30])
    extracted = items.extract()
    assert isinstance(extracted, list)
    assert extracted == [10, 20, 30]


def test_flat_list_extract_range(flat_list_factory):
    items = flat_list_factory("items", [10, 20, 30, 40, 50])
    assert items.extract_range(1, 4) == [20, 30, 40]
    assert items.extract_range(0, 2) == [10, 20]
    assert items.extract_range(3, 10) == [40, 50]


# ============================================================================
# MIXED PRIMITIVE TYPES
# ============================================================================


def test_flat_list_mixed_primitives(flat_list_factory):
    items = flat_list_factory("items", ["hello", 42, 3.14, True, None])
    extracted = items.extract()
    assert extracted[0] == "hello"
    assert extracted[1] == 42
    assert extracted[2] == 3.14
    assert extracted[3] is True
    assert extracted[4] is None
