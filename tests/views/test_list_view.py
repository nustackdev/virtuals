"""Functional tests for ListView."""


def test_list_factory_creates_empty_view(list_factory):
    """Verify list_factory creates an empty ListView."""
    items = list_factory("items")
    assert items is not None
    assert len(items) == 0


def test_list_factory_with_data(list_factory):
    """Verify list_factory can populate initial data."""
    items = list_factory("items", [1, 2, 3, 4, 5])

    data = items.extract()
    assert len(data) == 5
    assert data[0] == 1
    assert data[4] == 5


# ============================================================================
# BASIC OPERATIONS
# ============================================================================


def test_list_append(list_factory):
    """Test appending items."""
    items = list_factory("items")

    items.append(1)
    items.append(2)
    items.append(3)

    assert len(items) == 3
    assert items[0] == 1
    assert items[2] == 3


def test_list_indexing(list_factory):
    """Test index-based access."""
    items = list_factory("items", [10, 20, 30, 40, 50])

    assert items[0] == 10
    assert items[2] == 30
    assert items[4] == 50


def test_list_set_item(list_factory):
    """Test setting items by index."""
    items = list_factory("items", [1, 2, 3])

    items[1] = 99

    assert items[0] == 1
    assert items[1] == 99
    assert items[2] == 3


def test_list_len(list_factory):
    """Test length tracking."""
    items = list_factory("items")

    assert len(items) == 0

    items.append(1)
    assert len(items) == 1

    items.append(2)
    items.append(3)
    assert len(items) == 3


def test_list_iteration(list_factory):
    """Test iterating over list."""
    items = list_factory("items", [1, 2, 3, 4, 5])

    result = []
    for item in items:
        result.append(item)

    assert result == [1, 2, 3, 4, 5]


# ============================================================================
# LIST METHODS
# ============================================================================


def test_list_insert(list_factory):
    """Test insert operation."""
    items = list_factory("items", [1, 3, 4])

    items.insert(1, 2)  # Insert 2 at index 1

    extracted = items.extract()
    assert extracted == [1, 2, 3, 4]


def test_list_pop(list_factory):
    """Test pop operation."""
    items = list_factory("items", [1, 2, 3, 4, 5])

    value = items.pop()
    assert value == 5
    assert len(items) == 4

    value = items.pop(0)
    assert value == 1
    assert len(items) == 3


def test_list_clear(list_factory):
    """Test clear operation."""
    items = list_factory("items", [1, 2, 3, 4, 5])

    items.clear()

    assert len(items) == 0
    assert items.extract() == []


def test_list_contains(list_factory):
    """Test membership checking."""
    items = list_factory("items", [1, 2, 3, 4, 5])

    assert 3 in items
    assert 10 not in items


# ============================================================================
# NESTED DATA
# ============================================================================


def test_list_nested_dicts(list_factory):
    """Test lists containing dictionaries."""
    items = list_factory(
        "items",
        [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
        ],
    )

    extracted = items.extract()
    assert extracted[0]["name"] == "Alice"
    assert extracted[1]["age"] == 25


def test_list_nested_lists(list_factory):
    """Test nested list structures."""
    items = list_factory(
        "items",
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ],
    )

    extracted = items.extract()
    assert extracted[0] == [1, 2, 3]
    assert extracted[2][1] == 8


def test_list_mixed_types(list_factory):
    """Test list with mixed types."""
    items = list_factory(
        "items",
        [
            "string",
            42,
            3.14,
            True,
            {"key": "value"},
            [1, 2, 3],
        ],
    )

    extracted = items.extract()
    assert extracted[0] == "string"
    assert extracted[1] == 42
    assert extracted[4]["key"] == "value"
    assert extracted[5] == [1, 2, 3]


# ============================================================================
# STORE AND EXTRACT
# ============================================================================


def test_list_store_replaces_content(list_factory):
    """Test store() replaces existing content."""
    items = list_factory("items", [1, 2, 3])

    items.store([4, 5, 6, 7])

    assert len(items) == 4
    assert items.extract() == [4, 5, 6, 7]


def test_list_extract_full(list_factory):
    """Test extract() returns full list."""
    items = list_factory("items", [10, 20, 30, 40, 50])

    extracted = items.extract()

    assert extracted == [10, 20, 30, 40, 50]
    assert isinstance(extracted, list)


# ============================================================================
# MIDDLE-DELETE / INSERT WITH CONTAINER CHILDREN
# ============================================================================


def test_list_delitem_middle_container_child(list_factory):
    """Deleting a middle dict child shifts trailing container children down."""
    items = list_factory(
        "items",
        [
            {"title": "A", "year": 2001},
            {"title": "B", "year": 2002},
            {"title": "C", "year": 2003},
        ],
    )

    del items[1]

    assert len(items) == 2
    assert items.extract() == [
        {"title": "A", "year": 2001},
        {"title": "C", "year": 2003},
    ]


def test_list_delitem_first_container_child(list_factory):
    """Deleting index 0 of a list-of-dicts shifts every remaining child."""
    items = list_factory(
        "items",
        [
            {"title": "A"},
            {"title": "B"},
            {"title": "C"},
            {"title": "D"},
        ],
    )

    del items[0]

    assert len(items) == 3
    assert items.extract() == [
        {"title": "B"},
        {"title": "C"},
        {"title": "D"},
    ]


def test_list_delitem_last_container_child(list_factory):
    """Deleting the tail of a list-of-dicts still works (regression guard)."""
    items = list_factory(
        "items",
        [
            {"title": "A"},
            {"title": "B"},
            {"title": "C"},
        ],
    )

    del items[-1]

    assert len(items) == 2
    assert items.extract() == [{"title": "A"}, {"title": "B"}]


def test_list_delitem_nested_list_child(list_factory):
    """Middle-delete works when children are lists (nested containers)."""
    items = list_factory(
        "items",
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ],
    )

    del items[1]

    assert items.extract() == [[1, 2, 3], [7, 8, 9]]


def test_list_delitem_deeply_nested_container_child(list_factory):
    """Shifting a container child preserves its full subtree, not just the top layer."""
    items = list_factory(
        "items",
        [
            {"title": "A", "meta": {"tags": ["x", "y"], "score": 1}},
            {"title": "B", "meta": {"tags": ["z"], "score": 2}},
            {"title": "C", "meta": {"tags": ["w"], "score": 3}},
        ],
    )

    del items[0]

    assert items.extract() == [
        {"title": "B", "meta": {"tags": ["z"], "score": 2}},
        {"title": "C", "meta": {"tags": ["w"], "score": 3}},
    ]


def test_list_pop_first_container_child(list_factory):
    """pop(0) on a list-of-dicts returns the value and shifts the rest down."""
    items = list_factory(
        "items",
        [
            {"title": "A"},
            {"title": "B"},
            {"title": "C"},
        ],
    )

    value = items.pop(0)

    assert value == {"title": "A"}
    assert items.extract() == [{"title": "B"}, {"title": "C"}]


def test_list_insert_middle_with_container_children(list_factory):
    """Inserting in the middle of a list-of-dicts shifts trailing containers up."""
    items = list_factory(
        "items",
        [
            {"title": "A"},
            {"title": "C"},
        ],
    )

    items.insert(1, {"title": "B"})

    assert items.extract() == [
        {"title": "A"},
        {"title": "B"},
        {"title": "C"},
    ]


def test_list_insert_head_with_container_children(list_factory):
    """Inserting at head shifts every container child up by one."""
    items = list_factory(
        "items",
        [
            {"title": "B"},
            {"title": "C"},
        ],
    )

    items.insert(0, {"title": "A"})

    assert items.extract() == [
        {"title": "A"},
        {"title": "B"},
        {"title": "C"},
    ]


def test_list_remove_container_child_by_value(list_factory):
    """remove() finds and deletes a matching dict child via middle-delete."""
    items = list_factory(
        "items",
        [
            {"title": "A"},
            {"title": "B"},
            {"title": "C"},
        ],
    )

    items.remove({"title": "B"})

    assert items.extract() == [{"title": "A"}, {"title": "C"}]


def test_list_delitem_mixed_primitive_and_container_children(list_factory):
    """Deleting into a tail that mixes primitives and containers shifts both."""
    items = list_factory(
        "items",
        [
            "head",
            {"title": "A"},
            "middle",
            {"title": "B"},
        ],
    )

    del items[0]

    assert items.extract() == [{"title": "A"}, "middle", {"title": "B"}]
