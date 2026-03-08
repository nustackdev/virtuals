"""End-to-end tests for FrozenSetView."""


def test_frozenset_factory_creates_empty_view(frozenset_factory):
    tags = frozenset_factory("tags")
    assert tags is not None
    assert len(tags) == 0


def test_frozenset_factory_with_data(frozenset_factory):
    tags = frozenset_factory("tags", {"python", "rust", "go"})
    data = tags.extract()
    assert len(data) == 3
    assert "python" in data
    assert "rust" in data
    assert "go" in data


# ============================================================================
# BASIC OPERATIONS
# ============================================================================


def test_frozenset_contains(frozenset_factory):
    tags = frozenset_factory("tags", {"python", "rust", "go"})
    assert "python" in tags
    assert "rust" in tags
    assert "javascript" not in tags


def test_frozenset_len(frozenset_factory):
    empty = frozenset_factory("empty", set())
    assert len(empty) == 0

    tags = frozenset_factory("tags", {"a", "b", "c"})
    assert len(tags) == 3


def test_frozenset_iteration(frozenset_factory):
    tags = frozenset_factory("tags", {"python", "rust", "go"})
    result = set()
    for tag in tags:
        result.add(tag)
    assert result == {"python", "rust", "go"}


# ============================================================================
# SET ALGEBRA
# ============================================================================


def test_frozenset_isdisjoint(frozenset_factory):
    a = frozenset_factory("a", {"python", "rust"})
    assert a.isdisjoint({"go", "java"})
    assert not a.isdisjoint({"rust", "java"})


def test_frozenset_issubset(frozenset_factory):
    a = frozenset_factory("a", {"python", "rust"})
    assert a.issubset({"python", "rust", "go"})
    assert not a.issubset({"python", "go"})


def test_frozenset_issuperset(frozenset_factory):
    a = frozenset_factory("a", {"python", "rust", "go"})
    assert a.issuperset({"python", "rust"})
    assert not a.issuperset({"python", "java"})


def test_frozenset_union(frozenset_factory):
    a = frozenset_factory("a", {"python", "rust"})
    result = a | {"go", "java"}
    assert result == frozenset({"python", "rust", "go", "java"})


def test_frozenset_intersection(frozenset_factory):
    a = frozenset_factory("a", {"python", "rust", "go"})
    result = a & {"rust", "go", "java"}
    assert result == frozenset({"rust", "go"})


def test_frozenset_difference(frozenset_factory):
    a = frozenset_factory("a", {"python", "rust", "go"})
    result = a - {"rust"}
    assert result == frozenset({"python", "go"})


def test_frozenset_symmetric_difference(frozenset_factory):
    a = frozenset_factory("a", {"python", "rust"})
    result = a ^ {"rust", "go"}
    assert result == frozenset({"python", "go"})


def test_frozenset_le(frozenset_factory):
    a = frozenset_factory("a", {"python", "rust"})
    assert a <= {"python", "rust", "go"}
    assert a <= {"python", "rust"}
    assert not (a <= {"python"})


def test_frozenset_ge(frozenset_factory):
    a = frozenset_factory("a", {"python", "rust", "go"})
    assert a >= {"python", "rust"}
    assert not (a >= {"python", "java"})


# ============================================================================
# STORE AND EXTRACT
# ============================================================================


def test_frozenset_store_replaces_content(frozenset_factory):
    tags = frozenset_factory("tags", {"python", "rust"})
    tags.store({"go", "java"})
    extracted = tags.extract()
    assert "python" not in extracted
    assert extracted == frozenset({"go", "java"})


def test_frozenset_extract_returns_frozenset(frozenset_factory):
    tags = frozenset_factory("tags", {"a", "b", "c"})
    extracted = tags.extract()
    assert isinstance(extracted, frozenset)
    assert extracted == frozenset({"a", "b", "c"})


# ============================================================================
# MIXED TYPES
# ============================================================================


def test_frozenset_mixed_primitives(frozenset_factory):
    data = frozenset_factory("data", {1, "two", 3.0})
    extracted = data.extract()
    assert 1 in extracted
    assert "two" in extracted
    assert 3.0 in extracted
