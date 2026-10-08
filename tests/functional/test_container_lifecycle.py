"""Functional tests for container lifecycle operations.

Tests container creation, deletion, and descendant operations:
- create_container() - creating containers with parent validation
- delete_container() - deleting containers
- delete_descendants() - recursive deletion
- the container layer never creates parents
"""

import pytest

from virtuals.container import (
    ContainerExistsError,
    ContainerNotFoundError,
    ContainerProtocol,
    ContainerStructure,
    ContainerTypeError,
    create_container,
    delete_container,
    delete_descendants,
    get_node_info,
    get_node_type,
    node_exists,
    put_child_primitive,
)
from virtuals.container.types import NodeType
from virtuals.tkv.storage import TransactionProtocol


# ============================================================================
# CONTAINER CREATION TESTS
# ============================================================================


def test_create_container_basic(tx: TransactionProtocol) -> None:
    """Test basic container creation without parent validation."""
    create_container(
        ("users",),
        ContainerStructure(1),
        ContainerProtocol.MUTABLE,
        tx,
    )

    assert node_exists(("users",), tx)
    assert get_node_type(("users",), tx) == NodeType.CONTAINER

    info = get_node_info(("users",), tx)
    assert info.structure == ContainerStructure(1)
    assert info.protocol == ContainerProtocol.MUTABLE


def test_create_container_idempotent_compatible(tx: TransactionProtocol) -> None:
    """Test creating container twice with same type is idempotent."""
    create_container(
        ("users",),
        ContainerStructure(1),
        ContainerProtocol.MUTABLE,
        tx,
    )
    # Second call should be silent (idempotent)
    create_container(
        ("users",),
        ContainerStructure(1),
        ContainerProtocol.MUTABLE,
        tx,
    )

    # Should still exist with same type
    assert node_exists(("users",), tx)
    info = get_node_info(("users",), tx)
    assert info.structure == ContainerStructure(1)


def test_create_container_incompatible_type_raises(tx: TransactionProtocol) -> None:
    """Test creating container with incompatible type raises error."""
    create_container(
        ("users",),
        ContainerStructure(1),
        ContainerProtocol.MUTABLE,
        tx,
    )

    with pytest.raises(ContainerExistsError):
        create_container(
            ("users",),
            ContainerStructure(2),  # Different structure
            ContainerProtocol.MUTABLE,
            tx,
        )


def test_create_container_over_primitive_raises(tx: TransactionProtocol) -> None:
    """Test creating container where primitive exists raises error."""
    create_container(
        ("data",),
        ContainerStructure(1),
        ContainerProtocol.MUTABLE,
        tx,
    )
    put_child_primitive(("data",), "value", 42, tx)

    with pytest.raises(ContainerTypeError):
        create_container(
            ("data", "value"),
            ContainerStructure(1),
            ContainerProtocol.MUTABLE,
            tx,
        )


def test_create_container_various_protocols(tx: TransactionProtocol) -> None:
    """Test creating containers with various protocol combinations."""
    create_container(
        ("c1",),
        ContainerStructure(1),
        ContainerProtocol.NONE,
        tx,
    )
    create_container(
        ("c2",),
        ContainerStructure(2),
        ContainerProtocol.MUTABLE,
        tx,
    )
    create_container(
        ("c3",),
        ContainerStructure(3),
        ContainerProtocol.MUTABLE | ContainerProtocol.SIZED,
        tx,
    )
    create_container(
        ("c4",),
        ContainerStructure(4),
        ContainerProtocol.MUTABLE | ContainerProtocol.SIZED | ContainerProtocol.INDEXED,
        tx,
    )

    # Verify all created with correct protocols
    assert get_node_info(("c1",), tx).protocol == ContainerProtocol.NONE
    assert get_node_info(("c2",), tx).protocol == ContainerProtocol.MUTABLE
    assert (
        get_node_info(("c3",), tx).protocol == ContainerProtocol.MUTABLE | ContainerProtocol.SIZED
    )
    assert (
        get_node_info(("c4",), tx).protocol
        == ContainerProtocol.MUTABLE | ContainerProtocol.SIZED | ContainerProtocol.INDEXED
    )


# ============================================================================
# PARENT VALIDATION TESTS
# ============================================================================


def _create_chain(site: tuple, tx: TransactionProtocol) -> None:
    """Create every level of site, root first (the container layer creates one node)."""
    for depth in range(1, len(site) + 1):
        create_container(site[:depth], ContainerStructure(1), ContainerProtocol.MUTABLE, tx)


def test_create_container_missing_parent_raises(tx: TransactionProtocol) -> None:
    """Test creating container under a missing parent raises, creating nothing."""
    with pytest.raises(ContainerNotFoundError):
        create_container(
            ("a", "b", "c"),
            ContainerStructure(1),
            ContainerProtocol.MUTABLE,
            tx,
        )

    # No parents were auto-created, and no orphan target either
    assert not node_exists(("a",), tx)
    assert not node_exists(("a", "b"), tx)
    assert not node_exists(("a", "b", "c"), tx)


def test_create_container_partial_chain_raises(tx: TransactionProtocol) -> None:
    """Test a missing immediate parent raises even when higher ancestors exist."""
    _create_chain(("a",), tx)

    with pytest.raises(ContainerNotFoundError):
        create_container(
            ("a", "b", "c"),
            ContainerStructure(1),
            ContainerProtocol.MUTABLE,
            tx,
        )

    assert not node_exists(("a", "b"), tx)


def test_create_container_existing_parent(tx: TransactionProtocol) -> None:
    """Test creating container one level at a time succeeds."""
    _create_chain(("a", "b", "c"), tx)

    assert get_node_type(("a",), tx) == NodeType.CONTAINER
    assert get_node_type(("a", "b"), tx) == NodeType.CONTAINER
    assert get_node_type(("a", "b", "c"), tx) == NodeType.CONTAINER


def test_create_container_without_parent_validation(tx: TransactionProtocol) -> None:
    """Test validate_parent=False leaves the parent to the caller."""
    create_container(
        ("a", "b", "c"),
        ContainerStructure(1),
        ContainerProtocol.MUTABLE,
        tx,
        validate_parent=False,
    )

    assert node_exists(("a", "b", "c"), tx)
    # Parents are still never created
    assert not node_exists(("a",), tx)
    assert not node_exists(("a", "b"), tx)


def test_create_container_primitive_parent_raises(tx: TransactionProtocol) -> None:
    """Test creating container under a primitive raises error."""
    create_container(
        ("a",),
        ContainerStructure(1),
        ContainerProtocol.MUTABLE,
        tx,
    )
    put_child_primitive(("a",), "b", "wrong", tx)

    with pytest.raises(ContainerTypeError):
        create_container(
            ("a", "b", "c"),
            ContainerStructure(1),
            ContainerProtocol.MUTABLE,
            tx,
        )


# ============================================================================
# CONTAINER DELETION TESTS
# ============================================================================


def test_delete_container_basic(tx: TransactionProtocol) -> None:
    """Test basic container deletion."""
    create_container(
        ("users",),
        ContainerStructure(1),
        ContainerProtocol.MUTABLE,
        tx,
    )

    delete_container(("users",), tx)

    assert not node_exists(("users",), tx)


def test_delete_container_nonexistent(tx: TransactionProtocol) -> None:
    """Test deleting nonexistent container is silent (idempotent)."""
    # Should not raise - silent operation
    delete_container(("users",), tx)


def test_delete_container_primitive_raises(tx: TransactionProtocol) -> None:
    """Test deleting primitive as container raises error."""
    create_container(
        ("data",),
        ContainerStructure(1),
        ContainerProtocol.MUTABLE,
        tx,
    )
    put_child_primitive(("data",), "value", 42, tx)

    with pytest.raises(ContainerTypeError):
        delete_container(("data", "value"), tx)


def test_delete_container_with_children(tx: TransactionProtocol) -> None:
    """Test deleting container with children deletes entire descendants."""
    create_container(
        ("users",),
        ContainerStructure(1),
        ContainerProtocol.MUTABLE,
        tx,
    )
    put_child_primitive(("users",), "alice", {"name": "Alice"}, tx)
    put_child_primitive(("users",), "bob", {"name": "Bob"}, tx)

    delete_container(("users",), tx)

    assert not node_exists(("users",), tx)
    assert not node_exists(("users", "alice"), tx)
    assert not node_exists(("users", "bob"), tx)


def test_delete_container_deep_hierarchy(tx: TransactionProtocol) -> None:
    """Test deleting container with deep nested children."""
    _create_chain(("a", "b", "c", "d"), tx)
    put_child_primitive(("a", "b", "c", "d"), "value", "test", tx)

    # Delete intermediate container
    delete_container(("a", "b"), tx)

    assert node_exists(("a",), tx)  # Parent still exists
    assert not node_exists(("a", "b"), tx)
    assert not node_exists(("a", "b", "c"), tx)
    assert not node_exists(("a", "b", "c", "d"), tx)
    assert not node_exists(("a", "b", "c", "d", "value"), tx)


# ============================================================================
# DESCENDANTS DELETION TESTS
# ============================================================================


def test_delete_descendants_basic(tx: TransactionProtocol) -> None:
    """Test basic descendants deletion."""
    create_container(
        ("users",),
        ContainerStructure(1),
        ContainerProtocol.MUTABLE,
        tx,
    )

    delete_descendants(("users",), tx)

    assert not node_exists(("users",), tx)


def test_delete_descendants_with_children(tx: TransactionProtocol) -> None:
    """Test delete_descendants removes all descendants."""
    create_container(
        ("users",),
        ContainerStructure(1),
        ContainerProtocol.MUTABLE,
        tx,
    )
    put_child_primitive(("users",), "alice", {"name": "Alice"}, tx)
    put_child_primitive(("users",), "bob", {"name": "Bob"}, tx)

    delete_descendants(("users",), tx)

    assert not node_exists(("users",), tx)
    assert not node_exists(("users", "alice"), tx)
    assert not node_exists(("users", "bob"), tx)


def test_delete_descendants_deep_hierarchy(tx: TransactionProtocol) -> None:
    """Test delete_descendants with deeply nested structure."""
    # Create: users -> alice -> profile -> settings
    _create_chain(("users", "alice", "profile", "settings"), tx)
    put_child_primitive(("users", "alice", "profile", "settings"), "theme", "dark", tx)

    delete_descendants(("users", "alice"), tx)

    assert node_exists(("users",), tx)  # Parent still exists
    assert not node_exists(("users", "alice"), tx)


def test_delete_descendants_mixed_children(tx: TransactionProtocol) -> None:
    """Test delete_descendants with mixed containers and primitives."""
    create_container(
        ("root",),
        ContainerStructure(1),
        ContainerProtocol.MUTABLE,
        tx,
    )

    # Add primitive children
    put_child_primitive(("root",), "p1", "value1", tx)
    put_child_primitive(("root",), "p2", "value2", tx)

    # Add container children
    create_container(
        ("root", "c1"),
        ContainerStructure(1),
        ContainerProtocol.MUTABLE,
        tx,
    )
    put_child_primitive(("root", "c1"), "nested", "value", tx)

    delete_descendants(("root",), tx)

    assert not node_exists(("root",), tx)


def test_delete_descendants_nonexistent(tx: TransactionProtocol) -> None:
    """Test delete_descendants on nonexistent site is silent (idempotent)."""
    # Should not raise - silent operation
    delete_descendants(("nonexistent",), tx)


# ============================================================================
# EDGE CASES AND INTEGRATION
# ============================================================================


def test_create_delete_create_cycle(tx: TransactionProtocol) -> None:
    """Test creating, deleting, then recreating container works correctly."""
    # Create
    create_container(
        ("users",),
        ContainerStructure(1),
        ContainerProtocol.MUTABLE,
        tx,
    )
    assert node_exists(("users",), tx)

    # Delete
    delete_container(("users",), tx)
    assert not node_exists(("users",), tx)

    # Recreate
    create_container(
        ("users",),
        ContainerStructure(1),
        ContainerProtocol.MUTABLE,
        tx,
    )
    assert node_exists(("users",), tx)


def test_delete_preserves_siblings(tx: TransactionProtocol) -> None:
    """Test deleting container preserves sibling containers."""
    create_container(
        ("root",),
        ContainerStructure(1),
        ContainerProtocol.MUTABLE,
        tx,
    )
    create_container(
        ("root", "a"),
        ContainerStructure(1),
        ContainerProtocol.MUTABLE,
        tx,
    )
    create_container(
        ("root", "b"),
        ContainerStructure(1),
        ContainerProtocol.MUTABLE,
        tx,
    )
    create_container(
        ("root", "c"),
        ContainerStructure(1),
        ContainerProtocol.MUTABLE,
        tx,
    )

    # Delete one child
    delete_container(("root", "b"), tx)

    # Verify siblings still exist
    assert node_exists(("root",), tx)
    assert node_exists(("root", "a"), tx)
    assert not node_exists(("root", "b"), tx)
    assert node_exists(("root", "c"), tx)


def test_parent_validation_integration(tx: TransactionProtocol) -> None:
    """Test parent validation works correctly in complex scenarios."""
    # Create partial hierarchy
    _create_chain(("a", "b"), tx)

    # Two levels below the deepest existing container: the gap is not filled
    with pytest.raises(ContainerNotFoundError):
        create_container(
            ("a", "b", "c", "d"),
            ContainerStructure(1),
            ContainerProtocol.MUTABLE,
            tx,
        )
    assert not node_exists(("a", "b", "c"), tx)

    # One level at a time succeeds
    _create_chain(("a", "b", "c", "d"), tx)
    assert node_exists(("a", "b", "c", "d"), tx)
