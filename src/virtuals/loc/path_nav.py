"""Path navigation system.

Minimal, practical navigation that works through Views using the Nestable protocol.
Views handle their own path translation (e.g., ListView negative indexing).

This module provides:
- Types: ViewPath, ValuePath, segments
- Path helpers: Build, split, join paths (~6 functions)
- Navigation: Traverse through Views (~4 functions)

That's it. Everything else is just tuple operations Python already gives you.
"""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from virtuals.view import View

    from .path_def import (
        Path,
        PathAddress,
        PathSegment,
        PathToValue,
        PathToView,
        PathValueSegment,
        PathViewSegment,
    )


__all__ = [  # noqa: RUF022
    # Path helpers
    "build_view_path",
    "build_value_path",
    "split_value_path",
    "parent_view_path",
    "last_segment",
    "split_path",
    # Navigation
    "open_child_view",
    "navigate_view",
    "navigate_value",
    "navigate_and_ensure",
    "open_parent_view",
]


# =============================================================================
# PATH HELPERS
# =============================================================================


def build_view_path(*segments: PathViewSegment) -> PathToView:
    """Build ViewPath from segments.

    Example:
        >>> path = build_view_path(
        ...     ("users", DictView),
        ...     ("alice", DictView),
        ... )
    """
    return segments


def build_value_path(*segments: PathViewSegment, v: PathValueSegment) -> PathToValue:
    """Build ValuePath from segments.

    Example:
        >>> path = build_value_path(
        ...     ("users", DictView),
        ...     ("alice", DictView),
        ...     v=("name", str),
        ... )
    """
    return (*segments, v)


def split_value_path(path: PathToValue) -> tuple[PathToView, PathValueSegment]:
    """Split ValuePath into parent ViewPath and final value segment.

    Args:
        path: ValuePath to split

    Returns:
        (parent ViewPath, value segment)

    Example:
        >>> path = (("users", DictView), ("alice", DictView), ("name", str))
        >>> parent, (address, type) = split_value_path(path)
        >>> # parent = (("users", DictView), ("alice", DictView))
        >>> # address = "name", type = str
    """
    return path[:-1], path[-1]


def split_path(path: Path, index: int) -> tuple[Path, Path]:
    """Split Path (both value and view) into parent Path and final value segment.

    Args:
        path: Path to split
        index: Position to split

    Returns:
        (parent Path, value segment)

    Example:
        >>> path = (("users", DictView), ("alice", DictView), ("name", str))
        >>> parent, (address, type) = split_value_path(path)
        >>> # parent = (("users", DictView), ("alice", DictView))
        >>> # address = "name", type = str
    """
    return path[:index], path[index:]  # type: ignore


def parent_view_path(path: Path) -> PathToView:
    """Get parent ViewPath by removing last segment.

    Example:
        >>> path = (("users", DictView), ("alice", DictView))
        >>> parent = parent_view_path(path)
        >>> # parent = (("users", DictView),)
    """
    return path[:-1]


def last_segment(path: Path) -> PathSegment:
    """Get last segment from path.

    Example:
        >>> path = (("users", DictView), ("alice", DictView))
        >>> last_segment(path)
        ("alice", DictView)
    """
    return path[-1]


# =============================================================================
# NAVIGATION
# =============================================================================


def _is_address_static(view_type: type, address: object) -> bool:
    """Check if a view type considers an address static (no normalization needed)."""
    checker = getattr(view_type, "is_address_static", None)
    return checker(address) if checker is not None else False


def open_child_view(
    parent_view: View,
    address: PathAddress,
    child_view_type: type[View],
) -> View:
    """Navigate from parent to child View.

    Uses Nestable protocol - parent View handles path translation.

    Args:
        parent_view: Parent view (must be Nestable)
        address: Address in parent's domain (e.g., -1 for ListView)
        child_view_type: Expected child View type

    Returns:
        Child view

    Example:
        >>> users = get_root_view(DictView, tx, registry)
        >>> alice = open_child_view(users, "alice", DictView)
        >>> tags = open_child_view(alice, "tags", ListView)
        >>> last = open_child_view(tags, -1, DictView)  # Negative index!
    """
    from virtuals.collections import is_nestable

    if not is_nestable(parent_view):
        raise TypeError(
            f"{type(parent_view).__name__} is not Nestable. Cannot navigate to children."
        )

    return parent_view.open_child(address, child_view_type)


def navigate_view(
    start_view: View,
    path: PathToView,
) -> View:
    """Navigate ViewPath to reach target View.

    Pure navigation, no storage writes. Every level is opened as a view
    whose ``parent`` is the level above, so the returned view always knows
    its full parent chain (a later write through it creates missing
    ancestors through their parents).

    When segments have static addresses (``is_address_static`` returns
    True), skips ``open_child`` and its address normalization, building
    each level's site directly.

    Args:
        start_view: Starting view
        path: ViewPath to navigate

    Returns:
        View at end of path

    Example:
        >>> root = get_root_view(DictView, tx, registry)
        >>> path = (("users", DictView), ("alice", DictView))
        >>> alice = navigate_view(root, path)
    """
    if not path:
        return start_view

    # Find longest static prefix — segments that can skip normalize_address
    static_end = 0
    for address, view_type in path:
        if not _is_address_static(view_type, address):
            break
        static_end += 1

    # Fast-path the static prefix: build sites directly, no normalization.
    # Still one View per level (pure allocation) so the parent chain holds.
    current_view = start_view
    if static_end > 0:
        from virtuals.container import Container

        ctx = start_view.container.ctx
        registry = start_view.registry
        for address, view_type in path[:static_end]:
            container = Container(ctx=ctx, site=(*current_view.container.site, address))
            current_view = view_type(container, registry, parent=current_view)  # type: ignore[call-arg]

    # Slow-path the remaining dynamic segments
    for address, expected_type in path[static_end:]:
        current_view = open_child_view(current_view, address, expected_type)

    return current_view


def navigate_and_ensure(
    start_view: View,
    path: PathToView,
) -> View:
    """Navigate a ViewPath and ensure every level is materialized.

    Ensures every level is materialized with its declared view type.

    Sibling of ``navigate_view`` for the write path. Every missing level
    is created by its parent view (``ensure_child``): stamped with the
    DECLARED view type's structure, its ``_ensure_internal_layout`` hook
    run (e.g. building ``__keys__/`` + ``__data__/`` sub-containers on
    log/indexed dict views), and recorded by the parent's
    ``_on_child_created`` (a ``DictView`` bumps its length, an indexed
    dict records the key).

    The container layer never creates parents, so this walk (or a write
    through any view opened by ``navigate_view``, which delegates to its
    parent chain the same way) is how deep writes into never-touched
    storage get their ancestors.

    Fast path: navigate in memory (``navigate_view``, no storage writes),
    then one existence probe on the leaf. If it exists, every ancestor
    exists too (the container layer never creates orphans), so the view is
    returned as-is. Hot-path cost on static paths is a single storage read.

    Cold path: walk root -> leaf with ``ensure_child`` at each level.
    Existing levels only validate their marker; missing ones are created
    through their parent.

    Args:
        start_view: Starting view (usually a Navigator root).
        path: ViewPath -- ``((addr_1, view_type_1), (addr_2, view_type_2), ...)``.
            Each ``view_type`` is the DECLARED class the ref layer expects
            for that level; ancestors get stamped with the corresponding
            structure IDs.

    Returns:
        The view at the end of the path, guaranteed materialized.
    """
    if not path:
        start_view.ensure_created()
        return start_view

    # Fast path: the leaf exists, so its ancestors do too. One probe.
    leaf_view = navigate_view(start_view, path)
    if leaf_view.container.exists():
        return leaf_view

    # Cold path: each parent creates its missing child.
    start_view.ensure_created()
    current_view = start_view
    for address, expected_type in path:
        current_view = current_view.ensure_child(address, expected_type)  # type: ignore[attr-defined]
    return current_view


def navigate_value(
    start_view: View,
    path: PathToValue,
) -> tuple[View, PathAddress]:
    """Navigate ValuePath and return (parent View, value address).

    This returns the parent View and address so you can do view.get(address) or
    view[address] yourself. Useful when you need the View for other operations.

    Args:
        start_view: Starting view
        path: PathToValue to navigate

    Returns:
        (parent View, value address) - call parent._get_child_value(address)

    Example:
        >>> root = get_root_view(DictView, tx, registry)
        >>> path = (("users", DictView), ("alice", DictView), ("name", str))
        >>> parent, address = navigate_value(root, path)
        >>> name = parent._get_child_value(address)  # or parent[address]
        >>> # name = "Alice"

        >>> # With negative indexing
        >>> path = (("users", DictView), ("alice", DictView), ("tags", ListView), (-1, str))
        >>> parent, address = navigate_value(root, path)
        >>> # parent is ListView, address is -1
        >>> # parent handles -1 → actual last index
    """
    if len(path) == 0:
        raise ValueError("Cannot navigate empty ValuePath")

    parent_path, (value_address, _) = split_value_path(path)

    if len(parent_path) > 0:
        parent_view = navigate_view(start_view, parent_path)
    else:
        parent_view = start_view

    return parent_view, value_address


def open_parent_view(child_view: View) -> View:
    """Navigate to parent view.

    Example:
        >>> alice = navigate_view(root, (("users", DictView), ("alice", DictView)))
        >>> users = open_parent_view(alice)
    """
    return child_view.open_parent()
