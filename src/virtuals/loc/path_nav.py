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


def build_value_path(*segments: *tuple[PathViewSegment, ...], v: PathValueSegment) -> PathToValue:
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

    When all segments have static addresses (``is_address_static`` returns
    True), skips intermediate View/Container allocation and builds the final
    site tuple directly.

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

    # Fast-path the static prefix: build site directly, one View at the end
    current_view = start_view
    if static_end > 0:
        from virtuals.container import Container

        site = start_view.container.site
        for address, _ in path[:static_end]:
            site = (*site, address)
        pivot_type = path[static_end - 1][1]
        container = Container(ctx=start_view.container.ctx, site=site)
        current_view = pivot_type(container, start_view.registry)

    # Slow-path the remaining dynamic segments
    for address, expected_type in path[static_end:]:
        current_view = open_child_view(current_view, address, expected_type)

    return current_view


def navigate_and_ensure(
    start_view: View,
    path: PathToView,
) -> View:
    """Navigate a ViewPath and ensure every level is materialized with its
    declared view type.

    Sibling of ``navigate_view`` for the write path. Same walk shape; the
    difference is that at each step this calls ``ensure_created()`` on the
    child view -- stamping the container's marker with the DECLARED view
    type's structure and running the view's ``_ensure_internal_layout``
    hook (e.g. building ``__keys__/`` + ``__data__/`` sub-containers on
    log/indexed dict views).

    Why this exists: the container-layer auto-parent-creation
    (``ensure_healthy_parents=True`` inside ``Container.create``) is
    view-blind and stamps a hardcoded default structure at each missing
    ancestor. When a ref writes a leaf like ``blocks[100].committed``, the
    leaf's own ``ensure_created`` would auto-create ``/blocks`` with the
    default marker, silently -- so a later operation that opens
    ``/blocks`` as its DECLARED ``LogIndexedDictView`` (structure 15) hits
    a marker mismatch and raises ``ContainerExistsError``. Routing writes
    through this walk guarantees every ancestor is stamped with its
    declared type before any leaf write, so the invariant "any container
    reachable via a ref-write path has been ensured with its declared view
    type" holds by construction.

    Fast path: one existence probe on the deepest site in ``path``. If it
    already exists, every ancestor exists too (invariant maintained by
    prior walks through this same helper), so the walk is skipped and the
    view is opened directly. Hot-path cost is a single storage read --
    same as today's leaf-existence check in ``ensure_created``.

    Cold path: walk root -> leaf via ``open_child_view`` (pure navigation,
    zero storage), calling ``ensure_created`` at each level. Views whose
    markers already match short-circuit inside ``Container.create``, so
    ancestor levels are cheap.

    Args:
        start_view: Starting view (usually a Navigator root).
        path: ViewPath -- ``((addr_1, view_type_1), (addr_2, view_type_2), ...)``.
            Each ``view_type`` is the DECLARED class the ref layer expects
            for that level; ancestors get stamped with the corresponding
            structure IDs.

    Returns:
        The view at the end of the path, guaranteed materialized.
    """
    from virtuals.container import Container
    from virtuals.container.node_ops import node_exists

    if not path:
        start_view.ensure_created()
        return start_view

    # Fast path: if the deepest container already exists, ancestors are
    # already correct (invariant from prior walks). Skip the walk.
    leaf_site = start_view.container.site
    for address, _ in path:
        leaf_site = (*leaf_site, address)
    if node_exists(leaf_site, start_view.container.ctx):
        # Open the view at the final segment's declared class without
        # re-walking. This mirrors the fast-path in ``navigate_view``.
        leaf_view_type = path[-1][1]
        container = Container(ctx=start_view.container.ctx, site=leaf_site)
        return leaf_view_type(container, start_view.registry)

    # Cold path: walk with ensure at each level. ``open_child_view`` is
    # pure navigation; ``ensure_created`` on each level does one
    # ``get_node_info`` (silent return if the marker already matches) or
    # stamps + runs the view's ``_ensure_internal_layout`` hook.
    start_view.ensure_created()
    current_view = start_view
    for address, expected_type in path:
        current_view = open_child_view(current_view, address, expected_type)
        current_view.ensure_created()
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
