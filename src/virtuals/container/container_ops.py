"""Container operations for container layer.

This module provides container lifecycle management, child operations, and
traversal functionality. All operations work directly with storage and
delegate validation to the validation module.

All mutations are silent (return None) and idempotent.

Three-layer architecture for child operations:

- **General** (e.g. delete_child, iter_children): handles any child
  (primitive or container). Uses get_node_info + marker parsing.
  Container reads return NodeInfo — caller creates child containers
  imperatively. Container can't do nested reads or writes.

- **Primitive validated** (e.g. get_child_primitive, put_child_primitive):
  only handles primitives. Validates parent and/or asserts child is
  primitive. put_child_primitive has separate validate/validate_parent
  flags since writes are what ensure container chain integrity.

- **Unsafe primitive** (e.g. get_child_primitive_unsafe,
  put_child_primitive_unsafe, delete_child_primitive_unsafe): raw ctx.*
  calls. Caller guarantees container chain exists and children are
  primitives. Maximum performance, zero validation overhead.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from logging import getLogger
from typing import TYPE_CHECKING, TypeVar, cast

from virtuals.loc.constants import DATA_ROOT, METADATA_ROOT
from virtuals.tkv.filter import LengthFilter, PrefixFilter
from virtuals.tkv.storage import StorageScanOptions
from virtuals.types import EMPTY, Empty, Value, is_empty

from .context import (
    require_read_context,
    require_readwrite_context,
    require_write_context,
)
from .exceptions import (
    ContainerExistsError,
    ContainerInvalidDepthError,
    ContainerTypeError,
    absent_container,
)
from .marker import create_marker, is_marker
from .node_ops import get_node_info, get_node_type, node_exists
from .types import (
    ContainerProtocol,
    ContainerStructure,
    NodeInfo,
    NodeType,
)
from .validation_ops import (
    validate_compatible,
    validate_is_container,
    validate_is_primitive,
)


if TYPE_CHECKING:
    from collections.abc import Generator, Iterator

    from virtuals.loc import site as site_
    from virtuals.tkv.storage import ReadWriteAccessProtocol, StorageContextType

__all__ = [
    "clear_children",
    "clear_children_primitives_unsafe",
    "count_children",
    "create_child_container",
    "create_container",
    "delete_child",
    "delete_child_primitive_unsafe",
    "delete_container",
    "delete_descendants",
    "exists_child",
    "get_child_primitive",
    "get_child_primitive_unsafe",
    "get_child_type",
    "has_children",
    "iter_child_keys",
    "iter_child_primitive_values",
    "iter_child_values",
    "iter_children",
    "iter_descendants",
    "move_child_subtree",
    "put_child_primitive",
    "put_child_primitive_unsafe",
    "reject_empty_value",
    "scan_or_absent",
    "walk_descendants",
]


logger = getLogger(__name__)

T = TypeVar("T")

# ============================================================================
# CONTAINER LIFECYCLE
# ============================================================================


def create_container(
    site: site_.Site,
    structure: ContainerStructure,
    protocol: ContainerProtocol,
    ctx: StorageContextType,
    *,
    validate_parent: bool = True,
    node_info: NodeInfo | None = None,
) -> None:
    """Create container at site.

    Creates exactly one node. The parent must already exist as a container;
    the container layer never creates parents. Missing ancestors are the
    caller's job, normally a parent view creating its child
    (``View.ensure_child``), so every level is stamped with its own view
    type and the parent view can keep its bookkeeping.

    Idempotent: silent if container already exists with compatible type.

    Args:
        site: Container site
        structure: Container structure ID
        protocol: Container protocol flags
        ctx: Storage context (transaction)
        validate_parent: If True, validate the parent is a container
            (default True). Set to False only when the caller has already
            ensured the parent exists. Ignored for the data root, which
            has no parent.
        node_info: Prefetched node info for site (optional)

    Raises:
        ContainerExistsError: If exists with incompatible type
        ContainerNotFoundError: If the parent is missing (when validate_parent=True)
        ContainerTypeError: If type conflicts prevent creation
        StorageInterfaceError: If context doesn't support required operations

    Example:
        >>> from virtuals.container import (
        ...     create_container,
        ...     ContainerStructure,
        ...     ContainerProtocol,
        ... )
        >>> create_container(("/",), ContainerStructure(1), ContainerProtocol.MUTABLE, tx)
        >>> create_container(
        ...     ("/", "users"), ContainerStructure(1), ContainerProtocol.MUTABLE, tx
        ... )
        >>> # Raises ContainerNotFoundError: ("/", "teams") does not exist
        >>> create_container(
        ...     ("/", "teams", "core"), ContainerStructure(1), ContainerProtocol.MUTABLE, tx
        ... )
    """
    # Check if already exists
    node_info = get_node_info(site, ctx) if node_info is None else node_info

    # Validate type consistency if node already exists
    if node_info.exists:
        # Primitive check
        if node_info.node_type != NodeType.CONTAINER:
            logger.error(
                "Site exists as non-container, cannot create container",
                extra={"site": site, "node_type": node_info.node_type.name},
            )
            raise ContainerTypeError(
                f"Site exists as non-container, cannot create container: {site}"
            )

        # Existing container type compatibility
        try:
            validate_compatible(site, structure, protocol, ctx, node_info=node_info)
            logger.debug(
                "Container already exists with compatible type",
                extra={"site": site, "structure": structure, "protocol": protocol},
            )
            return  # Already exists with compatible type
        except ContainerTypeError:
            logger.error(
                "Container exists with incompatible type",
                extra={"site": site, "structure": structure, "protocol": protocol},
            )
            raise ContainerExistsError(f"Container exists with incompatible type: {site}") from None

    # Parent must exist: the container layer creates exactly one node
    if validate_parent and len(site) > 1:
        validate_is_container(site[:-1], ctx)

    # Create container
    marker = create_marker(structure, protocol)

    wctx = require_write_context(ctx)
    wctx.put(site, marker)

    logger.info(
        "Container created",
        extra={"site": site, "structure": structure, "protocol": protocol},
    )


def delete_container(site: site_.Site, ctx: StorageContextType) -> None:
    """Delete container and all descendants.

    Idempotent: silent if container doesn't exist.

    Args:
        site: Container site
        ctx: Storage context (transaction)

    Raises:
        ContainerTypeError: If site exists but is not a container
        StorageInterfaceError: If context doesn't support required operations
    """
    info = get_node_info(site, ctx)
    if not info.exists:
        logger.debug("Container does not exist, nothing to delete", extra={"site": site})
        return

    if info.node_type != NodeType.CONTAINER:
        logger.error(
            "Cannot delete container, site is not a container",
            extra={"site": site, "node_type": info.node_type.name},
        )
        raise ContainerTypeError(f"Site is not a container: {site}")

    delete_descendants(site, ctx)


def delete_descendants(site: site_.Site, ctx: StorageContextType) -> None:
    """Delete site and all descendants, with their metadata.

    For a site in the data tree the mirrored ``/m`` subtree goes too, so a
    view opened later at the same site (of any type) starts from a clean
    slate instead of inheriting bookkeeping left by a previous occupant.
    Two bounded prefix scans, one per tree; no backend has a range delete.

    Idempotent: silent if nothing exists at site.

    Args:
        site: Site to delete (with all descendants)
        ctx: Storage context (transaction)

    Raises:
        StorageInterfaceError: If context doesn't support required operations
    """
    rwctx = require_readwrite_context(ctx)

    _delete_subtree(site, rwctx)
    if site and site[0] == DATA_ROOT:
        _delete_subtree((METADATA_ROOT, *site[1:]), rwctx)

    logger.debug("Descendants deleted", extra={"site": site})


def _delete_subtree(site: site_.Site, rwctx: ReadWriteAccessProtocol) -> None:
    """Delete every key under ``site`` (``site`` itself included)."""
    prefix = PrefixFilter(prefix=site)
    scan_opts = StorageScanOptions(
        start=site,
        break_filter=prefix,
        filter=prefix,
    )
    for key in rwctx.scan(scan_opts).keys():
        rwctx.delete(key)


def move_child_subtree(
    parent_site: site_.Site,
    src_key: site_.SiteSegment,
    dst_key: site_.SiteSegment,
    ctx: StorageContextType,
) -> None:
    """Move a direct child (with its whole subtree) from src_key to dst_key.

    Rewrites every storage entry under ``(*parent_site, src_key)`` to the
    same relative location under ``(*parent_site, dst_key)``, then deletes
    the originals. Works for primitive and container children alike.

    Caller must ensure the dst_key slot is empty; existing entries under
    dst_key are not cleared. Raw storage rewrite — no observability, no
    validation, no marker parsing.

    O(n) in the number of storage entries under src_key. Prefer key-based
    collections (dicts) over lists when middle-position mutations of large
    subtrees are common.

    Args:
        parent_site: Parent container site
        src_key: Current child key
        dst_key: Target child key
        ctx: Storage context (transaction)
    """
    if src_key == dst_key:
        return

    rwctx = require_readwrite_context(ctx)

    src_site = (*parent_site, src_key)
    dst_site = (*parent_site, dst_key)

    prefix = PrefixFilter(prefix=src_site)
    scan_opts = StorageScanOptions(
        start=src_site,
        break_filter=prefix,
        filter=prefix,
    )

    # Materialize before mutating — modifying storage mid-scan is unsafe.
    items = list(rwctx.scan(scan_opts).items())

    src_prefix_len = len(src_site)
    for key, value in items:
        new_key = (*dst_site, *key[src_prefix_len:])
        rwctx.put(new_key, value)
        rwctx.delete(key)


# ============================================================================
# DIRECT CHILDREN QUERIES
# ============================================================================


def exists_child(
    site: site_.Site,
    key: site_.SiteSegment,
    ctx: StorageContextType,
    *,
    strict: bool = False,
    owner: site_.Site | None = None,
) -> bool:
    """Check if direct child exists.

    Args:
        site: Container site
        key: Child key
        ctx: Storage context (transaction, snapshot or write batch)
        strict: If True, raise ``ContainerNotFoundError`` when this
            container was never written (checked only on an empty result)
        owner: Site to name in that error (default this container's site)

    Returns:
        True if child exists

    Raises:
        StorageInterfaceError: If context doesn't support read access

    Example:
        >>> exists_child(("users",), "alice", tx)
        True
        >>> exists_child(("users",), "unknown", tx)
        False
    """
    child_site = (*site, key)
    child_type = get_node_type(child_site, ctx)
    if child_type != NodeType.NOT_FOUND:
        return True
    if strict and not node_exists(site, ctx):
        raise absent_container(owner or site)
    return False


def get_child_type(site: site_.Site, key: site_.SiteSegment, ctx: StorageContextType) -> NodeType:
    """Get type of direct child.

    Args:
        site: Container site
        key: Child key
        ctx: Storage context (transaction, snapshot or write batch)

    Returns:
        NodeType of child (CONTAINER, PRIMITIVE, or NOT_FOUND)

    Raises:
        StorageInterfaceError: If context doesn't support read access
    """
    child_site = (*site, key)
    return get_node_type(child_site, ctx)


def _child_scan_options(
    site: site_.Site,
    ctx: StorageContextType | None = None,
    *,
    reverse: bool = False,
) -> StorageScanOptions:
    """Build scan options for iterating direct children of a site.

    Returns StorageScanOptions with PrefixFilter (break on prefix mismatch)
    and LengthFilter (only direct children, not deeper descendants).
    ``max_depth`` makes the backend seek over each child's subtree instead
    of walking it, so the scan costs O(children), not O(descendants).

    Forward (default): ``start=site`` — every child key sorts ``> site``
    encoded, so seek lands at the first child.

    Reverse (``reverse=True``): ``start_encoded`` is set to the codec's
    ``upper_bound_of_prefix(site)``. In reverse mode ``start`` is treated
    as an inclusive upper bound; encoded ``site`` sorts *below* every
    child (the forward path relies on this), so it would yield nothing.
    The upper-bound sentinel sorts strictly *above* every child, which is
    what a reverse scan actually needs. ``ctx`` is required for reverse
    since we reach for the storage's codec.
    """
    prefix = PrefixFilter(prefix=site)
    child_len = LengthFilter(length=len(site) + 1)
    if reverse:
        if ctx is None:
            raise ValueError("reverse child scan requires a storage context")
        codec = ctx._storage.codec  # type: ignore[attr-defined]
        return StorageScanOptions(
            start_encoded=codec.upper_bound_of_prefix(site),
            reverse=True,
            break_filter=prefix,
            filter=prefix & child_len,
            max_depth=len(site) + 1,
        )
    return StorageScanOptions(
        start=site,
        break_filter=prefix,
        filter=prefix & child_len,
        max_depth=len(site) + 1,
    )


def _strict_child_scan(
    site: site_.Site,
    ctx: StorageContextType,
    *,
    reverse: bool = False,
    values: bool = False,
    limit: int | None = None,
    owner: site_.Site | None = None,
) -> Generator[tuple[site_.Site, object], None, None]:
    """Direct children of ``site``, raising if the container itself is absent.

    The child scan already starts at (forward) or ends on (reverse) the
    container's own marker row, so the scan keeps that row instead of
    filtering it out: seeing it, or any child, proves the container exists
    and costs no extra read. A scan that sees nothing raises
    ``ContainerNotFoundError`` naming ``owner`` (default ``site``).
    ``max_depth`` keeps every row at the site or one level below.

    ``limit`` counts children; the marker row is fetched on top of it.
    """
    base = _child_scan_options(site, ctx, reverse=reverse)
    options = replace(
        base,
        filter=base.break_filter,
        limit=None if limit is None else limit + 1,
    )
    scan = require_read_context(ctx).scan(options)
    depth = len(site) + 1
    seen = False
    yielded = 0
    rows = scan.items() if values else scan.keys()
    try:
        for row in rows:
            seen = True
            key = row[0] if values else row
            if len(key) != depth:
                continue  # the marker row (or a stray shallower row)
            yield key, (row[1] if values else None)
            yielded += 1
            if limit is not None and yielded >= limit:
                return
    finally:
        close = getattr(rows, "close", None)
        if close is not None:
            close()
    if not seen:
        raise absent_container(owner or site)


def scan_or_absent(
    rows: Iterator[T],
    site: site_.Site,
    ctx: StorageContextType,
    *,
    owner: site_.Site | None = None,
) -> Generator[T, None, None]:
    """Pass ``rows`` through; if none came, require the container at ``site``.

    For views that run their own scans (cursors, limits, internal layout):
    any row proves the container exists, so only an empty scan costs one
    marker read, raising ``ContainerNotFoundError`` (naming ``owner``,
    default ``site``) when the container was never written. An empty scan
    on an existing container ends normally.
    """
    seen = False
    try:
        for row in rows:
            seen = True
            yield row
    finally:
        close = getattr(rows, "close", None)
        if close is not None:
            close()
    if not seen and not node_exists(site, ctx):
        raise absent_container(owner or site)


def iter_child_keys(
    site: site_.Site,
    ctx: StorageContextType,
    validate: bool = False,
    *,
    reverse: bool = False,
    strict: bool = False,
    owner: site_.Site | None = None,
) -> Generator[site_.SiteSegment, None, None]:
    """Iterate over direct child keys.

    Args:
        site: Container site
        ctx: Storage context (transaction, snapshot or write batch)
        validate: If True, validate site is a container (default False)
        reverse: If True, yield in reverse insertion order (default False)
        strict: If True, raise ``ContainerNotFoundError`` when this
            container was never written (checked only on an empty result)
        owner: Site to name in that error (default this container's site)

    Yields:
        Child keys (last segment of each child site)

    Raises:
        ContainerNotFoundError: If container doesn't exist (when validate=True)
        ContainerTypeError: If site is not a container (when validate=True)
        StorageInterfaceError: If context doesn't support read access

    Example:
        >>> list(iter_child_keys(("users",), tx))
        ["alice", "bob", "charlie"]
    """
    if validate:
        validate_is_container(site, ctx)

    if strict:
        for key, _ in _strict_child_scan(site, ctx, reverse=reverse, owner=owner):
            yield key[-1]
        return

    for key in (
        require_read_context(ctx).scan(_child_scan_options(site, ctx, reverse=reverse)).keys()
    ):
        yield key[-1]


def iter_child_values(
    site: site_.Site,
    ctx: StorageContextType,
    validate: bool = False,
    *,
    reverse: bool = False,
) -> Generator[NodeInfo, None, None]:
    """Iterate over direct child node info.

    Args:
        site: Container site
        ctx: Storage context (transaction, snapshot or write batch)
        validate: If True, validate site is a container (default False)
        reverse: If True, yield in reverse insertion order (default False)

    Yields:
        NodeInfo for each direct child

    Raises:
        ContainerNotFoundError: If container doesn't exist (when validate=True)
        ContainerTypeError: If site is not a container (when validate=True)
        StorageInterfaceError: If context doesn't support read access
    """
    if validate:
        validate_is_container(site, ctx)

    for key, value in (
        require_read_context(ctx).scan(_child_scan_options(site, ctx, reverse=reverse)).items()
    ):
        yield get_node_info(key, ctx, raw_value=value)


def iter_children(
    site: site_.Site,
    ctx: StorageContextType,
    validate: bool = False,
    *,
    reverse: bool = False,
    strict: bool = False,
    owner: site_.Site | None = None,
) -> Generator[tuple[site_.SiteSegment, NodeInfo], None, None]:
    """Iterate over direct children with their info.

    Args:
        site: Container site
        ctx: Storage context (transaction, snapshot or write batch)
        validate: If True, validate site is a container (default False)
        reverse: If True, yield in reverse insertion order (default False)
        strict: If True, raise ``ContainerNotFoundError`` when this
            container was never written (checked only on an empty result)
        owner: Site to name in that error (default this container's site)

    Yields:
        Tuples of (child_key, NodeInfo)

    Raises:
        ContainerNotFoundError: If container doesn't exist (when validate=True)
        ContainerTypeError: If site is not a container (when validate=True)
        StorageInterfaceError: If context doesn't support read access
    """
    if validate:
        validate_is_container(site, ctx)

    if strict:
        for key, value in _strict_child_scan(site, ctx, reverse=reverse, values=True, owner=owner):
            yield (key[-1], get_node_info(key, ctx, raw_value=cast("Value", value)))
        return

    for key, value in (
        require_read_context(ctx).scan(_child_scan_options(site, ctx, reverse=reverse)).items()
    ):
        yield (key[-1], get_node_info(key, ctx, raw_value=value))


def count_children(
    site: site_.Site,
    ctx: StorageContextType,
    validate: bool = False,
    *,
    strict: bool = False,
    owner: site_.Site | None = None,
) -> int:
    """Count direct children.

    Args:
        site: Container site
        ctx: Storage context (transaction, snapshot or write batch)
        validate: If True, validate site is a container (default False)
        strict: If True, raise ``ContainerNotFoundError`` when this
            container was never written (checked only on an empty result)
        owner: Site to name in that error (default this container's site)

    Returns:
        Number of direct children

    Raises:
        ContainerNotFoundError: If container doesn't exist (when validate=True)
        ContainerTypeError: If site is not a container (when validate=True)
        StorageInterfaceError: If context doesn't support read access
    """
    if validate:
        validate_is_container(site, ctx)

    if strict:
        return sum(1 for _ in _strict_child_scan(site, ctx, owner=owner))

    counter = 0
    for _ in require_read_context(ctx).scan(_child_scan_options(site)).keys():
        counter += 1
    return counter


def has_children(
    site: site_.Site,
    ctx: StorageContextType,
    validate: bool = False,
    *,
    strict: bool = False,
    owner: site_.Site | None = None,
) -> bool:
    """Whether site has at least one direct child.

    One child scan stopped at its first hit, so it visits at most the
    site's own row and one child whatever the container's size.

    Args:
        site: Container site
        ctx: Storage context (transaction, snapshot or write batch)
        validate: If True, validate site is a container (default False)
        strict: If True, raise ``ContainerNotFoundError`` when this
            container was never written (checked only on an empty result)
        owner: Site to name in that error (default this container's site)

    Returns:
        True if at least one direct child exists

    Raises:
        ContainerNotFoundError: If container doesn't exist (when validate=True)
        ContainerTypeError: If site is not a container (when validate=True)
        StorageInterfaceError: If context doesn't support read access
    """
    if validate:
        validate_is_container(site, ctx)

    if strict:
        rows = _strict_child_scan(site, ctx, limit=1, owner=owner)
        try:
            next(rows)
        except StopIteration:
            return False
        else:
            return True
        finally:
            rows.close()

    options = replace(_child_scan_options(site), limit=1)
    keys = require_read_context(ctx).scan(options).keys()
    try:
        next(keys)
    except StopIteration:
        return False
    else:
        return True
    finally:
        keys.close()


# ============================================================================
# DIRECT CHILDREN MANIPULATION
# ============================================================================


def create_child_container(
    parent_site: site_.Site,
    key: site_.SiteSegment,
    structure: ContainerStructure,
    protocol: ContainerProtocol,
    ctx: StorageContextType,
    validate: bool = True,
) -> None:
    """Create child container.

    Idempotent: silent if child container already exists with compatible type.

    Args:
        parent_site: Parent container site
        key: Child key
        structure: Container structure ID
        protocol: Container protocol flags
        ctx: Storage context (transaction)
        validate: If True, validate parent is a container (default True)

    Raises:
        ContainerNotFoundError: If parent doesn't exist (when validate=True)
        ContainerTypeError: If parent is not a container (when validate=True)
        StorageInterfaceError: If context doesn't support required operations
    """
    child_site = (*parent_site, key)
    create_container(child_site, structure, protocol, ctx, validate_parent=validate)


def reject_empty_value(site: site_.Site, value: object) -> None:
    """Refuse to store EMPTY: it is the absent sentinel, never a value.

    A stored EMPTY row reads as absent (``get_node_info``) while scans still
    see it, so the container disagrees with itself. Callers that mean
    "absent" delete the key instead.

    Compound values (mappings, lists, tuples, sets) are walked in Python, so
    a nested EMPTY is refused before any part of the value is written.
    O(value size), no storage touch.

    Raises:
        ValueError: If ``value`` is or contains EMPTY
    """
    path = _find_empty(value, ())
    if path is not None:
        where = f"{site}" if not path else f"{site} (nested at {path})"
        raise ValueError(
            f"Cannot store EMPTY at {where}: EMPTY means absent, delete the key instead"
        )


def _find_empty(value: object, path: tuple[object, ...]) -> tuple[object, ...] | None:
    """Path to the first EMPTY inside ``value``, None if there is none."""
    if is_empty(value):
        return path
    if isinstance(value, Mapping):
        for key, item in value.items():
            found = _find_empty(item, (*path, key))
            if found is not None:
                return found
    elif isinstance(value, list | tuple | set | frozenset):
        for index, item in enumerate(value):
            found = _find_empty(item, (*path, index))
            if found is not None:
                return found
    return None


def put_child_primitive(
    parent_site: site_.Site,
    key: site_.SiteSegment,
    value: Value,
    ctx: StorageContextType,
    validate: bool = True,
    validate_parent: bool = True,
) -> None:
    """Put primitive child value.

    Idempotent: overwrites if already exists.

    Args:
        parent_site: Parent container site
        key: Child key
        value: Primitive value
        ctx: Storage context (transaction)
        validate: If True, check child isn't already a container (default True)
        validate_parent: If True, validate parent is a container (default True).
            Set to False when the caller has already ensured the parent exists
            (e.g. via ensure_created()).

    Raises:
        ValueError: If ``value`` is EMPTY (the absent sentinel is never stored)
        ContainerNotFoundError: If parent doesn't exist (when validate_parent=True)
        ContainerTypeError: If parent is not a container (when validate_parent=True)
            or child is a container (when validate=True)
        StorageInterfaceError: If context doesn't support required operations

    Example:
        >>> put_child_primitive(("users",), "alice", {"name": "Alice", "age": 30}, tx)
        >>> get_child_primitive(("users",), "alice", tx)
        {"name": "Alice", "age": 30}
    """
    child_site = (*parent_site, key)
    reject_empty_value(child_site, value)

    if validate_parent:
        validate_is_container(parent_site, ctx)

    if validate:
        child_node_info = get_node_info(child_site, ctx)
        if child_node_info.exists:
            validate_is_primitive(child_site, ctx, node_type=child_node_info.node_type)

    require_write_context(ctx).put(child_site, value)

    logger.debug(
        "Child primitive set",
        extra={
            "parent_site": parent_site,
            "key": key,
            "value_type": type(value).__name__,
        },
    )


def put_child_primitive_unsafe(
    parent_site: site_.Site,
    key: site_.SiteSegment,
    value: Value,
    ctx: StorageContextType,
) -> None:
    """Put primitive child value: raw ctx.put(), no storage validation.

    The caller must guarantee:
    - Parent container chain exists
    - Child is a primitive (not a container)

    The value is still checked for EMPTY: that check reads no storage and
    EMPTY is never stored through any door.

    Args:
        parent_site: Parent container site
        key: Child key
        value: Primitive value
        ctx: Storage context (transaction)

    Raises:
        ValueError: If ``value`` is EMPTY (the absent sentinel is never stored)
    """
    child_site = (*parent_site, key)
    reject_empty_value(child_site, value)
    require_write_context(ctx).put(child_site, value)


def get_child_primitive(
    parent_site: site_.Site,
    key: site_.SiteSegment,
    ctx: StorageContextType,
) -> Value | Empty:
    """Get primitive child value with validation.

    Parses markers and asserts child is a primitive (not a container).
    No parent validation — reads don't affect container integrity.

    Args:
        parent_site: Parent container site
        key: Child key
        ctx: Storage context (transaction, snapshot or write batch)

    Returns:
        Primitive value or EMPTY if child doesn't exist

    Raises:
        ContainerTypeError: If child is a container
        StorageInterfaceError: If context doesn't support read access
    """
    child_site = (*parent_site, key)
    child_node_info = get_node_info(child_site, ctx)

    if not child_node_info.exists:
        return EMPTY

    validate_is_primitive(child_site, ctx, node_type=child_node_info.node_type)

    return cast("Value", child_node_info.primitive_value)


def get_child_primitive_unsafe(
    parent_site: site_.Site,
    key: site_.SiteSegment,
    ctx: StorageContextType,
) -> Value | Empty:
    """Get primitive child value — raw ctx.get(), no validation.

    No marker parsing, no type checks. The caller must know the child
    is a primitive.

    Args:
        parent_site: Parent container site
        key: Child key
        ctx: Storage context (transaction, snapshot or write batch)

    Returns:
        Raw value or EMPTY if child doesn't exist
    """
    child_site = (*parent_site, key)
    return cast("Value", require_read_context(ctx).get(child_site))


def delete_child(
    parent_site: site_.Site,
    key: site_.SiteSegment,
    ctx: StorageContextType,
) -> None:
    """Delete direct child.

    Idempotent: silent if child doesn't exist.
    If child is a container, deletes all its descendants too.

    Args:
        parent_site: Parent container site
        key: Child key
        ctx: Storage context (transaction)

    Raises:
        ContainerNotFoundError: If parent doesn't exist
        ContainerTypeError: If parent is not a container
        StorageInterfaceError: If context doesn't support required operations
    """
    validate_is_container(parent_site, ctx)

    child_site = (*parent_site, key)
    info = get_node_info(child_site, ctx)

    if not info.exists:
        logger.debug(
            "Child does not exist, nothing to delete",
            extra={"parent_site": parent_site, "key": key},
        )
        return

    if info.node_type == NodeType.CONTAINER:
        delete_descendants(child_site, ctx)
    else:
        require_write_context(ctx).delete(child_site)

    logger.debug(
        "Child deleted",
        extra={
            "parent_site": parent_site,
            "key": key,
            "node_type": info.node_type.name,
        },
    )


def delete_child_primitive_unsafe(
    parent_site: site_.Site,
    key: site_.SiteSegment,
    ctx: StorageContextType,
) -> None:
    """Delete primitive child value — raw ctx.delete(), no validation.

    Single ctx.delete() — no parent validation, no get_node_info,
    no descendant cleanup. The caller must know the child is a primitive.

    For validated deletes that handle both primitives and containers
    (including subtree cleanup), use delete_child() instead.

    Args:
        parent_site: Parent container site
        key: Child key
        ctx: Storage context (transaction)
    """
    child_site = (*parent_site, key)
    require_write_context(ctx).delete(child_site)


def iter_child_primitive_values(
    site: site_.Site,
    ctx: StorageContextType,
) -> Generator[object, None, None]:
    """Iterate over direct primitive child values.

    Raw storage scan — no marker parsing, no type checks.
    The caller must know all children are primitives.

    Args:
        site: Container site
        ctx: Storage context (transaction, snapshot or write batch)

    Yields:
        Raw primitive values
    """
    yield from require_read_context(ctx).scan(_child_scan_options(site)).values()


def clear_children(
    site: site_.Site,
    ctx: StorageContextType,
    validate: bool = True,
) -> None:
    """Delete all direct children.

    Idempotent: silent if no children exist.

    Args:
        site: Container site
        ctx: Storage context (transaction)
        validate: If True, validate site is a container (default True)

    Raises:
        ContainerNotFoundError: If container doesn't exist (when validate=True)
        ContainerTypeError: If site is not a container (when validate=True)
        StorageInterfaceError: If context doesn't support required operations
    """
    # Pass validate=False to iter_child_keys since we validate once here
    if validate:
        validate_is_container(site, ctx)

    for key in iter_child_keys(site, ctx, validate=False):
        delete_child(site, key, ctx)

    logger.debug("Children cleared", extra={"site": site})


def clear_children_primitives_unsafe(
    site: site_.Site,
    ctx: StorageContextType,
) -> None:
    """Delete all direct primitive children.

    Raw scan + ctx.delete() each — no validation, no descendant cleanup.
    The caller must know all children are primitives.

    Args:
        site: Container site
        ctx: Storage context (transaction)
    """
    wctx = require_write_context(ctx)
    for key in require_read_context(ctx).scan(_child_scan_options(site)).keys():
        wctx.delete(key)


# ============================================================================
# RECURSIVE OPERATIONS
# ============================================================================


def iter_descendants(
    site: site_.Site,
    ctx: StorageContextType,
    *,
    depth: int = -1,
) -> Generator[site_.Site, None, None]:
    """Iterate over all descendants.

    Args:
        site: Container site
        ctx: Storage context (transaction, snapshot or write batch)
        depth: Depth to traverse (-1=unlimited, 1=children, >1 exact depth match)

    Yields:
        Descendant sites

    Raises:
        ContainerNotFoundError: If container doesn't exist
        ContainerTypeError: If site is not a container
        StorageInterfaceError: If context doesn't support read access
        ContainerInvalidDepthError: If depth argument is invalid

    Example:
        >>> # Get all descendants at any depth
        >>> list(iter_descendants(("users",), tx))
        [("users",), ("users", "alice"), ("users", "alice", "profile"), ("users", "bob")]
        >>> # Get only direct children (depth=1)
        >>> list(iter_descendants(("users",), tx, depth=1))
        [("users", "alice"), ("users", "bob")]
    """
    if depth < -2 or depth == 0:
        raise ContainerInvalidDepthError(
            f"Depth argument should be either -1 or >= 1. {depth} given"
        )

    rctx = require_read_context(ctx)
    validate_is_container(site, ctx)

    # Descendants: prefix match, optionally filter by exact depth
    prefix = PrefixFilter(prefix=site)
    if depth == -1:
        # All descendants at any depth
        scan_opts = StorageScanOptions(
            start=site,
            break_filter=prefix,
            filter=prefix,
        )
    else:
        # Exact depth match
        depth_len = LengthFilter(length=len(site) + depth)
        scan_opts = StorageScanOptions(
            start=site,
            break_filter=prefix,
            filter=prefix & depth_len,
        )

    yield from rctx.scan(scan_opts).keys()


def walk_descendants(
    site: site_.Site,
    ctx: StorageContextType,
) -> Generator[tuple[site_.Site, NodeType], None, None]:
    """Walk over descendant structure.

    Args:
        site: Container site
        ctx: Storage context (transaction, snapshot or write batch)

    Yields:
        Tuples of (site, NodeType) for each descendant

    Raises:
        ContainerNotFoundError: If container doesn't exist
        ContainerTypeError: If site is not a container
        StorageInterfaceError: If context doesn't support read access
    """
    rctx = require_read_context(ctx)
    validate_is_container(site, ctx)

    # All descendants including site itself
    prefix = PrefixFilter(prefix=site)
    scan_opts = StorageScanOptions(
        start=site,
        break_filter=prefix,
        filter=prefix,
    )

    for key, value in rctx.scan(scan_opts).items():
        node_type = NodeType.CONTAINER if is_marker(value) else NodeType.PRIMITIVE
        yield (key, node_type)
