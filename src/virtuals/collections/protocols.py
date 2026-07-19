"""Atomic capability protocols.

These protocols define the smallest composable units of collection behavior.
Collection types (Mapping, Sequence, Set) compose these into complete interfaces.

Check protocol support at runtime with the is_* type guard functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeGuard, runtime_checkable


if TYPE_CHECKING:
    import random

    from virtuals.loc import key
    from virtuals.tkv.observer import SubscriptionOptions
    from virtuals.types import Empty
    from virtuals.view import View


__all__ = [
    "Assignable",
    "ChildObservable",
    "Clearable",
    "Containable",
    "Convertible",
    "Deletable",
    "DescendantsObservable",
    "Initializable",
    "Nestable",
    "Observable",
    "Sampleable",
    "Sizeable",
    "Subscriptable",
    "is_assignable",
    "is_child_observable",
    "is_clearable",
    "is_containable",
    "is_convertible",
    "is_deletable",
    "is_descendants_observable",
    "is_initializable",
    "is_nestable",
    "is_observable",
    "is_sampleable",
    "is_sizeable",
    "is_subscriptable",
]


# =============================================================================
# CORE CAPABILITY PROTOCOLS
# =============================================================================


# TODO: should inherit from View protocol? class Convertible[V](Protocol):


@runtime_checkable
class Convertible[V](Protocol):
    """Protocol for containers that can convert their contents to Python values.

    Convertible containers can materialize their entire stored state into
    native Python data structures (dict, list, set, etc.).

    Type Parameters:
        T: The type of value this container extracts to

    Example:
        >>> if isinstance(container, Convertible):
        ...     data = container.extract()
        ...     # data is now a native Python dict/list/etc
    """

    def extract(self) -> V | Empty:
        """Extract container contents as native Python value.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Native Python value representing container contents
        """
        ...


@runtime_checkable
class Initializable[V](Protocol):
    """Protocol for containers that can be initialized from Python values.

    Initializable containers can populate their contents from native Python
    data structures, handling the conversion and storage automatically.

    Type Parameters:
        T: The type of value this container accepts for initialization

    Example:
        >>> if isinstance(container, Initializable):
        ...     container.store({"key": "value"}, replace=True)
    """

    def store(self, value: V) -> None:
        """Store Python value into container.

        Args:
            value: Native Python value to store
            *args: Positional arguments
            replace: If True, clear existing content before storing
            **kwargs: Keyword arguments

        Raises:
            TypeError: If value type not supported
            ValueError: If value format invalid
        """
        ...


@runtime_checkable
class Nestable[A](Protocol):
    """Protocol for containers that support navigation to child containers.

    Nestable containers can navigate their hierarchy, returning appropriate
    container instances for child nodes.

    Example:
        >>> if isinstance(container, Nestable):
        ...     child = container.open_child("users", DictView)
        ...     # child is another container at the given location
    """

    def open_child[ViewT: View](self, address: A, view: type[ViewT]) -> ViewT:
        """Navigate to child container.

        Args:
            address: Child container address
            view: View to open child container with
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Container instance for child

        Raises:
            KeyError: If child doesn't exist
            TypeError: If child is not a container
        """
        ...


@runtime_checkable
class Subscriptable[A, V](Protocol):
    """Protocol for containers that support item access via subscript notation.

    Subscriptable containers implement __getitem__ to retrieve values by
    address using bracket notation.

    Example:
        >>> if isinstance(container, Subscriptable):
        ...     value = container["key"]
        ...     # or: value = container[0]
    """

    def __getitem__(self, address: A) -> V | Empty:
        """Get item by address.

        Args:
            address: Item address (index or other identifier)

        Returns:
            Value at the given address

        Raises:
            KeyError: If address doesn't exist
            IndexError: If index out of range
        """
        ...


@runtime_checkable
class Assignable[A, V](Protocol):
    """Protocol for containers that support item assignment via subscript notation.

    Assignable containers implement __setitem__ to store values by
    address using bracket notation.

    Example:
        >>> if isinstance(container, Assignable):
        ...     container["key"] = value
        ...     # or: container[0] = value
    """

    def __setitem__(self, address: A, value: V) -> None:
        """Set item at address.

        Args:
            address: Item address (index or other identifier)
            value: Value to store

        Raises:
            TypeError: If value type not supported
            IndexError: If index out of range
        """
        ...


@runtime_checkable
class Containable[V](Protocol):
    """Protocol for containers that support membership testing.

    Containable containers implement __contains__ to check if an address
    exists using the 'in' operator.

    Type Parameters:
        A: The type of address to check for membership

    Example:
        >>> if isinstance(container, Containable):
        ...     if "key" in container:
        ...         print("Key exists")
    """

    def __contains__(self, obj: V) -> bool:
        """Check if address exists in container.

        Args:
            obj: Object to check for existence (existence dimension is based on the view semantics - value, address, etc)

        Returns:
            True if address exists in container
        """
        ...


@runtime_checkable
class Sizeable(Protocol):
    """Protocol for containers that support size queries.

    Sizeable containers implement __len__ to return the number of items
    using the len() function.

    Example:
        >>> if isinstance(container, Sizeable):
        ...     size = len(container)
    """

    def __len__(self) -> int:
        """Get number of items in container.

        Returns:
            Number of items
        """
        ...


@runtime_checkable
class Sampleable[V](Protocol):
    """Protocol for containers that support kh57-style range reservoir sampling.

    Sampleable containers implement sample() to return up to ``n`` uniform
    (int_key, value) pairs from an optional integer key sub-range.

    Type Parameters:
        V: The type of value paired with each sampled key.

    Example:
        >>> if isinstance(container, Sampleable):
        ...     picks = container.sample(500, begin=0, end=1_000_000)
    """

    def sample(
        self,
        n: int,
        begin: int | None = None,
        end: int | None = None,
        *,
        rng: random.Random | None = None,
    ) -> list[tuple[int, V]]:
        """Return up to `n` uniform samples from [begin, end).

        Args:
            n: Number of items to sample. Fewer are returned if the range
                holds fewer than `n` items.
            begin: Inclusive lower bound on the original int key. None
                means unbounded from below.
            end: Exclusive upper bound on the original int key. None means
                unbounded from above.
            rng: Optional seeded random.Random for deterministic sampling.

        Returns:
            List of (int_key, value) pairs. Order is unspecified.
        """
        ...


@runtime_checkable
class Deletable[A](Protocol):
    """Protocol for containers that support item deletion.

    Deletable containers implement __delitem__ to remove items by
    address using the del statement.

    Type Parameters:
        A: The type of address to delete

    Example:
        >>> if isinstance(container, Deletable):
        ...     del container["key"]
        ...     # or: del container[0]
    """

    def __delitem__(self, address: A) -> None:
        """Delete item at address.

        Args:
            address: Item address to delete

        Raises:
            KeyError: If address doesn't exist
            IndexError: If index out of range
        """
        ...


@runtime_checkable
class Clearable(Protocol):
    """Protocol for containers that support clearing all items.

    Clearable containers implement clear() to remove all items at once.

    Example:
        >>> if isinstance(container, Clearable):
        ...     container.clear()
    """

    def clear(self) -> None:
        """Remove all items from container."""
        ...


@runtime_checkable
class Observable(Protocol):
    """Protocol for containers that support observing changes.

    Observable containers implement on_change() to produce
    SubscriptionOptions for all changes within this view. Callers hand
    those options to a process-scope observer's subscribe().

    Example:
        >>> if isinstance(container, Observable):
        ...     options = container.on_change()
        ...     sub = observer.subscribe(options)
        ...     sub.bind(my_callback)
        ...     # ... later
        ...     sub.close()
    """

    def on_change(self) -> SubscriptionOptions:
        """Build subscription options for all changes in this view.

        Returns:
            SubscriptionOptions describing the filter shape.

        Example:
            >>> options = view.on_change()
            >>> sub = observer.subscribe(options)
            >>> sub.bind(callback)
            >>> sub.close()
        """
        ...


@runtime_checkable
class ChildObservable[A](Protocol):
    """Protocol for containers that support observing child changes.

    ChildObservable containers implement on_child_change() and
    on_children_change() to produce SubscriptionOptions for child
    changes. Callers hand those options to observer.subscribe().

    Type Parameters:
        A: The type of address for children

    Example:
        >>> if isinstance(container, ChildObservable):
        ...     options = container.on_child_change("alice")
        ...     sub = observer.subscribe(options)
        ...     sub.bind(my_callback)
        ...     # ... later
        ...     sub.close()
    """

    def on_child_change(self, address: A) -> SubscriptionOptions:
        """Build subscription options for a specific child and its subtree.

        Args:
            address: Child address to watch

        Returns:
            SubscriptionOptions describing the filter shape.

        Example:
            >>> options = view.on_child_change("users")
            >>> sub = observer.subscribe(options)
            >>> sub.bind(callback)
            >>> sub.close()
        """
        ...

    def on_children_change(self) -> SubscriptionOptions:
        """Build subscription options for all children.

        Returns:
            SubscriptionOptions describing the filter shape.

        Example:
            >>> options = view.on_children_change()
            >>> sub = observer.subscribe(options)
            >>> sub.bind(callback)
            >>> sub.close()
        """
        ...


@runtime_checkable
class DescendantsObservable(Protocol):
    """Protocol for containers that support observing descendant changes.

    DescendantsObservable containers implement on_descendants_change() to
    produce SubscriptionOptions for descendants matching a pattern.

    Example:
        >>> if isinstance(container, DescendantsObservable):
        ...     options = container.on_descendants_change("users", "*", "age")
        ...     sub = observer.subscribe(options)
        ...     sub.bind(my_callback)
        ...     # ... later
        ...     sub.close()
    """

    def on_descendants_change(
        self,
        address: key.KeySegment,
        *addresses: key.KeySegment,
    ) -> SubscriptionOptions:
        """Build subscription options for descendants matching a pattern.

        Args:
            address: First address segment in the pattern
            *addresses: Additional address segments (use "*" for wildcards)

        Returns:
            SubscriptionOptions describing the filter shape.

        Example:
            >>> options = view.on_descendants_change("users", "*", "age")
            >>> sub = observer.subscribe(options)
            >>> sub.bind(callback)
            >>> sub.close()
        """
        ...


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def is_convertible(obj: object) -> TypeGuard[Convertible]:
    """Check if object supports extract operation.

    Args:
        obj: Object to check

    Returns:
        True if object implements Convertible protocol
    """
    return isinstance(obj, Convertible)


def is_initializable(obj: object) -> TypeGuard[Initializable]:
    """Check if object supports store operation.

    Args:
        obj: Object to check

    Returns:
        True if object implements Initializable protocol
    """
    return isinstance(obj, Initializable)


def is_nestable(obj: object) -> TypeGuard[Nestable]:
    """Check if object supports child navigation.

    Args:
        obj: Object to check

    Returns:
        True if object implements Nestable protocol
    """
    return isinstance(obj, Nestable)


def is_subscriptable(obj: object) -> TypeGuard[Subscriptable]:
    """Check if object supports item access via subscript.

    Args:
        obj: Object to check

    Returns:
        True if object implements Subscriptable protocol
    """
    return isinstance(obj, Subscriptable)


def is_assignable(obj: object) -> TypeGuard[Assignable]:
    """Check if object supports item assignment via subscript.

    Args:
        obj: Object to check

    Returns:
        True if object implements Assignable protocol
    """
    return isinstance(obj, Assignable)


def is_containable(obj: object) -> TypeGuard[Containable]:
    """Check if object supports membership testing.

    Args:
        obj: Object to check

    Returns:
        True if object implements Containable protocol
    """
    return isinstance(obj, Containable)


def is_sizeable(obj: object) -> TypeGuard[Sizeable]:
    """Check if object supports size queries.

    Args:
        obj: Object to check

    Returns:
        True if object implements Sizeable protocol
    """
    return isinstance(obj, Sizeable)


def is_sampleable(obj: object) -> TypeGuard[Sampleable]:
    """Check if object supports kh57-style range sampling.

    Args:
        obj: Object to check

    Returns:
        True if object implements Sampleable protocol
    """
    return isinstance(obj, Sampleable)


def is_deletable(obj: object) -> TypeGuard[Deletable]:
    """Check if object supports item deletion.

    Args:
        obj: Object to check

    Returns:
        True if object implements Deletable protocol
    """
    return isinstance(obj, Deletable)


def is_clearable(obj: object) -> TypeGuard[Clearable]:
    """Check if object supports clearing all items.

    Args:
        obj: Object to check

    Returns:
        True if object implements Clearable protocol
    """
    return isinstance(obj, Clearable)


def is_observable(obj: object) -> TypeGuard[Observable]:
    """Check if object supports observing changes.

    Args:
        obj: Object to check

    Returns:
        True if object implements Observable protocol
    """
    return isinstance(obj, Observable)


def is_child_observable(obj: object) -> TypeGuard[ChildObservable]:
    """Check if object supports observing child changes.

    Args:
        obj: Object to check

    Returns:
        True if object implements ChildObservable protocol
    """
    return isinstance(obj, ChildObservable)


def is_descendants_observable(obj: object) -> TypeGuard[DescendantsObservable]:
    """Check if object supports observing descendant changes.

    Args:
        obj: Object to check

    Returns:
        True if object implements DescendantsObservable protocol
    """
    return isinstance(obj, DescendantsObservable)
