"""View bases for composing custom observable behaviors.

This module provides reusable bases for common patterns:
- ChildObservableBase
- DescendantsObservableBase
- ObservableBase
"""

from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING

from virtuals.tkv.filter import LengthFilter, PrefixFilter, WildcardFilter
from virtuals.tkv.observer import SubscriptionOptions

from .bases import AddressMappingBase


if TYPE_CHECKING:
    from virtuals.container import Container
    from virtuals.loc import site as site_


__all__ = [
    "ChildObservableBase",
    "DescendantsObservableBase",
    "ObservableBase",
]

logger = getLogger(__name__)


class ObservableBase:
    """Base providing subscription-options for the whole view.

    Views expose the filter shape; callers pass the returned
    `SubscriptionOptions` to a process-scope observer's `subscribe`.

    Example:
        >>> class MyView(ObservableBase, View): ...
        >>> options = view.on_change()
        >>> sub = observer.subscribe(options)
        >>> sub.bind(my_callback)
        >>> # ... later
        >>> sub.close()
    """

    container: Container

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
        return SubscriptionOptions(PrefixFilter(prefix=(self.container.site)))


class ChildObservableBase[A](AddressMappingBase[A]):
    """Base providing subscription-options for a view's children.

    Views expose the filter shape; callers pass the returned
    `SubscriptionOptions` to a process-scope observer's `subscribe`.

    Type Parameters:
        A: The type of address for children

    Example:
        >>> class MyView(ChildObservableBase[int], View): ...
        >>> options = view.on_child_change(0)
        >>> sub = observer.subscribe(options)
        >>> sub.bind(my_callback)
        >>> # ... later
        >>> sub.close()
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
        normalized = self.normalize_address(address)
        child_full_site = (*self.container.site, normalized)
        return SubscriptionOptions(
            filter=PrefixFilter(prefix=child_full_site)
            & LengthFilter(length=len(child_full_site))
        )

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
        child_full_site = (*self.container.site, "*")
        return SubscriptionOptions(
            WildcardFilter(pattern=child_full_site),
        )


class DescendantsObservableBase:
    """Base providing subscription-options for a view's descendants.

    Views expose the filter shape; callers pass the returned
    `SubscriptionOptions` to a process-scope observer's `subscribe`.

    Example:
        >>> class MyView(DescendantsObservableBase, View): ...
        >>> options = view.on_descendants_change("users", "*", "age")
        >>> sub = observer.subscribe(options)
        >>> sub.bind(my_callback)
        >>> # ... later
        >>> sub.close()
    """

    container: Container

    def on_descendants_change(
        self,
        address: site_.SiteSegment,
        *addresses: site_.SiteSegment,
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
        pattern = (address, *addresses)
        wildcard_site = (*self.container.site, *pattern)
        return SubscriptionOptions(WildcardFilter(pattern=wildcard_site))
