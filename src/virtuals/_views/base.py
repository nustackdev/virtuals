"""Base class for standard views."""

from __future__ import annotations

from typing import TYPE_CHECKING

from virtuals.view import View, ViewBase


__all__ = [
    "StdView",
]


class StdView(ViewBase):
    """Base class for standard views.

    Type Parameters:
        AddressT: Type of addresses/keys this view accepts (default: str | int)
        ValueT: Type of values this view stores/returns (default: Value)
    """

    @classmethod
    def get_available_views(cls) -> tuple[type[View], ...]:
        """Returns tuple of views defined in collections module."""
        from .bytearray_view import ByteArrayView
        from .dict_view import EagerDictView
        from .frozenset_view import FrozenSetView
        from .list_view import EagerListView
        from .set_view import SetView
        from .tuple_view import TupleView

        return (ByteArrayView, EagerDictView, FrozenSetView, EagerListView, SetView, TupleView)

    @classmethod
    def get_default_parent_view(cls) -> type[View]:
        """Returns EagerDictView as a default view."""
        from .dict_view import EagerDictView

        return EagerDictView


if TYPE_CHECKING:
    _: type[View] = StdView
