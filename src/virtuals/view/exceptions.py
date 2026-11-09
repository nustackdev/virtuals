"""View layer exception hierarchy.

This module defines all view-specific exceptions.
"""

from __future__ import annotations


__all__ = [
    "AddressNotFoundError",
    "IndexNotFoundError",
    "KeyNotFoundError",
    "ViewError",
    "ViewOperationError",
    "ViewRegistryError",
]


class ViewError(Exception):
    """Base exception for view-related errors."""


class ViewRegistryError(ViewError):
    """Raised when registry operations fail."""


class ViewOperationError(ViewError):
    """Raised when view operations fail."""


class AddressNotFoundError(ViewError, LookupError):
    """Nothing is stored at the address a read or a delete names.

    The view's own answer for a missing key or index, kept apart from the
    other ``KeyError``s a read can let out (a storage fault, a container
    above the view that is gone), so a caller can treat exactly this as
    absence.
    """


class KeyNotFoundError(AddressNotFoundError, KeyError):
    """A mapping view holds nothing under the key, like ``d[k]`` on a dict."""


class IndexNotFoundError(AddressNotFoundError, IndexError):
    """A sequence view holds nothing at the index, like ``l[i]`` on a list."""
