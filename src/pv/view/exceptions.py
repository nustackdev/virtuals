"""View layer exception hierarchy.

This module defines all view-specific exceptions.
"""

from __future__ import annotations

from pv._exception import PVError


__all__ = [
    "ViewError",
    "ViewOperationError",
    "ViewRegistryError",
]


class ViewError(PVError):
    """Base exception for view-related errors."""


class ViewRegistryError(ViewError):
    """Raised when registry operations fail."""


class ViewOperationError(ViewError):
    """Raised when view operations fail."""
