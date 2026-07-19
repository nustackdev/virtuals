"""Publisher-specific exceptions."""

from __future__ import annotations


__all__ = [
    "PublisherConnectionError",
    "PublisherError",
    "PublisherOperationError",
]


class PublisherError(Exception):
    """Base exception for publisher errors."""

    pass


class PublisherConnectionError(PublisherError):
    """Raised when publisher connection fails."""

    pass


class PublisherOperationError(PublisherError):
    """Raised when a publisher operation fails."""

    pass
