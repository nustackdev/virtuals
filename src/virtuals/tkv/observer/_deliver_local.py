"""Internal helper: local-callback delivery for observers.

Used by observer backends after matching a batch of keys against the
subscription registry, to fan out matched notifications to bound
callbacks. Not part of the public tkv surface.
"""

from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ..types import Key
    from .subscription import Subscription


logger = getLogger(__name__)


__all__ = [
    "deliver_local",
]


def deliver_local(
    notifications: list[tuple[Key, list[Subscription]]],
) -> None:
    """Deliver notifications to local subscribers.

    Args:
        notifications: Matched (key, subscriptions) pairs.
    """
    for key, subs in notifications:
        for sub in subs:
            for error in sub.notify(key):
                logger.error("Callback failed for %s: %s", key, error)
