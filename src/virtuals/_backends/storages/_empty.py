"""The one sentinel check layer 1 makes: ``EMPTY`` is never stored.

Every raw put door runs ``if value is EMPTY: raise empty_error(key)``
inline, before any write. The check is one identity test, so it costs
nothing per put; only the error path calls into this module. A stored
sentinel would read back as a second ``Empty`` that is not ``EMPTY``,
count as existing and show up in scans (contract section 6).
"""

from __future__ import annotations


__all__ = ["empty_error"]


def empty_error(key: object) -> ValueError:
    """The error a raw put raises for the absent sentinel."""
    return ValueError(f"Cannot store EMPTY at {key}: EMPTY means absent, delete the key instead")
