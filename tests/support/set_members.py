"""Spawned workers for the cross-process set member tests."""

from __future__ import annotations

from pathlib import Path

from tests.support.counting import open_backend
from virtuals._views import SetView
from virtuals.navigator import Navigator


MEMBERS = (frozenset("abcd"), ("t", frozenset({1, 2, 3})), frozenset({frozenset({"x", "y"}), 4}))


def add_members(backend: str, path: str) -> None:
    storage = open_backend(backend, Path(path))
    try:
        with storage.transaction() as tx:
            view = Navigator(storage).root(tx).open_child("s", SetView)
            for m in MEMBERS:
                view.add(m)
    finally:
        storage.close()


def check_and_add_again(backend: str, path: str) -> dict:
    """Report membership as seen here, then add the same members once more."""
    storage = open_backend(backend, Path(path))
    try:
        with storage.transaction() as tx:
            view = Navigator(storage).root(tx).open_child("s", SetView)
            seen = {repr(m): (m in view) for m in MEMBERS}
            for m in MEMBERS:
                view.add(m)
            return {"seen": seen, "len": len(view), "members": list(view)}
    finally:
        storage.close()
