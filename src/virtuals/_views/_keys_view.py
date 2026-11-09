"""Keys view of an ordered mapping view: set-like, and reversible like ``dict.keys()``."""

from __future__ import annotations

from collections.abc import Iterator, KeysView


class ReversibleKeysView(KeysView):
    """``KeysView`` whose ``reversed()`` reads the mapping's own ``__reversed__``: one reverse scan."""

    def __reversed__(self) -> Iterator:
        return reversed(self._mapping)  # type: ignore[call-overload]
