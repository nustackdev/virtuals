"""Base context and the read and write mixins shared by in-memory contexts.

A context decides what a read sees through two hooks the mixins call:
``_lookup`` for a point read and ``_page`` for one page of a scan. A
snapshot answers them from the store as of its pinned sequence; a
transaction answers them from live state overlaid with its own pending
writes. The mixins own the codec work and the protocol's error mapping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from virtuals._backends.storages._empty import empty_error
from virtuals.tkv.storage import (
    ScanProtocol,
    StorageClosedError,
    StorageOperationError,
    StorageScanOptions,
)
from virtuals.tkv.types import EMPTY, Empty

from .store import ABSENT, DELETED, Encoded, Entry, Marker


if TYPE_CHECKING:
    from virtuals.tkv.types import Key, Value

    from .storage import InMemoryStorage


__all__ = [
    "ContextBase",
    "ReadOperationsMixin",
    "WriteOperationsMixin",
]


class ContextBase:
    """What every in-memory context has: its storage and a closed flag."""

    __slots__ = ("_closed", "_storage")

    def __init__(self, storage: InMemoryStorage) -> None:
        """Bind the context to its storage."""
        self._storage = storage
        self._closed = False

    @property
    def storage(self) -> InMemoryStorage:
        """The storage this context was begun on."""
        return self._storage

    def _require_active(self) -> None:
        """Raise ``StorageClosedError`` if the context is closed."""
        if self._closed:
            raise StorageClosedError("Context is closed")

    def _claim_end(self) -> bool:
        """Flip to closed exactly once. True for the caller that gets to end the context.

        Commit, abort and ``storage.close`` may race on one context from
        different threads; whoever claims first finishes it, the others
        see it closed.
        """
        with self._storage._state_lock:
            if self._closed:
                return False
            self._closed = True
            return True

    @property
    def is_closed(self) -> bool:
        """True once committed, aborted or closed."""
        return self._closed

    @property
    def is_active(self) -> bool:
        """True until committed, aborted or closed."""
        return not self._closed


class ReadOperationsMixin:
    """Point reads and scans over a context's view of the store."""

    __slots__ = ()

    _storage: InMemoryStorage
    _require_active: Any

    def _lookup(self, enc: Encoded) -> Entry | Marker:
        """Entry visible to this context for an encoded key, or ABSENT."""
        raise NotImplementedError

    def _page(
        self, bound: Encoded | None, *, inclusive: bool, reverse: bool, n: int, stop: Encoded | None
    ) -> tuple[list[tuple[Any, Entry]], Any, bool]:
        """One page of rows visible to this context. Returns ``(rows, cover, done)``."""
        raise NotImplementedError

    def get(self, key: Key) -> Value | Empty:
        """Value at ``key``, or EMPTY when absent. Never raises on a missing key."""
        self._require_active()
        codec = self._storage.codec
        try:
            enc = codec.encode_key(key)
        except Exception as e:
            raise StorageOperationError(f"Failed to encode key {key}: {e}") from e
        entry = self._lookup(enc)
        if entry is ABSENT:
            return EMPTY
        try:
            return codec.decode_value(entry[1])
        except Exception as e:
            raise StorageOperationError(f"Failed to decode value for key {key}: {e}") from e

    def exists(self, key: Key) -> bool:
        """True when ``key`` is visible to this context."""
        self._require_active()
        try:
            enc = self._storage.codec.encode_key(key)
        except Exception as e:
            raise StorageOperationError(f"Failed to encode key {key}: {e}") from e
        return self._lookup(enc) is not ABSENT

    def multiget(self, keys: list[Key]) -> dict[Key, Value]:
        """Values for ``keys``; missing keys are omitted."""
        result: dict[Key, Value] = {}
        for key in keys:
            value = self.get(key)
            if value is not EMPTY:
                result[key] = value
        return result

    def scan(self, options: StorageScanOptions) -> ScanProtocol:
        """A lazy scan over this context's view with ``options`` applied."""
        from .scan import InMemoryScan

        self._require_active()
        return InMemoryScan(self, options)  # type: ignore[arg-type]


class WriteOperationsMixin:
    """Buffered writes: a write set applied as a whole on commit or write.

    ``_pending`` maps encoded key to ``(key, value)`` or ``(key, DELETED)``.
    """

    __slots__ = ()

    _storage: InMemoryStorage
    _require_active: Any
    _pending: dict[Encoded, Entry]
    _modified_keys: set[Key]

    def put(self, key: Key, value: Value) -> None:
        """Set ``key`` to ``value`` in the write set."""
        self._require_active()
        if value is EMPTY:
            raise empty_error(key)
        codec = self._storage.codec
        try:
            enc = codec.encode_key(key)
            encoded = codec.encode_value(value)
        except Exception as e:
            raise StorageOperationError(f"Failed to encode key/value for {key}: {e}") from e
        self._stage(enc, key, encoded)
        self._modified_keys.add(key)

    def delete(self, key: Key) -> None:
        """Delete ``key``; silently nothing when absent."""
        self._require_active()
        try:
            enc = self._storage.codec.encode_key(key)
        except Exception as e:
            raise StorageOperationError(f"Failed to encode key {key}: {e}") from e
        self._stage(enc, key, DELETED)

    def _stage(self, enc: Encoded, key: Key, value: object) -> None:
        """Record one write in the write set."""
        self._pending[enc] = (key, value)
