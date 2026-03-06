"""Base context classes and operation mixins for in-memory storage."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tkv.tkv.storage import (
    ScanProtocol,
    StorageClosedError,
    StorageOperationError,
    StorageScanOptions,
)
from tkv.tkv.types import EMPTY, Empty

from .state import TransactionState


if TYPE_CHECKING:
    from tkv.tkv.types import Key, Value

    from .storage import InMemoryStorage


__all__ = [
    "ContextBase",
    "ReadOperationsMixin",
    "WriteOperationsMixin",
]


class ContextBase:
    """Base class for in-memory storage contexts.

    Provides common functionality for state management, validation, and
    resource cleanup.
    """

    __slots__ = ("_closed", "_state", "_storage")

    def __init__(self, storage: InMemoryStorage, state: dict[str, Any] | TransactionState) -> None:
        """Initialize context with storage reference and state.

        Args:
            storage: Parent storage instance
            state: State dictionary or overlay
        """
        self._storage = storage
        self._state = state
        self._closed = False

    @property
    def storage(self) -> InMemoryStorage:
        """Get the storage instance.

        Returns:
            Storage this context was initiated from.
        """
        return self._storage

    def _require_active(self) -> dict[str, Any] | TransactionState:
        """Validate context is active and return state.

        Returns:
            Active state

        Raises:
            StorageClosedError: If context is closed
        """
        if self._closed:
            raise StorageClosedError("Context is closed")
        return self._state

    def _mark_closed(self) -> None:
        """Mark context as closed."""
        self._closed = True

    @property
    def is_closed(self) -> bool:
        """Check if context is closed.

        Returns:
            True if closed, False otherwise.
        """
        return self._closed

    @property
    def is_active(self) -> bool:
        """Check if context is active.

        Returns:
            True if active and not closed, False otherwise.
        """
        return not self._closed


class ReadOperationsMixin:
    """Mixin providing read operations for in-memory storage contexts."""

    __slots__ = ()

    # Type hints for mixed-in attributes
    _storage: InMemoryStorage
    _require_active: Any

    def get(self, key: Key) -> Value | Empty:
        """Get value by key.

        Args:
            key: Key to retrieve

        Returns:
            Value at key, or EMPTY if key not found.
            Never raises on missing keys.

        Raises:
            StorageOperationError: If operation fails
            StorageClosedError: If context is closed
        """
        state = self._require_active()
        codec = self._storage.codec

        # Encode key using codec
        try:
            key_str = codec.encode_key(key)
        except Exception as e:
            raise StorageOperationError(f"Failed to encode key {key}: {e}") from e

        # Check if key exists - return EMPTY instead of raising
        try:
            if isinstance(state, TransactionState):
                value_encoded = state.get(key_str)
            else:
                if key_str not in state:
                    return EMPTY
                value_encoded = state[key_str]
        except KeyError:
            return EMPTY

        # Decode and return value
        try:
            return codec.decode_value(value_encoded)
        except Exception as e:
            raise StorageOperationError(f"Failed to decode value for key {key}: {e}") from e

    def exists(self, key: Key) -> bool:
        """Check if key exists.

        Args:
            key: Key to check

        Returns:
            True if key exists, False otherwise

        Raises:
            StorageOperationError: If check fails
            StorageClosedError: If context is closed
        """
        state = self._require_active()
        codec = self._storage.codec

        try:
            key_str = codec.encode_key(key)
        except Exception as e:
            raise StorageOperationError(f"Failed to encode key {key}: {e}") from e

        return key_str in state

    def multiget(self, keys: list[Key]) -> dict[Key, Value]:
        """Get multiple keys.

        Args:
            keys: List of keys to retrieve

        Returns:
            Dict mapping keys to values. Missing keys are omitted.

        Raises:
            StorageOperationError: If operation fails
            StorageClosedError: If context is closed
        """
        result: dict[Key, Value] = {}

        for key in keys:
            value = self.get(key)
            if value is not EMPTY:
                result[key] = value

        return result

    def scan(self, options: StorageScanOptions) -> ScanProtocol:
        """Create scan iterator with configured options.

        Args:
            options: Scan configuration (start, reverse, limit, filter, break_filter)

        Returns:
            Scan iterator conforming to ScanProtocol

        Raises:
            StorageOperationError: If scan creation fails
            StorageClosedError: If context is closed
        """
        from .scan import InMemoryScan

        # Validate context is active before creating scan
        self._require_active()
        return InMemoryScan(self, options)  # type: ignore[arg-type]


class WriteOperationsMixin:
    """Mixin providing write operations for in-memory storage contexts."""

    __slots__ = ()

    # Type hints for mixed-in attributes
    _storage: InMemoryStorage
    _require_active: Any
    _modified_keys: set[Key]

    def put(self, key: Key, value: Value) -> None:
        """Put key-value pair.

        Args:
            key: Key to set
            value: Value to store

        Raises:
            StorageOperationError: If write fails
            StorageClosedError: If context is closed
        """
        state = self._require_active()
        codec = self._storage.codec

        # Encode key using codec
        try:
            key_str = codec.encode_key(key)
        except Exception as e:
            raise StorageOperationError(f"Failed to encode key {key}: {e}") from e

        # Encode and store value
        try:
            state[key_str] = codec.encode_value(value)
        except Exception as e:
            raise StorageOperationError(f"Failed to encode value for key {key}: {e}") from e

        # Track modification
        self._modified_keys.add(key)

    def delete(self, key: Key) -> None:
        """Delete key (idempotent).

        Silent if key doesn't exist (never raises for missing keys).

        Args:
            key: Key to delete

        Raises:
            StorageOperationError: If deletion fails
            StorageClosedError: If context is closed
        """
        state = self._require_active()
        codec = self._storage.codec

        try:
            key_str = codec.encode_key(key)
        except Exception as e:
            raise StorageOperationError(f"Failed to encode key {key}: {e}") from e

        # Delete if exists, otherwise silently do nothing
        if key_str in state:
            del state[key_str]
            # Track modification only if key existed
            self._modified_keys.add(key)
