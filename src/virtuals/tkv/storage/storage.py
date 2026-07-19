"""Storage protocol definitions.

Defines the abstract interfaces for storage operations, transactions,
and iteration. Implementations must conform to these protocols.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

from .context import TransactionalStorageProtocol


if TYPE_CHECKING:
    from types import TracebackType


@runtime_checkable
class StorageProtocol(TransactionalStorageProtocol, Protocol):
    """Storage interface with transactions.

    Top-level interface for storage operations. Provides transaction
    management. Change notifications are delivered via an optional
    `PublisherProtocol` injected at construction time; subscriptions
    live on an `ObserverProtocol`, not on the storage.
    """

    # ========================================================================
    # Properties
    # ========================================================================

    @property
    def read_only(self) -> bool:
        """Storage access."""
        ...

    # ========================================================================
    # Lifecycle
    # ========================================================================

    def open(self) -> None:
        """Open storage and initialize resources.

        Raises:
            StorageOperationError: If open fails.
        """
        ...

    def close(self) -> None:
        """Close storage and release resources.

        All transactions must be completed before closing.

        Raises:
            StorageOperationError: If close fails.
        """
        ...

    def __enter__(self) -> Self:
        """Open storage via context manager."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close storage via context manager."""
        ...

    # ========================================================================
    # Transaction Management
    # ------------------------------------------------------------------------
    # Transaction management methods are inherited from TransactionalStorageProtocol.
    # begin() -> TransactionProtocol | SnapshotProtocol | WriteBatchProtocol
    # ========================================================================


__all__ = [
    "StorageProtocol",
]
