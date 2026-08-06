"""Text-based storage backend for debugging and learning.

WARNING: TOY IMPLEMENTATION - NOT FOR PRODUCTION USE

This storage backend prioritizes human readability and simplicity over performance.
Perfect for tutorials, examples, and understanding how storage layers work.

Purpose:
  - Learning and onboarding (understand storage concepts)
  - Debugging (inspect state.json with cat/jq/text editor)
  - Toy projects and experimentation
  - Example code and documentation

Features:
  - Human-readable JSON format
  - Simple file-based persistence
  - Optional operation logging
  - Implements StorageProtocol correctly

Limitations:
  - Writes serialized (one transaction at a time)
  - Last writer wins (no conflict detection or optimistic locking)
  - Memory-bound (entire state kept in RAM)
  - Slow writes (full state written to disk on every commit)
  - Single process only (no file locking or coordination)
  - Not suitable for datasets >1000 keys

Use RocksDB adapter for real workloads.
"""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, overload

from virtuals.tkv.storage import (
    SnapshotProtocol,
    StorageClosedError,
    StorageError,
    StorageOperationError,
    TransactionProtocol,
    WriteBatchProtocol,
)

from .snapshot import TextSnapshot
from .transaction import TextTransaction
from .write_batch import TextWriteBatch


if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import TracebackType

    from virtuals.tkv.codec import CodecProtocol
    from virtuals.tkv.publisher import PublisherProtocol
    from virtuals.tkv.types import Key, Value


__all__ = ["TextStorage"]


logger = getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

STATE_FILE = "state.json"
OPERATIONS_FILE = "operations.jsonl"
STATE_VERSION = 1


class TextStorage:
    """Text-based storage for debugging and learning.

    WARNING: Toy implementation - prioritizes simplicity and readability over performance.

    File structure:
        storage_dir/
        ├── state.json          # Current key-value state (human-readable)
        └── operations.jsonl    # Operation log (optional, for tracing)

    Thread Safety:
        - Snapshots: Safe to create and use concurrently
        - Transactions: Must not share transaction objects between threads
        - Writes: Automatically serialized (only one commit at a time)
        - Lost updates: Possible - last writer wins, no conflict detection

    Limitations:
        - ONE write at a time (commits fully serialized via _write_lock)
        - NO conflict detection (concurrent transactions on different keys → last wins)
        - Entire state in memory (bounded by RAM, max ~1000 keys recommended)
        - Full state written to disk on every commit (slow, not for high-throughput)
        - Single process only (no file locking or multi-process coordination)

    Attributes:
        path: Storage directory path
        codec: Codec for key/value encoding
    """

    def __init__(
        self,
        path: str | Path,
        codec: CodecProtocol,
        publisher: PublisherProtocol | None = None,
        log_operations: bool = False,
        read_only: bool = False,
    ) -> None:
        """Initialize text storage.

        Args:
            path: Directory path for storage files
            codec: Codec for key/value encoding
            publisher: Publisher instance for managing update notifications
            log_operations: Enable operation logging (default: False)
            read_only: Mode
        """
        self.path = Path(path)
        self.codec = codec
        self._publisher = publisher
        self._log_operations = log_operations
        self._read_only = read_only

        # State
        self._state: dict[str, Any] = {}  # key_str -> value
        self._opened = False

        # Synchronization
        # - _lock: protects in-memory state and context tracking (reentrant for close)
        # - _write_lock: enforces single-writer (transaction/batch) semantics
        self._lock = threading.RLock()
        self._write_lock = threading.Lock()

        # Context tracking
        self._active_transactions: set[TextTransaction] = set()
        self._active_snapshots: set[TextSnapshot] = set()
        self._active_write_batches: set[TextWriteBatch] = set()

    @property
    def read_only(self) -> bool:
        """Storage access."""
        return self._read_only

    def _require_open(self) -> None:
        """Validate storage is open.

        Raises:
            StorageClosedError: If storage is not open
        """
        if not self._opened:
            raise StorageClosedError("Storage is not open")

    def _read_state(self) -> dict[str, Any]:
        """Read state from disk.

        Returns:
            State dictionary

        Raises:
            StorageError: If read fails
        """
        state_path = self.path / STATE_FILE

        if not state_path.exists():
            return {}

        try:
            with state_path.open() as f:
                data = json.load(f)

            # Validate version
            if data.get("version") != STATE_VERSION:
                raise StorageError(
                    f"Unsupported state version: {data.get('version')} (expected {STATE_VERSION})"
                )

            return data.get("data", {})
        except json.JSONDecodeError as e:
            raise StorageError(f"Failed to parse state file: {e}") from e
        except Exception as e:
            raise StorageError(f"Failed to read state file: {e}") from e

    def _write_state(self, state: dict[str, Any]) -> None:
        """Write state to disk atomically.

        Args:
            state: State dictionary to write

        Raises:
            StorageError: If write fails
        """
        # Lock entire operation to ensure disk and memory stay consistent
        with self._lock:
            state_path = self.path / STATE_FILE

            # Create directory if needed
            self.path.mkdir(parents=True, exist_ok=True)

            # Prepare data
            data = {"version": STATE_VERSION, "data": state}

            # Write to temp file
            temp_path = state_path.with_suffix(".tmp")
            try:
                with temp_path.open("w") as f:
                    json.dump(data, f, indent=2, sort_keys=True)
                    f.flush()
            except Exception as e:
                # Clean up temp file
                try:
                    temp_path.unlink(missing_ok=True)
                except Exception as e_unlink:
                    logger.error(
                        "Failed to clean up temp state file",
                        extra={"error": str(e_unlink)},
                        exc_info=True,
                    )
                raise StorageError(f"Failed to write state file: {e}") from e

            # Atomic rename
            try:
                temp_path.replace(state_path)
            except Exception as e:
                # Clean up temp file
                try:
                    temp_path.unlink(missing_ok=True)
                except Exception as e_unlink:
                    logger.error(
                        "Failed to clean up temp state file",
                        extra={"error": str(e_unlink)},
                        exc_info=True,
                    )
                raise StorageError(f"Failed to replace state file: {e}") from e

            # Update in-memory state
            self._state = state.copy()

    def _log_operation(self, op: str, key: Key | None, value: Value | None) -> None:
        """Log operation to operations.jsonl.

        Args:
            op: Operation name (put, delete, commit, abort, write)
            key: Key involved (if applicable)
            value: Value involved (if applicable)
        """
        if not self._log_operations:
            return

        ops_path = self.path / OPERATIONS_FILE

        # Prepare log entry
        entry: dict[str, Any] = {
            "op": op,
            "ts": datetime.now(timezone.utc).isoformat(),
        }

        if key is not None:
            entry["key"] = key
        if value is not None:
            entry["value"] = value

        # Append to log file
        try:
            with ops_path.open("a") as f:
                json.dump(entry, f, separators=(",", ":"))
                f.write("\n")
        except Exception as e:
            logger.error("Failed to log operation", extra={"error": str(e)}, exc_info=True)

    def _untrack_transaction(self, txn: TextTransaction) -> None:
        """Remove transaction from active set.

        Args:
            txn: Transaction to untrack
        """
        with self._lock:
            self._active_transactions.discard(txn)
            # Release write lock when no writers remain so other writers can proceed
            if not self._active_transactions and not self._active_write_batches:
                try:
                    self._write_lock.release()
                except RuntimeError:
                    # Lock may already be released or not held; ignore
                    pass

    def _untrack_snapshot(self, snap: TextSnapshot) -> None:
        """Remove snapshot from active set.

        Args:
            snap: Snapshot to untrack
        """
        with self._lock:
            self._active_snapshots.discard(snap)

    def _untrack_write_batch(self, batch: TextWriteBatch) -> None:
        """Remove write batch from active set.

        Args:
            batch: Write batch to untrack
        """
        with self._lock:
            self._active_write_batches.discard(batch)
            # Release write lock when no writers remain so other writers can proceed
            if not self._active_transactions and not self._active_write_batches:
                try:
                    self._write_lock.release()
                except RuntimeError:
                    # Lock may already be released or not held; ignore
                    pass

    def _notify_batch(self, keys: set[Key]) -> None:
        """Notify publisher of key changes (batch).

        Fire-and-forget: writer enqueues and returns. Callers that need a
        delivery barrier call publisher.flush() explicitly.

        Args:
            keys: Keys that changed
        """
        if self._publisher is not None and keys:
            try:
                self._publisher.notify(keys)
            except Exception:
                logger.error("Publisher notification failed")

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def open(self) -> None:
        """Open storage and load state.

        Raises:
            StorageOperationError: If open fails
        """
        if self._opened:
            return

        try:
            # Read state from disk
            self._state = self._read_state()
            self._opened = True
        except Exception as e:
            raise StorageOperationError(f"Failed to open storage: {e}") from e

    def close(self) -> None:
        """Close storage and release resources.

        Raises:
            StorageOperationError: If close fails
        """
        if not self._opened:
            return

        with self._lock:
            # Close all active transactions
            for txn in list(self._active_transactions):
                try:
                    txn.abort()
                except Exception as e:
                    logger.error(
                        "Failed to abort transaction", extra={"error": str(e)}, exc_info=True
                    )

            # Close all active snapshots
            for snap in list(self._active_snapshots):
                try:
                    snap.close()
                except Exception as e:
                    logger.error("Failed to close snapshot", extra={"error": str(e)}, exc_info=True)

            # Close all active write batches
            for batch in list(self._active_write_batches):
                try:
                    batch.abort()
                except Exception as e:
                    logger.error(
                        "Failed to abort write batch", extra={"error": str(e)}, exc_info=True
                    )

            # Clear tracking sets
            self._active_transactions.clear()
            self._active_snapshots.clear()
            self._active_write_batches.clear()

            self._opened = False

    def __enter__(self) -> TextStorage:
        """Enter context manager."""
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager."""
        self.close()

    # =========================================================================
    # Transaction Management
    # =========================================================================

    @overload
    def begin(self, *, read_only: Literal[True]) -> SnapshotProtocol: ...

    @overload
    def begin(self, *, write_only: Literal[True]) -> WriteBatchProtocol: ...

    @overload
    def begin(
        self, *, read_only: Literal[False], write_only: Literal[False]
    ) -> TransactionProtocol: ...

    def begin(
        self,
        *,
        read_only: bool = False,
        write_only: bool = False,
    ) -> WriteBatchProtocol | SnapshotProtocol | TransactionProtocol:
        """Begin new transaction with specified access level.

        Args:
            read_only: If True, creates a read-only snapshot
            write_only: If True, creates a write-only batch

        Returns:
            SnapshotProtocol if read_only=True
            WriteBatchProtocol if write_only=True
            TransactionProtocol otherwise

        Raises:
            StorageOperationError: If transaction creation fails
        """
        if read_only:
            return self.begin_snapshot()
        elif write_only:
            return self.begin_write_batch()
        else:
            return self.begin_transaction()

    def begin_snapshot(self) -> TextSnapshot:
        """Begin read-only snapshot.

        Returns:
            New snapshot instance

        Raises:
            StorageOperationError: If snapshot creation fails
        """
        self._require_open()

        with self._lock:
            # Create snapshot with copy of current state
            snapshot = TextSnapshot(self, self._state.copy())
            self._active_snapshots.add(snapshot)
            return snapshot

    def begin_transaction(self) -> TextTransaction:
        """Begin read-write transaction.

        Returns:
            New transaction instance

        Raises:
            StorageOperationError: If transaction creation fails
        """
        self._require_open()

        # Enforce single-writer semantics: only one transaction or write batch
        # may be active at a time. Other writers block until the current one
        # commits or aborts.
        self._write_lock.acquire()
        try:
            with self._lock:
                # Create transaction with copy of current state
                transaction = TextTransaction(self, self._state.copy())
                self._active_transactions.add(transaction)
                return transaction
        except Exception:
            # If creation fails, release lock so other writers aren't blocked
            self._write_lock.release()
            raise

    def begin_write_batch(self) -> TextWriteBatch:
        """Begin write-only batch.

        Returns:
            New write batch instance

        Raises:
            StorageOperationError: If batch creation fails
        """
        self._require_open()

        # Enforce single-writer semantics: only one transaction or write batch
        # may be active at a time. Other writers block until the current one
        # completes.
        self._write_lock.acquire()
        try:
            with self._lock:
                # Create write batch with copy of current state
                write_batch = TextWriteBatch(self, self._state.copy())
                self._active_write_batches.add(write_batch)
                return write_batch
        except Exception:
            # If creation fails, release lock so other writers aren't blocked
            self._write_lock.release()
            raise

    @contextmanager
    def transaction(self) -> Iterator[TextTransaction]:
        """Context manager for transactions: commit on success, abort on exception."""
        txn = self.begin_transaction()
        try:
            yield txn
        except Exception:
            if not txn._committed and not txn._aborted:
                txn.abort()
            raise
        else:
            if not txn._committed and not txn._aborted:
                txn.commit()

    @contextmanager
    def snapshot(self) -> Iterator[TextSnapshot]:
        """Context manager for read-only snapshots: always closes snapshot on exit."""
        snap = self.begin_snapshot()
        try:
            yield snap
        finally:
            try:
                snap.close()
            except Exception as e:
                logger.error("Failed to close snapshot", extra={"error": str(e)}, exc_info=True)

    @contextmanager
    def batch_write(self) -> Iterator[TextWriteBatch]:
        """Context manager for write batches: write on success, abort on exception."""
        batch = self.begin_write_batch()
        try:
            yield batch
        except Exception:
            if not batch._written and not batch._aborted:
                batch.abort()
            raise
        else:
            if not batch._written and not batch._aborted:
                batch.write()
