"""Backend specs for the storage conformance suite.

Each spec opens a fresh storage under a directory. The suite parametrizes
every case over ``BACKENDS``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from virtuals.codecs import BinaryCodec


if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class BackendSpec:
    name: str
    process_model: str  # "multi" | "single"
    factory: Callable[..., Any]

    def make(self, root: Path, *, lock_timeout: float | None = None) -> Any:
        """Build an unopened storage rooted at ``root``.

        ``lock_timeout`` (seconds) shortens the backend's wait for a held
        write lock where it has such a knob, so contention surfaces as an
        error instead of a wait.
        """
        return self.factory(Path(root), lock_timeout)

    def has_write_slot(self, storage: Any) -> bool:
        return callable(getattr(storage, "areserve_write_slot", None))


def _sqlite(root: Path, lock_timeout: float | None):
    from virtuals.storages.sqlite import SQLiteStorage

    return SQLiteStorage(
        path=root / "db.sqlite",
        codec=BinaryCodec(),
        busy_timeout=5.0 if lock_timeout is None else lock_timeout,
    )


def _lmdb(root: Path, lock_timeout: float | None):
    from virtuals.storages.lmdb import LMDBStorage

    return LMDBStorage(path=root / "lmdb", codec=BinaryCodec(), map_size=1 << 26)


def _rocksdb(root: Path, lock_timeout: float | None):
    from virtuals.storages.rocksdb import RocksDBStorage

    kwargs: dict[str, Any] = {}
    if lock_timeout is not None:
        ms = max(1, int(lock_timeout * 1000))
        kwargs["txn_db_options"] = {"transaction_lock_timeout": ms, "default_lock_timeout": ms}
    return RocksDBStorage(path=root / "rocks", codec=BinaryCodec(), **kwargs)


def _mem(root: Path, lock_timeout: float | None):
    from virtuals.storages.mem import InMemoryStorage

    return InMemoryStorage(codec=BinaryCodec())


BACKENDS: dict[str, BackendSpec] = {
    "sqlite": BackendSpec("sqlite", "multi", _sqlite),
    "lmdb": BackendSpec("lmdb", "multi", _lmdb),
    "rocksdb": BackendSpec("rocksdb", "single", _rocksdb),
    "mem": BackendSpec("mem", "single", _mem),
}
