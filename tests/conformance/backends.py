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

    def make(self, root: Path, *, lock_timeout: float | None = None, codec: Any = None) -> Any:
        """Build an unopened storage rooted at ``root``.

        ``lock_timeout`` (seconds) shortens the backend's wait for a held
        write lock where it has such a knob, so contention surfaces as an
        error instead of a wait. ``codec`` replaces the default
        ``BinaryCodec`` (section 7 passes a counting one).
        """
        return self.factory(Path(root), lock_timeout, BinaryCodec() if codec is None else codec)

    def has_write_slot(self, storage: Any) -> bool:
        return callable(getattr(storage, "areserve_write_slot", None))


def _sqlite(root: Path, lock_timeout: float | None, codec: Any):
    from virtuals.storages.sqlite import SQLiteStorage

    return SQLiteStorage(
        path=root / "db.sqlite",
        codec=codec,
        busy_timeout=5.0 if lock_timeout is None else lock_timeout,
    )


def _lmdb(root: Path, lock_timeout: float | None, codec: Any):
    from virtuals.storages.lmdb import LMDBStorage

    kwargs: dict[str, Any] = {}
    if lock_timeout is not None:
        kwargs["busy_timeout"] = lock_timeout  # in-process slot only, see contract section 4
    return LMDBStorage(path=root / "lmdb", codec=codec, map_size=1 << 26, **kwargs)


def _rocksdb(root: Path, lock_timeout: float | None, codec: Any):
    from virtuals.storages.rocksdb import RocksDBStorage

    kwargs: dict[str, Any] = {}
    if lock_timeout is not None:
        ms = max(1, int(lock_timeout * 1000))
        kwargs["txn_db_options"] = {"transaction_lock_timeout": ms, "default_lock_timeout": ms}
        kwargs["busy_timeout"] = lock_timeout  # in-process slot, see contract section 4
    return RocksDBStorage(path=root / "rocks", codec=codec, **kwargs)


def _mem(root: Path, lock_timeout: float | None, codec: Any):
    from virtuals.storages.mem import InMemoryStorage

    kwargs: dict[str, Any] = {}
    if lock_timeout is not None:
        kwargs["busy_timeout"] = lock_timeout  # in-process slot only, see contract section 4
    return InMemoryStorage(codec=codec, **kwargs)


BACKENDS: dict[str, BackendSpec] = {
    "sqlite": BackendSpec("sqlite", "multi", _sqlite),
    "lmdb": BackendSpec("lmdb", "multi", _lmdb),
    "rocksdb": BackendSpec("rocksdb", "single", _rocksdb),
    "mem": BackendSpec("mem", "single", _mem),
}
