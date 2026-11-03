"""Storage op counting for cost tests, on every backend.

Counts point ops (``get``, ``exists``, ``put``, ``delete``) and ``scan``
calls on the transaction class, plus the rows a scan decodes through the
codec (``scan_rows``). mem decodes nothing (NoOp codec), so its row count
is always 0; sqlite, lmdb and rocksdb decode every row they look at
exactly once.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from virtuals.codecs import BinaryCodec, NoOpCodec
from virtuals.navigator import Navigator


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import pytest

    from virtuals.tkv.storage import StorageProtocol


__all__ = [
    "BACKENDS",
    "COUNTED_OPS",
    "POINT_OPS",
    "SEEKING_BACKENDS",
    "install_counter",
    "measure",
    "open_backend",
]

POINT_OPS = ("get", "exists", "put", "delete")
COUNTED_OPS = (*POINT_OPS, "scan")
BACKENDS = ("mem", "sqlite", "lmdb", "rocksdb")
SEEKING_BACKENDS = ("sqlite", "lmdb", "rocksdb")


class _CountingCodec:
    """Codec proxy that counts rows a scan visits (one ``decode_key`` each)."""

    def __init__(self, inner: object, counts: Counter) -> None:
        self._inner = inner
        self._counts = counts

    def decode_key(self, encoded: object) -> object:
        self._counts["scan_rows"] += 1
        return self._inner.decode_key(encoded)  # type: ignore[attr-defined]

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)


def open_backend(backend: str, tmp_path: Path) -> StorageProtocol:
    """Open a storage of the named backend under ``tmp_path``."""
    if backend == "mem":
        from virtuals.storages.mem import InMemoryStorage

        storage = InMemoryStorage(codec=NoOpCodec())
    elif backend == "sqlite":
        from virtuals.storages.sqlite import SQLiteStorage

        storage = SQLiteStorage(path=tmp_path / "kv.sqlite", codec=BinaryCodec())
    elif backend == "lmdb":
        from virtuals.storages.lmdb import LMDBStorage

        storage = LMDBStorage(path=tmp_path / "kv.lmdb", codec=BinaryCodec(), map_size=2**28)
    elif backend == "rocksdb":
        from virtuals.storages.rocksdb import RocksDBStorage

        storage = RocksDBStorage(path=tmp_path / "db", codec=BinaryCodec())
    else:
        raise ValueError(backend)
    storage.open()
    return storage


def install_counter(storage: StorageProtocol, monkeypatch: pytest.MonkeyPatch) -> Counter:
    """Wrap the storage's transaction class and codec; return the live counter."""
    counter: Counter = Counter()
    with storage.transaction() as tx:
        tx_cls = type(tx)
    for name in COUNTED_OPS:
        original = getattr(tx_cls, name)

        def counted(self, *args, _name=name, _original=original, **kwargs):
            counter[_name] += 1
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(tx_cls, name, counted)
    monkeypatch.setattr(storage, "_codec", _CountingCodec(storage._codec, counter))  # type: ignore[attr-defined]
    return counter


def measure(storage: StorageProtocol, counts: Counter, op: Callable[[object], object]) -> Counter:
    """Run ``op`` on the root view in its own transaction, return the ops it cost."""
    counts.clear()
    with storage.transaction() as tx:
        op(Navigator(storage).root(tx))
    return Counter(counts)
