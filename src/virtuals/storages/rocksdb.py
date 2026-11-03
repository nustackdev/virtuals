"""RocksDB storage backend implementation.

``RocksDBStorage`` is the serial variant: one writer at a time per storage,
every read at a real snapshot. ``RocksDBSerialStorage`` is the same class
under its variant name.
"""

from __future__ import annotations

from virtuals._backends.storages.rocksdb import RocksDBSerialStorage, RocksDBStorage


__all__ = [
    "RocksDBSerialStorage",
    "RocksDBStorage",
]
