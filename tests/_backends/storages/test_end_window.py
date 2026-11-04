"""An interrupt between a context's end flag and its close, then the end retried.

On lmdb and rocksdb the committed, aborted or written flag is set one
statement before ``_mark_closed``. An interrupt there (a signal handler
raising) leaves the backend work done and the bookkeeping missing. The
retry the caller then makes must not run the end again: a second commit or
write raises ``StorageClosedError``, a second abort is a no-op, both finish
the bookkeeping, and the writer slot is free afterwards. sqlite and mem set
closed first, so they have no such window.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from virtuals.codecs import BinaryCodec
from virtuals.tkv.storage import StorageClosedError


if TYPE_CHECKING:
    from pathlib import Path


def _open(backend: str, root: Path):
    if backend == "lmdb":
        from virtuals.storages.lmdb import LMDBStorage

        st = LMDBStorage(path=root / "kv.lmdb", codec=BinaryCodec(), map_size=2**26)
    else:
        from virtuals.storages.rocksdb import RocksDBStorage

        st = RocksDBStorage(path=root / "db", codec=BinaryCodec(), busy_timeout=2.0)
    st.open()
    return st


@pytest.fixture(params=["lmdb", "rocksdb"])
def storage(request, tmp_path: Path):
    st = _open(request.param, tmp_path)
    try:
        yield st
    finally:
        st.close()


class _Interrupt(BaseException):
    """What a signal handler raises: not an ``Exception``, so no except clause eats it."""


def _arm(monkeypatch, ctx, method: str = "_mark_closed") -> None:
    """Raise once at entry to ``method`` on ``ctx``, as an interrupt would."""
    cls = type(ctx)
    original = getattr(cls, method)
    fired = []

    def patched(self, *args, **kwargs):
        if self is ctx and not fired:
            fired.append(True)
            raise _Interrupt
        return original(self, *args, **kwargs)

    monkeypatch.setattr(cls, method, patched)


def _write(storage, value: int) -> None:
    with storage.transaction() as t:
        t.put(("k",), value)


def _read(storage) -> object:
    with storage.snapshot() as s:
        return s.get(("k",))


@pytest.mark.timeout(30)
@pytest.mark.parametrize("kind", ["tx", "batch"])
def test_retried_end_after_interrupt_in_the_close_window(storage, monkeypatch, kind):
    """A second commit or write after the window raises closed and applies nothing."""
    ctx = storage.begin_write_batch() if kind == "batch" else storage.begin_transaction()
    if kind == "tx":
        ctx.get(("k",))  # lmdb: begin the transaction for real, not pending
    ctx.put(("k",), 1)
    end = ctx.write if kind == "batch" else ctx.commit
    _arm(monkeypatch, ctx)
    with pytest.raises(_Interrupt):
        end()
    assert _read(storage) == 1  # the end itself landed

    with pytest.raises(StorageClosedError):
        end()  # the retry finishes the bookkeeping, runs no second end
    ctx.abort()  # a no-op after the end
    _write(storage, 2)  # the slot is free
    assert _read(storage) == 2

    with pytest.raises(StorageClosedError):
        end()
    assert _read(storage) == 2  # nothing re-applied


@pytest.mark.timeout(30)
@pytest.mark.parametrize("kind", ["tx", "batch"])
def test_retried_abort_after_interrupt_in_the_close_window(storage, monkeypatch, kind):
    """A second abort after the window is a no-op that frees the slot."""
    ctx = storage.begin_write_batch() if kind == "batch" else storage.begin_transaction()
    if kind == "tx":
        ctx.get(("k",))
    ctx.put(("k",), 1)
    _arm(monkeypatch, ctx)
    with pytest.raises(_Interrupt):
        ctx.abort()
    ctx.abort()
    with pytest.raises(StorageClosedError):
        (ctx.write if kind == "batch" else ctx.commit)()
    _write(storage, 2)
    assert _read(storage) == 2


@pytest.mark.timeout(30)
def test_rocksdb_batch_never_applies_twice(tmp_path, monkeypatch):
    """An interrupt right after the native write: a second write() re-applies nothing.

    The batch claims its end before the native write, so whatever the
    window, write() is one application at most.
    """
    st = _open("rocksdb", tmp_path)
    try:
        original = type(st)._write_batch

        def apply_then_interrupt(self, batch):
            original(self, batch)
            raise _Interrupt

        b = st.begin_write_batch()
        b.put(("k",), 1)
        monkeypatch.setattr(type(st), "_write_batch", apply_then_interrupt)
        with pytest.raises(_Interrupt):
            b.write()
        monkeypatch.setattr(type(st), "_write_batch", original)
        assert _read(st) == 1

        _write(st, 2)
        with pytest.raises(StorageClosedError):
            b.write()
        assert _read(st) == 2
    finally:
        st.close()
