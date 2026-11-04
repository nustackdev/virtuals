"""Interrupts on the async reserve paths, and signal handlers left alone, on sqlite and lmdb.

Round 3 re-review (report 43): B2 and D1.
"""

from __future__ import annotations

import asyncio
import gc
import os
import signal
import time
from typing import TYPE_CHECKING

import pytest

from tests._backends.storages import _write_slot_cases as cases
from virtuals._backends.storages import _write_slot
from virtuals._backends.storages.lmdb import LMDBStorage
from virtuals._backends.storages.sqlite import SQLiteStorage
from virtuals.codecs import BinaryCodec


if TYPE_CHECKING:
    from pathlib import Path


def _open(backend: str, root: Path):
    if backend == "lmdb":
        st = LMDBStorage(path=root / "db", codec=BinaryCodec(), map_size=1 << 26, busy_timeout=2.0)
    else:
        st = SQLiteStorage(path=root / "db.sqlite", codec=BinaryCodec(), busy_timeout=2.0)
    st.open()
    return st


@pytest.fixture(params=["lmdb", "sqlite"])
def storage(request, tmp_path: Path):
    st = _open(request.param, tmp_path)
    yield st
    st.close()


def _kill_self_then(original):
    """Wrap ``original`` so the call first delivers SIGUSR1 to this process."""

    def wrapped(*args, **kwargs):
        os.kill(os.getpid(), signal.SIGUSR1)
        time.sleep(0.002)
        return original(*args, **kwargs)

    return wrapped


async def _reserve(st) -> None:
    slot = await st.areserve_write_slot()
    slot.release()


def _slot_is_free(st) -> None:
    gc.collect()
    assert not st._writer.lock.locked()
    assert st._writer._owner is None
    if isinstance(st, LMDBStorage):
        assert not st._live_writes and not st._gate_held
    with st.transaction() as tx:  # before: a timeout "held by a task that has ended"
        tx.put(("after",), b"1")


@pytest.mark.timeout(30)
def test_signal_as_the_async_reserve_resumes_gives_the_slot_back(storage, monkeypatch) -> None:
    """A signal as ``await areserve`` resumes, before its result is bound: the task releases its slot."""
    original = _write_slot.WriterLock.areserve

    async def areserve_then_signal(self, timeout):
        lock = await original(self, timeout)
        os.kill(os.getpid(), signal.SIGUSR1)  # the handler raises here, as at the await's return
        time.sleep(0.002)
        return lock

    monkeypatch.setattr(_write_slot.WriterLock, "areserve", areserve_then_signal)
    with cases.interrupt_after(60), pytest.raises(cases.Interrupt):
        asyncio.run(_reserve(storage))
    monkeypatch.undo()
    _slot_is_free(storage)


@pytest.mark.timeout(30)
def test_signal_as_the_lmdb_begin_resumes_aborts_the_begun_txn(tmp_path: Path, monkeypatch) -> None:
    """A signal right after the writer thread begun the LMDB txn for a reservation: aborted, not orphaned."""
    from virtuals._backends.storages.lmdb._writer import WriterThread

    st = _open("lmdb", tmp_path)
    original = WriterThread.arun

    async def arun_then_signal(self, *args, **kwargs):
        result = await original(self, *args, **kwargs)
        os.kill(os.getpid(), signal.SIGUSR1)
        time.sleep(0.002)
        return result

    monkeypatch.setattr(WriterThread, "arun", arun_then_signal)
    try:
        with cases.interrupt_after(60), pytest.raises(cases.Interrupt):
            asyncio.run(_reserve(st))
        monkeypatch.undo()
        st._writer_thread.run(lambda: None)  # the abort lands ahead of this
        _slot_is_free(st)  # before: live_writes 1, "another write transaction is active"
    finally:
        st.close()


@pytest.mark.timeout(30)
@pytest.mark.parametrize("backend", ["lmdb", "sqlite"])
def test_no_signal_handler_is_installed(backend: str, tmp_path: Path) -> None:
    """The storage never touches signal handlers: sync and async writes leave every one as it was.

    So asyncio's Ctrl-C handling on 3.11+ (installed only while the
    default handler is in place) and any handler the application set stay
    exactly as Python ships them.
    """
    before = {signum: signal.getsignal(signum) for signum in signal.valid_signals()}
    st = _open(backend, tmp_path)
    try:
        with st.transaction() as tx:
            tx.put(("k",), b"1")
        with st.batch_write() as batch:
            batch.put(("b",), b"1")

        async def reserved() -> None:
            slot = await st.areserve_write_slot()
            tx = st.begin_transaction(write_slot=slot)
            tx.put(("a",), b"1")
            tx.commit()

        asyncio.run(reserved())
        with st.snapshot() as snap:
            snap.get(("k",))
    finally:
        st.close()
    after = {signum: signal.getsignal(signum) for signum in signal.valid_signals()}
    assert after == before
    assert after[signal.SIGINT] is signal.default_int_handler
