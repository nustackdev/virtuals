"""The in-process writer slot of a SQLite storage, taken ahead of a transaction."""

from __future__ import annotations

from .._write_slot import WriteSlot


__all__ = ["SQLiteWriteSlot"]


SQLiteWriteSlot = WriteSlot
"""Returned by ``SQLiteStorage.areserve_write_slot``; see ``WriteSlot``."""
