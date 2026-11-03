"""The writer slot of an in-memory storage, taken ahead of a transaction."""

from __future__ import annotations

from .._write_slot import WriteSlot


__all__ = ["InMemoryWriteSlot"]


InMemoryWriteSlot = WriteSlot
"""Returned by ``InMemoryStorage.areserve_write_slot``; see ``WriteSlot``."""
