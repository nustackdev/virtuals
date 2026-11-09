"""Differential checks: run the same call on a view and on the Python value it stands for."""

from __future__ import annotations

from virtuals.view import ViewBase


ERRORS = (KeyError, IndexError, ValueError, TypeError, AttributeError)


def plain(value):
    """``value`` with every view in it extracted to the Python value it stands for."""
    if isinstance(value, ViewBase):
        return value.eager.extract()  # type: ignore[attr-defined]
    if isinstance(value, tuple):
        return tuple(plain(v) for v in value)
    if isinstance(value, list):
        return [plain(v) for v in value]
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    return value


def outcome(fn, target):
    """``("ok", plain result)`` or ``("raise", the built-in error type it is)``."""
    try:
        return ("ok", plain(fn(target)))
    except ERRORS as e:
        for kind in ERRORS:
            if isinstance(e, kind):
                return ("raise", kind)
        raise  # pragma: no cover


def same(view, ref, fn, state=lambda x: x):
    """Run ``fn`` on both; the outcome and the state left behind must match."""
    got, want = outcome(fn, view), outcome(fn, ref)
    assert got == want, (got, want)
    assert plain(view) == state(ref), (plain(view), ref)
