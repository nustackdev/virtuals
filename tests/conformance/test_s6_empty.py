"""Contract section 6: ``EMPTY`` is never stored, the raw doors included.

Every view door already refuses the absent sentinel. Layer 1 refuses it
too, on ``tx.put`` and ``batch.put``, before anything is written: a stored
sentinel would read back as a second ``Empty`` that is not ``EMPTY``,
exist, and show up in scans.
"""

from __future__ import annotations

import pytest

from virtuals.tkv import StorageScanOptions
from virtuals.tkv.filter import PrefixFilter
from virtuals.tkv.types import EMPTY


def _rows(storage, prefix):
    with storage.snapshot() as s:
        opts = StorageScanOptions(start=prefix, break_filter=PrefixFilter(prefix=prefix))
        return dict(s.scan(opts).items())


@pytest.mark.timeout(20)
def test_s6_raw_put_rejects_empty(storage):
    """``tx.put`` and ``batch.put`` raise ``ValueError`` on EMPTY and store nothing."""
    with storage.transaction() as t:
        t.put(("p", 0), 0)

    with storage.transaction() as t:
        with pytest.raises(ValueError, match="EMPTY"):
            t.put(("p", 1), EMPTY)
        # the transaction stays usable and commits the rest
        t.put(("p", 2), 2)
        assert t.get(("p", 1)) is EMPTY
        assert not t.exists(("p", 1))

    with storage.batch_write() as b:
        with pytest.raises(ValueError, match="EMPTY"):
            b.put(("p", 3), EMPTY)
        b.put(("p", 4), 4)

    assert _rows(storage, ("p",)) == {("p", 0): 0, ("p", 2): 2, ("p", 4): 4}
    with storage.snapshot() as s:
        for k in (("p", 1), ("p", 3)):
            assert s.get(k) is EMPTY
            assert not s.exists(k)
        assert s.multiget([("p", 1), ("p", 3), ("p", 4)]) == {("p", 4): 4}
