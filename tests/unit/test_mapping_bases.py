"""The mapping bases follow ``collections.abc`` and ``dict``, with no probing.

``MutableMappingBase.popitem`` pops the first key iteration yields, as
``MutableMapping.popitem`` does; ``DictMethods`` pops the last, as
``dict.popitem`` does. A plain mapping is not reversible, as
``collections.abc.Mapping`` is not.
"""

from __future__ import annotations

import pytest

from virtuals.collections import DictMethods, MutableMappingBase


class _Plain(MutableMappingBase):
    def __init__(self, data: dict) -> None:
        self._data = dict(data)

    def __getitem__(self, key: object) -> object:
        return self._data[key]

    def __setitem__(self, key: object, value: object) -> None:
        self._data[key] = value

    def __delitem__(self, key: object) -> None:
        del self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


class _Dict(_Plain, DictMethods):
    def __reversed__(self):
        return reversed(self._data)


def test_plain_mapping_pops_the_first_key():
    m = _Plain({"a": 1, "b": 2})
    assert m.popitem() == ("a", 1)
    assert m.popitem() == ("b", 2)
    with pytest.raises(KeyError):
        m.popitem()


def test_dict_methods_pop_the_last_key():
    m = _Dict({"a": 1, "b": 2})
    assert m.popitem() == ("b", 2)
    assert m.popitem() == ("a", 1)
    with pytest.raises(KeyError):
        m.popitem()


def test_plain_mapping_is_not_reversible():
    with pytest.raises(TypeError, match="not reversible"):
        reversed(_Plain({"a": 1}))
    assert list(reversed(_Dict({"a": 1, "b": 2}))) == ["b", "a"]
