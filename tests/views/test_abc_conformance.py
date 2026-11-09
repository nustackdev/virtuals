"""Every built-in view is the ``collections.abc`` type it claims, and behaves like the built-in.

Differential tests: each ABC method runs on a view and on the Python value
it stands for (``dict``, ``list``, ``tuple``, ``bytearray``, ``set``,
``frozenset``, a ``deque(maxlen=)`` for a ring), from the same data. The
result, the error type, and the state left behind must match. Both facets
of a view with an eager and a lazy facet are run.

Where a layout cannot do what the built-in does, the view raises, and the
test pins that error:

- ``reversed()`` on a kh57 view: TypeError, as on a plain ``Mapping``
  (rows sit in kh57 level order, no reverse path).
- a write at a position on a ring (``ring[i] = x``, ``del ring[i]``):
  TypeError, as on a ``tuple``; ``insert``, ``pop``, ``remove``,
  ``reverse`` do not exist.
- a key or index keeps its kind: writing a compound value over a primitive
  one, or the reverse, raises ``ContainerTypeError`` and changes nothing
  (docs/view/overview.md, "Write rules"). Delete first, then set.

Documented differences, not checked against the built-in:

- iteration order follows the layout (key order for dict, flat and light
  views, int order for kh57, insertion order for indexed, fractional and
  log views). The data below is inserted in key order so every layout
  agrees with ``dict``.
- ``popitem`` pops the last pair in the view's order, as ``dict`` does;
  a kh57 view has no order to pop from the end of and pops the first.

What a built-in has beyond its ABC (``copy``, ``|``, ``sort``, slice
writes, ``union``, ...) is checked in ``test_builtin_conformance.py``.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, MutableMapping, MutableSequence, MutableSet, Sequence, Set

import pytest
from tests.support.differential import plain, same

from virtuals._views import (
    ByteArrayView,
    EagerDictView,
    EagerFractionalIndexedDictView,
    EagerIndexedDictView,
    EagerKh57View,
    EagerListView,
    EagerLogIndexedDictView,
    FlatDictView,
    FlatListView,
    FrozenSetView,
    LazyDictView,
    LazyFractionalIndexedDictView,
    LazyIndexedDictView,
    LazyKh57View,
    LazyListView,
    LazyLogIndexedDictView,
    LightDictView,
    RingView,
    SetView,
    TupleView,
)
from virtuals.container import ContainerTypeError
from virtuals.view import Faceted, ViewBase


# =============================================================================
# MAPPINGS
# =============================================================================

NESTED = {"a": 1, "b": {"x": [1, 2]}, "c": [3, {"y": 4}], "d": None}
FLAT = {"a": 1, "b": "two", "c": None, "d": 4.5}
NESTED_INT = dict(zip((1, 2, 3, 4), NESTED.values(), strict=True))

MAPPINGS = [
    pytest.param(EagerDictView, NESTED, id="dict"),
    pytest.param(LazyDictView, NESTED, id="dict-lazy"),
    pytest.param(EagerIndexedDictView, NESTED, id="indexed"),
    pytest.param(LazyIndexedDictView, NESTED, id="indexed-lazy"),
    pytest.param(EagerFractionalIndexedDictView, NESTED, id="fractional"),
    pytest.param(LazyFractionalIndexedDictView, NESTED, id="fractional-lazy"),
    pytest.param(EagerLogIndexedDictView, NESTED, id="log"),
    pytest.param(LazyLogIndexedDictView, NESTED, id="log-lazy"),
    pytest.param(EagerKh57View, NESTED_INT, id="kh57"),
    pytest.param(LazyKh57View, NESTED_INT, id="kh57-lazy"),
    pytest.param(FlatDictView, FLAT, id="flat"),
    pytest.param(LightDictView, FLAT, id="light"),
]
ORDERLESS = (EagerKh57View, LazyKh57View)


@pytest.fixture
def mapping(root_view, view_cls, data):
    """A view of ``view_cls`` holding ``data``, its reference dict, and fresh keys/values."""
    view = root_view.open_child("m", view_cls)
    view.store(data)
    keys = list(data)
    new = (5, 6) if isinstance(keys[0], int) else ("e", "f")  # sorts after the rest
    nested = view_cls not in (FlatDictView, LightDictView)
    return view, dict(data), keys, new, ({"z": [7]} if nested else 7)


@pytest.mark.parametrize(("view_cls", "data"), MAPPINGS)
def test_mapping_is_its_abc(mapping, view_cls):
    view, *_ = mapping
    assert isinstance(view, Mapping)
    assert isinstance(view, MutableMapping)
    with pytest.raises(TypeError):
        hash(view)  # unhashable, as a dict


@pytest.mark.parametrize(("view_cls", "data"), MAPPINGS)
def test_mapping_reads(mapping):
    view, ref, keys, new, _ = mapping
    for key in (*keys, new[0]):
        same(view, ref, lambda m, k=key: m[k])
        same(view, ref, lambda m, k=key: m.get(k))
        same(view, ref, lambda m, k=key: m.get(k, "dflt"))
        same(view, ref, lambda m, k=key: k in m)
    same(view, ref, len)
    same(view, ref, list)
    same(view, ref, lambda m: list(m.keys()))
    same(view, ref, lambda m: list(m.values()))
    same(view, ref, lambda m: list(m.items()))
    same(view, ref, lambda m: len(m.keys()))
    same(view, ref, lambda m: keys[0] in m.keys())
    same(view, ref, lambda m: (keys[0], m[keys[0]]) in m.items())
    same(view, ref, lambda m: set(m.keys()) & {keys[0], new[0]})
    assert view.keys() == ref.keys()


@pytest.mark.parametrize(("view_cls", "data"), MAPPINGS)
def test_mapping_equality(mapping):
    view, ref, keys, new, _ = mapping
    assert view == ref
    assert ref == view
    assert not (view != ref)
    assert view != {**ref, new[0]: 1}
    assert view != {k: ref[k] for k in keys[1:]}
    assert view != list(ref)
    if isinstance(view, Faceted):
        assert view == view.eager
        assert view == view.lazy


@pytest.mark.parametrize(("view_cls", "data"), MAPPINGS)
def test_mapping_reversed(mapping, view_cls):
    view, ref, *_ = mapping
    if view_cls in ORDERLESS:
        with pytest.raises(TypeError, match="not reversible"):
            reversed(view)
        return
    same(view, ref, lambda m: list(reversed(m)))
    same(view, ref, lambda m: list(reversed(m.keys())))
    same(view, ref, lambda m: list(reversed(m.values())))
    same(view, ref, lambda m: list(reversed(m.items())))


@pytest.mark.parametrize(("view_cls", "data"), MAPPINGS)
def test_mapping_set_and_delete(mapping):
    view, ref, keys, new, value = mapping
    same(view, ref, lambda m: m.__setitem__(new[0], value))
    same(view, ref, lambda m: m.__setitem__(keys[0], 9))  # same kind: a primitive over a primitive
    same(view, ref, lambda m: m.__setitem__(keys[1], value))  # same kind
    same(view, ref, lambda m: m.__delitem__(keys[1]))
    same(view, ref, lambda m: m.__delitem__(keys[1]))  # KeyError
    same(view, ref, lambda m: m.__delitem__(new[1]))  # KeyError


@pytest.mark.parametrize(("view_cls", "data"), MAPPINGS)
def test_mapping_pop(mapping):
    view, ref, keys, new, _ = mapping
    for key in keys[1:3]:  # a nested value: a lazy pop hands back data, not a dead view
        same(view, ref, lambda m, k=key: m.pop(k))
    same(view, ref, lambda m: m.pop(keys[1]))  # KeyError
    same(view, ref, lambda m: m.pop(keys[1], "dflt"))
    same(view, ref, lambda m: m.pop(new[0], None))


@pytest.mark.parametrize(("view_cls", "data"), MAPPINGS)
def test_mapping_popitem(mapping, view_cls):
    view, ref, *_ = mapping
    if view_cls in ORDERLESS:  # no order to pop from the end of: the first, as the ABC does
        ref = dict(reversed(ref.items()))
    while ref:
        same(view, ref, lambda m: m.popitem())
    same(view, ref, lambda m: m.popitem())  # KeyError


@pytest.mark.parametrize(("view_cls", "data"), MAPPINGS)
def test_mapping_setdefault(mapping, view_cls):
    view, ref, keys, new, value = mapping
    same(view, ref, lambda m: m.setdefault(keys[0], "dflt"))
    same(view, ref, lambda m: m.setdefault(new[0], value))
    same(view, ref, lambda m: m.setdefault(new[1]))
    if view_cls in (LazyDictView, LazyIndexedDictView, LazyKh57View, LazyLogIndexedDictView):
        held = view.setdefault(new[0], value)  # lazy: the child view, read back as the facet reads
        assert isinstance(held, ViewBase)
        assert held == value


@pytest.mark.parametrize(("view_cls", "data"), MAPPINGS)
def test_mapping_update(mapping):
    view, ref, keys, new, value = mapping

    class KeysOnly:
        def keys(self):
            return [new[1]]

        def __getitem__(self, key):
            return "k"

    same(view, ref, lambda m: m.update({new[0]: value, keys[0]: 9}))
    same(view, ref, lambda m: m.update([(keys[3], 2), (new[1], 3)]))
    same(view, ref, lambda m: m.update(KeysOnly()))
    same(view, ref, lambda m: m.update())
    if isinstance(keys[0], str):
        same(view, ref, lambda m: m.update(f=1, a=2))
        same(view, ref, lambda m: m.update({"d": 0}, d=1))
    same(view, ref, lambda m: m.update([(1, 2, 3)]))  # ValueError: not a pair


@pytest.mark.parametrize(("view_cls", "data"), [p for p in MAPPINGS if p.values[1] is not FLAT])
def test_mapping_kind_change_raises(mapping):
    view, ref, keys, _, value = mapping
    for key, kind_change in ((keys[0], value), (keys[1], 5)):
        with pytest.raises(ContainerTypeError):
            view[key] = kind_change
        with pytest.raises(ContainerTypeError):
            view.update({key: kind_change})
    assert plain(view) == ref
    del view[keys[0]]
    view[keys[0]] = value  # delete first, then set
    assert plain(view[keys[0]]) == value


@pytest.mark.parametrize(("view_cls", "data"), MAPPINGS)
def test_mapping_clear(mapping):
    view, ref, *_ = mapping
    same(view, ref, lambda m: m.clear())
    same(view, ref, len)
    same(view, ref, lambda m: m.popitem())  # KeyError


# =============================================================================
# SEQUENCES
# =============================================================================

NESTED_SEQ = [1, {"x": [1]}, [2, 3], None, 1]
FLAT_SEQ = [1, "two", None, 4.5, 1]
BYTES = bytearray(b"hello")

MUTABLE_SEQUENCES = [
    pytest.param(EagerListView, NESTED_SEQ, {"y": 5}, id="list"),
    pytest.param(LazyListView, NESTED_SEQ, {"y": 5}, id="list-lazy"),
    pytest.param(FlatListView, FLAT_SEQ, "z", id="flat-list"),
    pytest.param(ByteArrayView, BYTES, 33, id="bytearray"),
]
SEQUENCES = [
    *MUTABLE_SEQUENCES,
    pytest.param(TupleView, tuple(NESTED_SEQ), None, id="tuple"),
    pytest.param(RingView.of(10), NESTED_SEQ, None, id="ring"),
]
SLICES = [
    slice(None),
    slice(1, 3),
    slice(-2, None),
    slice(None, None, 2),
    slice(None, None, -1),
    slice(4, 0, -2),
    slice(3, 99),
    slice(3, 1),
]


@pytest.fixture
def sequence(root_view, view_cls, data, new):
    view = root_view.open_child("s", view_cls)
    view.store(data)
    return view, type(data)(data), new


@pytest.mark.parametrize(("view_cls", "data", "new"), SEQUENCES)
def test_sequence_is_its_abc(sequence, view_cls):
    view, *_ = sequence
    assert isinstance(view, Sequence)
    assert isinstance(view, MutableSequence) == (
        view_cls not in (TupleView,) and not issubclass(view_cls, RingView)
    )


@pytest.mark.parametrize(("view_cls", "data", "new"), SEQUENCES)
def test_sequence_reads(sequence):
    view, ref, _ = sequence
    as_list = list
    for i in (0, 2, -1, -len(ref), len(ref), -len(ref) - 1):
        same(view, ref, lambda s, i=i: s[i])
    same(view, ref, lambda s: s["0"])  # TypeError
    for sl in SLICES:
        same(view, ref, lambda s, sl=sl: s[sl])
    same(view, ref, len)
    same(view, ref, as_list)
    same(view, ref, lambda s: as_list(reversed(s)))
    for value in (ref[0], ref[1], "missing"):
        same(view, ref, lambda s, v=value: v in s)
        same(view, ref, lambda s, v=value: s.index(v))
        same(view, ref, lambda s, v=value: s.count(v))
    same(view, ref, lambda s: s.index(ref[0], 1))
    same(view, ref, lambda s: s.index(ref[0], -1))
    same(view, ref, lambda s: s.index(ref[0], 1, 3))  # ValueError for a list's 1 at [4]


@pytest.mark.parametrize(("view_cls", "data", "new"), MUTABLE_SEQUENCES)
def test_sequence_equality(sequence):
    view, ref, _ = sequence
    assert view == ref
    assert ref == view
    if isinstance(view, Faceted):
        assert view == view.eager
        assert view == view.lazy
    assert view != ref[:-1]
    with pytest.raises(TypeError):
        hash(view)


@pytest.mark.parametrize(("view_cls", "data", "new"), MUTABLE_SEQUENCES)
def test_sequence_set_and_delete(sequence):
    view, ref, new = sequence
    same(view, ref, lambda s: s.__setitem__(1, new))
    same(view, ref, lambda s: s.__setitem__(-1, 7))  # same kind: a primitive over a primitive
    same(view, ref, lambda s: s.__setitem__(len(s), new))  # IndexError
    same(view, ref, lambda s: s.__delitem__(1))
    same(view, ref, lambda s: s.__delitem__(-1))
    same(view, ref, lambda s: s.__delitem__(99))  # IndexError


@pytest.mark.parametrize(("view_cls", "data", "new"), MUTABLE_SEQUENCES[:2])
def test_sequence_kind_change_raises(sequence):
    view, ref, new = sequence
    with pytest.raises(ContainerTypeError):
        view[0] = new  # a compound over the primitive 1
    with pytest.raises(ContainerTypeError):
        view[1] = 5  # a primitive over a compound
    assert plain(view) == ref


@pytest.mark.parametrize(("view_cls", "data", "new"), MUTABLE_SEQUENCES)
def test_sequence_insert_append_extend(sequence):
    view, ref, new = sequence
    for index in (0, 2, -1, -99, 99):
        same(view, ref, lambda s, i=index: s.insert(i, new))
    same(view, ref, lambda s: s.append(new))
    same(view, ref, lambda s: s.extend([new, new]))
    same(view, ref, lambda s: s.extend(iter([new])))
    same(view, ref, lambda s: s.extend(s))  # doubles, as list.extend(itself)


@pytest.mark.parametrize(("view_cls", "data", "new"), MUTABLE_SEQUENCES)
def test_sequence_pop_remove(sequence):
    view, ref, _ = sequence
    same(view, ref, lambda s: s.pop())
    same(view, ref, lambda s: s.pop(1))  # a nested item: a lazy pop hands back data
    same(view, ref, lambda s: s.pop(-99))  # IndexError
    same(view, ref, lambda s: s.remove(ref[0]))
    same(view, ref, lambda s: s.remove("missing"))  # ValueError
    while len(ref):
        same(view, ref, lambda s: s.pop(0))
    same(view, ref, lambda s: s.pop())  # IndexError, empty


@pytest.mark.parametrize(("view_cls", "data", "new"), MUTABLE_SEQUENCES)
def test_sequence_reverse_clear(sequence):
    view, ref, _ = sequence
    same(view, ref, lambda s: s.reverse())
    same(view, ref, lambda s: s.clear())
    same(view, ref, lambda s: s.reverse())


@pytest.mark.parametrize(("view_cls", "data", "new"), MUTABLE_SEQUENCES)
def test_sequence_in_place_add(sequence):
    view, ref, new = sequence
    held = view
    held += type(ref)([new])
    ref += type(ref)([new])
    assert held is view
    assert plain(view) == ref


def test_tuple_has_no_writes(root_view):
    view = root_view.open_child("t", TupleView)
    view.store((1, 2))
    with pytest.raises(TypeError):
        view[0] = 3  # type: ignore[index]
    for name in ("append", "insert", "pop", "remove", "reverse", "clear", "extend"):
        assert not hasattr(view, name), name


# =============================================================================
# RING: a Sequence plus append / extend / clear, against deque(maxlen=)
# =============================================================================


@pytest.mark.parametrize("capacity", [1, 3])
def test_ring_appends_like_a_bounded_deque(root_view, capacity):
    view = root_view.open_child("r", RingView.of(capacity))
    ref: deque = deque(maxlen=capacity)
    view.ensure_created()

    def check():
        assert list(view) == list(ref)
        assert len(view) == len(ref)
        for i in range(-len(ref) - 1, len(ref) + 1):
            same(view, ref, lambda s, i=i: s[i], state=list)
        assert view[:] == list(ref)

    for step in (["a"], ["b", "c"], [], list("defgh"), [[1, "x"]]):
        assert view.append(step[0]) is None if step else view.extend(step) is None
        if step:
            ref.append(step[0])
            view.extend(step[1:])
            ref.extend(step[1:])
        check()
    same(view, ref, lambda s: s.clear(), state=list)
    check()
    view.extend(["z"])
    ref.extend(["z"])
    check()


def test_ring_refuses_writes_at_a_position(root_view):
    view = root_view.open_child("r", RingView.of(3))
    view.extend(["a", "b"])
    with pytest.raises(TypeError):
        view[0] = "x"
    with pytest.raises(TypeError):
        del view[0]
    for name in ("insert", "pop", "remove", "reverse"):
        assert not hasattr(view, name), name
    assert list(view) == ["a", "b"]


# =============================================================================
# SETS
# =============================================================================

MEMBERS = {1, "two", (3, 4), None}
SETS = [
    pytest.param(SetView, set, id="set"),
    pytest.param(FrozenSetView, frozenset, id="frozenset"),
]


@pytest.fixture
def a_set(root_view, view_cls, kind):
    view = root_view.open_child("s", view_cls)
    view.store(MEMBERS)
    return view, kind(MEMBERS)


@pytest.mark.parametrize(("view_cls", "kind"), SETS)
def test_set_is_its_abc(a_set, view_cls):
    view, _ = a_set
    assert isinstance(view, Set)
    assert isinstance(view, MutableSet) == (view_cls is SetView)
    with pytest.raises(TypeError):
        hash(view)


@pytest.mark.parametrize(("view_cls", "kind"), SETS)
def test_set_reads(a_set):
    view, ref = a_set
    for value in (1, (3, 4), None, "missing", [1]):  # a list: TypeError, unhashable
        same(view, ref, lambda s, v=value: v in s)
    same(view, ref, len)
    same(view, ref, lambda s: set(iter(s)))
    others = [set(), {1}, set(MEMBERS), {*MEMBERS, 5}, {1, 5}, {5}]
    for other in (*others, *map(frozenset, others)):
        for op in ("__le__", "__lt__", "__ge__", "__gt__", "__eq__", "__ne__"):
            same(view, ref, lambda s, o=other, op=op: getattr(s, op)(o))
        for op in ("__and__", "__or__", "__sub__", "__xor__"):
            same(view, ref, lambda s, o=other, op=op: getattr(s, op)(o))
        same(view, ref, lambda s, o=other: o & s)
        same(view, ref, lambda s, o=other: o | s)
        same(view, ref, lambda s, o=other: o - s)
        same(view, ref, lambda s, o=other: o ^ s)
        same(view, ref, lambda s, o=other: o <= s)
        same(view, ref, lambda s, o=other: s.isdisjoint(o))
        same(view, ref, lambda s, o=other: s.issubset(o))
        same(view, ref, lambda s, o=other: s.issuperset(o))


@pytest.mark.parametrize(("view_cls", "kind"), SETS)
def test_set_operators_return_the_plain_type(a_set, kind):
    view, _ = a_set
    for result in (view & {1}, view | {1}, view - {1}, view ^ {1}, {1} & view, {1} | view):
        assert type(result) is kind or type(result) is set  # reflected on a set gives a set


def test_set_mutations(root_view):
    view = root_view.open_child("s", SetView)
    view.store(MEMBERS)
    ref = set(MEMBERS)
    same(view, ref, lambda s: s.add(5))
    same(view, ref, lambda s: s.add(5))
    same(view, ref, lambda s: s.add([5]))  # TypeError, unhashable
    same(view, ref, lambda s: s.discard(5))
    same(view, ref, lambda s: s.discard(5))
    same(view, ref, lambda s: s.remove(1))
    same(view, ref, lambda s: s.remove(1))  # KeyError
    popped = view.pop()
    assert popped in ref
    ref.discard(popped)
    assert plain(view) == ref
    for other in ({1, 2}, {None, 6}, set()):
        for op in ("__ior__", "__iand__", "__isub__", "__ixor__"):
            held = view
            result = getattr(view, op)(other)
            getattr(ref, op)(other)
            assert result is held
            assert plain(view) == ref, (op, other)
    same(view, ref, lambda s: s.clear())
    same(view, ref, lambda s: s.pop())  # KeyError, empty
