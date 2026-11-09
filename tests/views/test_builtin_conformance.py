"""A view that stands in for a built-in is that built-in: every method, same results.

``test_abc_conformance.py`` checks the ``collections.abc`` part. This file
checks the rest of each built-in's API (``dict.copy``, ``dict |``,
``list.sort``, slice writes, ``set.union``, ``bytearray.split``, ...) the
same way: each call runs on the view and on the Python value it stands
for, from the same data, and the result, the error type and the state
left behind must match. Both facets of a view with an eager and a lazy
facet are run.

Which view claims which built-in:

- ``dict``: dict, fractional, log and indexed views (eager and lazy), flat
  and light dict views
- ``list``: list views (eager and lazy), flat list view
- ``tuple``: tuple view
- ``set`` / ``frozenset``: set view / frozenset view
- ``bytearray``: bytearray view

Kh57 views and rings stay at their ABC (``MutableMapping``, ``Sequence``).

Declined, pinned below, because a view cannot do them honestly:

- ``hash()`` on a tuple or frozenset view: the stored value can be
  replaced under the view, so a hash would not hold for its lifetime.
- ``dict.fromkeys`` and ``bytearray.fromhex``: they build a new value from
  nothing, and a view needs a place in storage to stand at.
- the buffer protocol on a bytearray view (``memoryview(view)``): a Python
  class cannot expose a buffer below 3.12, and a buffer over a copy would
  drop writes made through it. ``bytes(view)`` gives a plain copy.
"""

from __future__ import annotations

import pytest
from tests.support.differential import plain, same

from virtuals._views import (
    ByteArrayView,
    EagerDictView,
    EagerFractionalIndexedDictView,
    EagerIndexedDictView,
    EagerListView,
    EagerLogIndexedDictView,
    FlatDictView,
    FlatListView,
    FrozenSetView,
    LazyDictView,
    LazyFractionalIndexedDictView,
    LazyIndexedDictView,
    LazyListView,
    LazyLogIndexedDictView,
    LightDictView,
    SetView,
    TupleView,
)


# =============================================================================
# API SURFACE: every name the built-in has, the view has
# =============================================================================

# Object plumbing every class has, not part of a type's API.
PLUMBING = {
    "__class__",
    "__class_getitem__",
    "__delattr__",
    "__dir__",
    "__doc__",
    "__format__",
    "__getattribute__",
    "__getnewargs__",
    "__getstate__",
    "__init__",
    "__init_subclass__",
    "__new__",
    "__reduce__",
    "__reduce_ex__",
    "__repr__",
    "__setattr__",
    "__sizeof__",
    "__str__",
    "__subclasshook__",
    "__alloc__",
}
DECLINED = {
    dict: {"fromkeys"},
    tuple: {"__hash__"},
    frozenset: {"__hash__"},
    bytearray: {"fromhex", "__buffer__", "__release_buffer__"},
}
CLAIMS = [
    (dict, EagerDictView),
    (dict, LazyDictView),
    (dict, EagerFractionalIndexedDictView),
    (dict, LazyFractionalIndexedDictView),
    (dict, EagerLogIndexedDictView),
    (dict, LazyLogIndexedDictView),
    (dict, EagerIndexedDictView),
    (dict, LazyIndexedDictView),
    (dict, FlatDictView),
    (dict, LightDictView),
    (list, EagerListView),
    (list, LazyListView),
    (list, FlatListView),
    (tuple, TupleView),
    (set, SetView),
    (frozenset, FrozenSetView),
    (bytearray, ByteArrayView),
]


@pytest.mark.parametrize(("kind", "view_cls"), CLAIMS, ids=lambda x: x.__name__)
def test_view_has_the_whole_builtin_api(kind, view_cls):
    declined = DECLINED.get(kind, set())
    missing = [
        name
        for name in dir(kind)
        if name not in PLUMBING
        and name not in declined
        and getattr(kind, name) is not None
        and getattr(view_cls, name, None) is None
    ]
    assert not missing, missing
    for name in declined - {"__hash__", "__buffer__", "__release_buffer__"}:
        assert not hasattr(view_cls, name), name


@pytest.mark.parametrize(("kind", "view_cls"), CLAIMS, ids=lambda x: x.__name__)
def test_view_adds_no_helpers_the_builtin_lacks(kind, view_cls):
    """The facets carry the built-in's surface, not a functional toolkit of their own."""
    for name in ("map", "map_values", "map_items", "filter", "reduce", "find_key", "find_item"):
        assert not hasattr(view_cls, name), name
    if kind is not bytearray:  # bytearray.find is the built-in's own
        assert not hasattr(view_cls, "find")
    assert not hasattr(view_cls, "find_index")


# =============================================================================
# DICT
# =============================================================================

NESTED = {"a": 1, "b": {"x": [1, 2]}, "c": [3, {"y": 4}], "d": None}
FLAT = {"a": 1, "b": "two", "c": None, "d": 4.5}

DICTS = [
    pytest.param(EagerDictView, NESTED, id="dict"),
    pytest.param(LazyDictView, NESTED, id="dict-lazy"),
    pytest.param(EagerFractionalIndexedDictView, NESTED, id="fractional"),
    pytest.param(LazyFractionalIndexedDictView, NESTED, id="fractional-lazy"),
    pytest.param(EagerLogIndexedDictView, NESTED, id="log"),
    pytest.param(LazyLogIndexedDictView, NESTED, id="log-lazy"),
    pytest.param(EagerIndexedDictView, NESTED, id="indexed"),
    pytest.param(LazyIndexedDictView, NESTED, id="indexed-lazy"),
    pytest.param(FlatDictView, FLAT, id="flat"),
    pytest.param(LightDictView, FLAT, id="light"),
]


@pytest.fixture
def a_dict(root_view, view_cls, data):
    view = root_view.open_child("m", view_cls)
    view.store(data)
    return view, dict(data)


@pytest.fixture
def other_dict(root_view, view_cls):
    """A second view of the same class, to merge with."""
    view = root_view.open_child("o", view_cls)
    view.store({"a": 7, "z": 8})
    return view


@pytest.mark.parametrize(("view_cls", "data"), DICTS)
def test_dict_copy(a_dict):
    view, ref = a_dict
    same(view, ref, lambda m: m.copy())
    copied = view.copy()
    assert type(copied) is dict
    copied["new"] = 1
    assert "new" not in view


@pytest.mark.parametrize(("view_cls", "data"), DICTS)
def test_dict_merge(a_dict, other_dict):
    view, ref = a_dict
    for other in ({}, {"a": 9}, {"e": 5, "b": 0}, other_dict.copy()):
        same(view, ref, lambda m, o=other: m | o)
        same(view, ref, lambda m, o=other: o | m)
    for refused in ([("e", 1)], {"e"}, 5, None):
        same(view, ref, lambda m, o=refused: m | o)  # TypeError: dict | dict only
        same(view, ref, lambda m, o=refused: o | m)
    assert type(view | {}) is dict
    assert type({} | view) is dict
    assert view | other_dict == ref | other_dict.copy()
    assert other_dict | view == other_dict.copy() | ref


@pytest.mark.parametrize(("view_cls", "data"), DICTS)
def test_dict_merge_in_place(a_dict, other_dict):
    view, ref = a_dict

    def ior(m, other):
        held = m
        m |= other
        assert m is held
        return None

    same(view, ref, lambda m: ior(m, {"e": 5, "a": 2}))
    same(view, ref, lambda m: ior(m, [("f", 6)]))  # |= takes what update takes
    same(view, ref, lambda m: ior(m, 5))  # TypeError
    same(view, ref, lambda m: ior(m, [(1, 2, 3)]))  # ValueError
    view |= other_dict
    ref |= other_dict.copy()
    assert plain(view) == ref


@pytest.mark.parametrize(("view_cls", "data"), DICTS)
def test_dict_has_no_ordering(a_dict):
    view, ref = a_dict
    same(view, ref, lambda m: m < {})  # TypeError, as dict
    same(view, ref, lambda m: m >= ref)
    same(view, ref, lambda m: ref > m)


# =============================================================================
# LIST
# =============================================================================

LISTS = [
    pytest.param(EagerListView, id="list"),
    pytest.param(LazyListView, id="list-lazy"),
    pytest.param(FlatListView, id="flat-list"),
]
NUMBERS = [3, 1, 2, 1, 5]
NESTED_LISTS = [[2], [1, 3], [1], [2, 0]]  # lists order: a lazy facet compares child list views


def a_list(root_view, view_cls, data):
    view = root_view.open_child("s", view_cls)
    view.store(data)
    return view, list(data)


@pytest.mark.parametrize("view_cls", LISTS)
def test_list_copy(root_view, view_cls):
    view, ref = a_list(root_view, view_cls, NUMBERS)
    same(view, ref, lambda s: s.copy())
    assert type(view.copy()) is list


@pytest.mark.parametrize("view_cls", LISTS)
@pytest.mark.parametrize("data", [NUMBERS, NESTED_LISTS, ["b", "a", "c"], []])
def test_list_sort(root_view, view_cls, data):
    view, ref = a_list(root_view, view_cls, data)
    same(view, ref, lambda s: s.sort())
    same(view, ref, lambda s: s.sort(reverse=True))
    same(view, ref, lambda s: s.sort(key=len if data and not isinstance(data[0], int) else abs))
    same(view, ref, lambda s: s.sort(key=lambda v: 0, reverse=True))  # stable: unchanged
    same(view, ref, lambda s: s.sort(lambda v: v))  # TypeError: key is keyword-only


@pytest.mark.parametrize("view_cls", LISTS)
def test_list_sort_stable_with_key(root_view, view_cls):
    data = ["bb", "a", "cc", "d", "ee"]
    view, ref = a_list(root_view, view_cls, data)
    same(view, ref, lambda s: s.sort(key=len))
    same(view, ref, lambda s: s.sort(key=len, reverse=True))


@pytest.mark.parametrize("view_cls", LISTS[:2])
def test_list_sort_refused_comparison_writes_nothing(root_view, view_cls):
    view, ref = a_list(root_view, view_cls, [1, {"x": 1}, None])
    with pytest.raises(TypeError):
        view.sort()
    assert plain(view) == ref


@pytest.mark.parametrize("view_cls", LISTS)
def test_list_concat_and_repeat(root_view, view_cls):
    view, ref = a_list(root_view, view_cls, NUMBERS)
    for other in ([], [9], NUMBERS):
        same(view, ref, lambda s, o=other: s + o)
        same(view, ref, lambda s, o=other: o + s)
    for refused in ((9,), "ab", 5, {9}):
        same(view, ref, lambda s, o=refused: s + o)  # TypeError: list + list only
        same(view, ref, lambda s, o=refused: o + s)
    assert view + view == ref + ref
    for n in (0, 1, 3, -2, True):
        same(view, ref, lambda s, n=n: s * n)
        same(view, ref, lambda s, n=n: n * s)
    for refused in (2.0, "2", None, [2]):
        same(view, ref, lambda s, n=refused: s * n)  # TypeError
        same(view, ref, lambda s, n=refused: n * s)
    assert type(view + []) is list  # noqa: RUF005
    assert type(view * 2) is list


@pytest.mark.parametrize("view_cls", LISTS)
def test_list_in_place_concat_and_repeat(root_view, view_cls):
    view, ref = a_list(root_view, view_cls, NUMBERS)

    def iadd(s, other):
        held = s
        s += other
        assert s is held

    def imul(s, n):
        held = s
        s *= n
        assert s is held

    same(view, ref, lambda s: iadd(s, (7, 8)))  # += takes any iterable, as list.extend
    same(view, ref, lambda s: iadd(s, iter([9])))
    same(view, ref, lambda s: iadd(s, 5))  # TypeError
    same(view, ref, lambda s: imul(s, 2))
    same(view, ref, lambda s: imul(s, 1))
    same(view, ref, lambda s: imul(s, "2"))  # TypeError
    same(view, ref, lambda s: imul(s, 0))
    same(view, ref, lambda s: imul(s, 3))


@pytest.mark.parametrize("view_cls", LISTS)
def test_list_ordering(root_view, view_cls):
    view, ref = a_list(root_view, view_cls, NUMBERS)
    others = [[], NUMBERS, NUMBERS[:-1], [*NUMBERS, 0], [4], [3, 1, 2, 1, 6]]
    for other in others:
        for op in ("__lt__", "__le__", "__gt__", "__ge__", "__eq__", "__ne__"):
            same(view, ref, lambda s, o=other, op=op: getattr(s, op)(o))
        same(view, ref, lambda s, o=other: o < s)
        same(view, ref, lambda s, o=other: o >= s)
    for refused in (tuple(NUMBERS), "x", 5):
        same(view, ref, lambda s, o=refused: s < o)  # TypeError
        same(view, ref, lambda s, o=refused: o > s)
        same(view, ref, lambda s, o=refused: s == o)  # False, not TypeError
    other = root_view.open_child("o", view_cls)
    other.store([3, 1])
    assert other < view
    assert view > other
    assert sorted([view, other], key=list) == [other, view]


@pytest.mark.parametrize("view_cls", LISTS)
@pytest.mark.parametrize(
    "index",
    [
        slice(None),
        slice(1, 3),
        slice(3, 1),
        slice(-2, None),
        slice(None, None, 2),
        slice(None, None, -1),
        slice(4, 0, -2),
        slice(10, 20),
    ],
)
def test_list_slice_writes(root_view, view_cls, index):
    view, ref = a_list(root_view, view_cls, NUMBERS)
    width = len(ref[index])
    for values in ([], [7], [7, 8, 9], list(range(width)), iter([4, 4]), (6,)):
        values = list(values)
        same(view, ref, lambda s, v=values: s.__setitem__(index, v))
    same(view, ref, lambda s: s.__setitem__(index, 5))  # TypeError: not iterable
    same(view, ref, lambda s: s.__delitem__(index))
    same(view, ref, lambda s: s.__setitem__(index, NUMBERS))


@pytest.mark.parametrize("view_cls", LISTS[:2])
def test_list_slice_writes_nested(root_view, view_cls):
    data = [1, {"x": [1]}, [2, 3], None, 1]
    view, ref = a_list(root_view, view_cls, data)
    same(view, ref, lambda s: s.__setitem__(slice(0, 2), [{"y": 1}, 5, [6]]))  # kinds may change
    same(view, ref, lambda s: s.__delitem__(slice(1, None, 2)))
    same(view, ref, lambda s: s.__setitem__(slice(None), s))  # s[:] = s
    same(view, ref, lambda s: s.__setitem__(slice(None, None, 2), [1]))  # ValueError: size


# =============================================================================
# TUPLE
# =============================================================================

TUPLE = (3, 1, (2, 0), None, 1)


@pytest.fixture
def a_tuple(root_view):
    view = root_view.open_child("t", TupleView)
    view.store(TUPLE)
    return view, TUPLE


def test_tuple_concat_and_repeat(a_tuple):
    view, ref = a_tuple
    for other in ((), (9,), TUPLE):
        same(view, ref, lambda s, o=other: s + o)
        same(view, ref, lambda s, o=other: o + s)
    for refused in ([9], "ab", 5):
        same(view, ref, lambda s, o=refused: s + o)  # TypeError: tuple + tuple only
        same(view, ref, lambda s, o=refused: o + s)
    for n in (0, 2, -1):
        same(view, ref, lambda s, n=n: s * n)
        same(view, ref, lambda s, n=n: n * s)
    same(view, ref, lambda s: s * 2.0)  # TypeError
    assert type(view + ()) is tuple  # noqa: RUF005
    assert view + view == ref + ref


def test_tuple_ordering(root_view):
    view = root_view.open_child("t", TupleView)
    view.store((3, 1, 2))
    ref = (3, 1, 2)
    for other in ((), ref, (3, 1), (3, 1, 2, 0), (4,), (3, 0, 9)):
        for op in ("__lt__", "__le__", "__gt__", "__ge__", "__eq__", "__ne__"):
            same(view, ref, lambda s, o=other, op=op: getattr(s, op)(o))
        same(view, ref, lambda s, o=other: o < s)
    for refused in ([3, 1, 2], "x"):
        same(view, ref, lambda s, o=refused: s < o)  # TypeError
        same(view, ref, lambda s, o=refused: s == o)  # False


def test_tuple_is_not_hashable(a_tuple):
    view, _ = a_tuple  # declined: the stored tuple can be replaced under the view
    with pytest.raises(TypeError):
        hash(view)


# =============================================================================
# SET AND FROZENSET
# =============================================================================

MEMBERS = {1, "two", (3, 4), None}
SETS = [
    pytest.param(SetView, set, id="set"),
    pytest.param(FrozenSetView, frozenset, id="frozenset"),
]
OTHERS = [set(), {1}, set(MEMBERS), {*MEMBERS, 5}, {1, 5}, {5}]


@pytest.fixture
def a_set(root_view, view_cls, kind):
    view = root_view.open_child("s", view_cls)
    view.store(MEMBERS)
    return view, kind(MEMBERS)


@pytest.mark.parametrize(("view_cls", "kind"), SETS)
def test_set_copy(a_set, kind):
    view, ref = a_set
    same(view, ref, lambda s: s.copy())
    assert type(view.copy()) is kind


@pytest.mark.parametrize(("view_cls", "kind"), SETS)
def test_set_named_methods(a_set, kind):
    view, ref = a_set
    for other in (*OTHERS, [1, 5], (None,), "two", iter([1, 7])):
        for name in ("union", "intersection", "difference", "symmetric_difference"):
            arg = list(other) if not isinstance(other, str | set) else other
            same(view, ref, lambda s, o=arg, n=name: getattr(s, n)(o))
            assert type(getattr(view, name)(arg)) is kind
        for name in ("issubset", "issuperset", "isdisjoint"):
            arg = list(other) if not isinstance(other, str | set) else other
            same(view, ref, lambda s, o=arg, n=name: getattr(s, n)(o))
            same(view, ref, lambda s, o=arg, n=name: getattr(s, n)(iter(o)))
    for name in ("union", "intersection", "difference"):
        same(view, ref, lambda s, n=name: getattr(s, n)())
        same(view, ref, lambda s, n=name: getattr(s, n)([1, 5], {None}, "xy"))
        same(view, ref, lambda s, n=name: getattr(s, n)([[1]]))  # TypeError: unhashable
    same(view, ref, lambda s: s.symmetric_difference())  # TypeError: one argument
    same(view, ref, lambda s: s.issubset(5))  # TypeError


@pytest.mark.parametrize(("view_cls", "kind"), SETS)
def test_set_operators_take_sets_only(a_set):
    view, ref = a_set
    for other in ([1], (1,), "x"):
        same(view, ref, lambda s, o=other: s | o)  # TypeError
        same(view, ref, lambda s, o=other: s & o)
        same(view, ref, lambda s, o=other: s - o)
        same(view, ref, lambda s, o=other: s ^ o)
        same(view, ref, lambda s, o=other: o & s)
        same(view, ref, lambda s, o=other: o - s)
        same(view, ref, lambda s, o=other: o ^ s)
    keys = dict.fromkeys([1, 9]).keys()  # a Set that is not a set: its own operators answer
    same(view, ref, lambda s: s | keys)
    same(view, ref, lambda s: keys - s)


@pytest.mark.parametrize(("view_cls", "kind"), SETS)
def test_set_reflected_operators_give_the_left_type(a_set):
    view, ref = a_set
    for left in ({1, 5}, frozenset({1, 5})):
        for op in ("__or__", "__and__", "__sub__", "__xor__"):
            got, want = getattr(left, op)(view), getattr(left, op)(ref)
            if got is NotImplemented:
                got = getattr(type(view), "__r" + op[2:])(view, left)
            assert type(got) is type(want), (left, op)
            assert got == want
        assert type(left | view) is type(left)


def test_set_in_place_methods(root_view):
    view = root_view.open_child("s", SetView)
    view.store(MEMBERS)
    ref = set(MEMBERS)
    same(view, ref, lambda s: s.update([5, 6], {7}, "ab"))
    same(view, ref, lambda s: s.update())
    same(view, ref, lambda s: s.update(s))
    same(view, ref, lambda s: s.update([[1]]))  # TypeError: unhashable
    same(view, ref, lambda s: s.intersection_update([1, 5, 6, "a", None], {1, 5, 6, None, "b"}))
    same(view, ref, lambda s: s.intersection_update())
    same(view, ref, lambda s: s.difference_update([5], {None}))
    same(view, ref, lambda s: s.symmetric_difference_update([1, 1, 8]))
    same(view, ref, lambda s: s.symmetric_difference_update(iter([8, 9])))
    same(view, ref, lambda s: s.difference_update(s))
    same(view, ref, lambda s: s.update([1, 2]))
    same(view, ref, lambda s: s.symmetric_difference_update(s))


def test_set_in_place_operators_take_sets_only(root_view):
    view = root_view.open_child("s", SetView)
    view.store(MEMBERS)
    ref = set(MEMBERS)

    def iop(s, op, other):
        held = s
        s = getattr(s, op)(other)
        if s is NotImplemented:
            raise TypeError(op)
        assert s is held

    for op in ("__ior__", "__iand__", "__isub__", "__ixor__"):
        same(view, ref, lambda s, op=op: iop(s, op, [1, 9]))  # TypeError, nothing written
        same(view, ref, lambda s, op=op: iop(s, op, {1, 9, None}))
        same(view, ref, lambda s, op=op: iop(s, op, frozenset({"two", 1})))
    with pytest.raises(TypeError):
        view |= [1]


@pytest.mark.parametrize(("view_cls", "kind"), SETS)
def test_set_has_no_frozen_hash(a_set):
    view, _ = a_set  # declined for a frozenset view: the members can be replaced under it
    with pytest.raises(TypeError):
        hash(view)


# =============================================================================
# BYTEARRAY
# =============================================================================

TEXT = bytearray(b"  Hello World\tfoo-bar baz\nnext Line  ")


@pytest.fixture
def a_bytearray(root_view):
    view = root_view.open_child("b", ByteArrayView)
    view.store(TEXT)
    return view, bytearray(TEXT)


BYTES_CALLS = [
    ("capitalize", ()),
    ("center", (60,)),
    ("center", (60, b"*")),
    ("center", (60, b"**")),  # TypeError
    ("count", (b"o",)),
    ("count", (b"o", 5, 20)),
    ("decode", ()),
    ("decode", ("ascii", "strict")),
    ("endswith", (b"  ",)),
    ("endswith", ((b"x", b"e  "),)),
    ("endswith", (b"Line", None, -2)),
    ("expandtabs", ()),
    ("expandtabs", (4,)),
    ("find", (b"o",)),
    ("find", (111, 6)),
    ("find", (b"zz",)),
    ("hex", ()),
    ("hex", (":",)),
    ("hex", ("-", 2)),
    ("index", (b"World",)),
    ("index", (b"zz",)),  # ValueError
    ("isalnum", ()),
    ("isalpha", ()),
    ("isascii", ()),
    ("isdigit", ()),
    ("islower", ()),
    ("isspace", ()),
    ("istitle", ()),
    ("isupper", ()),
    ("join", ([b"a", b"b"],)),
    ("join", (["a"],)),  # TypeError
    ("ljust", (50, b".")),
    ("lower", ()),
    ("lstrip", ()),
    ("lstrip", (b" H",)),
    ("partition", (b"o",)),
    ("partition", (b"zz",)),
    ("removeprefix", (b"  He",)),
    ("removesuffix", (b"e  ",)),
    ("replace", (b"o", b"0")),
    ("replace", (b"o", b"0", 1)),
    ("rfind", (b"o",)),
    ("rindex", (b"o", 0, 10)),
    ("rjust", (50,)),
    ("rpartition", (b"o",)),
    ("rsplit", ()),
    ("rsplit", (b" ", 2)),
    ("rstrip", ()),
    ("split", ()),
    ("split", (b"o",)),
    ("split", (None, 1)),
    ("split", (b"",)),  # ValueError: empty separator
    ("splitlines", ()),
    ("splitlines", (True,)),
    ("startswith", (b"  H",)),
    ("startswith", (b"Hello", 2)),
    ("strip", ()),
    ("strip", (b" e",)),
    ("swapcase", ()),
    ("title", ()),
    ("translate", (bytearray.maketrans(b"lo", b"01"),)),
    ("translate", (None, b"lo")),
    ("translate", (b"short",)),  # ValueError: table must be 256 long
    ("upper", ()),
    ("zfill", (50,)),
    ("copy", ()),
]


@pytest.mark.parametrize(("name", "args"), BYTES_CALLS, ids=lambda x: str(x))
def test_bytearray_methods(a_bytearray, name, args):
    view, ref = a_bytearray
    same(view, ref, lambda b: getattr(b, name)(*args))
    try:
        want = getattr(ref, name)(*args)
    except (TypeError, ValueError):
        return
    assert type(getattr(view, name)(*args)) is type(want)


def test_bytearray_operators(a_bytearray):
    view, ref = a_bytearray
    for other in (b"!", bytearray(b"?"), memoryview(b"m")):
        same(view, ref, lambda b, o=other: b + o)
    for left in (b"!", bytearray(b"?")):
        same(view, ref, lambda b, o=left: o + b)
        assert type(left + view) is type(left + ref)
    for refused in ("s", [1], 5):
        same(view, ref, lambda b, o=refused: b + o)  # TypeError
        same(view, ref, lambda b, o=refused: o + b)
    for n in (0, 2, -1):
        same(view, ref, lambda b, n=n: b * n)
        same(view, ref, lambda b, n=n: n * b)
    same(view, ref, lambda b: b * 1.5)  # TypeError
    for other in (b"", bytes(TEXT), b"  Hello", b"z", bytearray(b"  A"), memoryview(b"  Z")):
        for op in ("__lt__", "__le__", "__gt__", "__ge__", "__eq__", "__ne__"):
            same(view, ref, lambda b, o=other, op=op: getattr(b, op)(o))
    same(view, ref, lambda b: b < "x")  # TypeError
    same(view, ref, lambda b: b == "x")  # False
    assert view + view == ref + ref
    assert bytes(view) == bytes(ref)
    view.store(b"%d-%s")
    ref = bytearray(b"%d-%s")
    same(view, ref, lambda b: b % (1, b"x"))
    same(view, ref, lambda b: b % 1)  # TypeError: not enough arguments
    assert type(view % (1, b"x")) is bytearray
    assert b"<%b>" % view == b"<%b>" % ref
    assert ByteArrayView.maketrans(b"a", b"b") == bytearray.maketrans(b"a", b"b")


def test_bytearray_in_place(a_bytearray):
    view, ref = a_bytearray

    def iop(b, op, other):
        held = b
        b = getattr(b, op)(other)
        if b is NotImplemented:
            raise TypeError(op)
        assert b is held

    same(view, ref, lambda b: iop(b, "__iadd__", b"!!"))
    same(view, ref, lambda b: iop(b, "__iadd__", memoryview(b"m")))
    same(view, ref, lambda b: iop(b, "__iadd__", [1]))  # TypeError, nothing written
    same(view, ref, lambda b: iop(b, "__imul__", 2))
    same(view, ref, lambda b: iop(b, "__imul__", "2"))  # TypeError
    same(view, ref, lambda b: iop(b, "__imul__", 0))


@pytest.mark.parametrize(
    "index", [slice(None), slice(1, 3), slice(3, 1), slice(None, None, 2), slice(-4, None, -3)]
)
def test_bytearray_slice_writes(root_view, index):
    view = root_view.open_child("b", ByteArrayView)
    view.store(b"hello world")
    ref = bytearray(b"hello world")
    width = len(ref[index])
    for values in (b"", b"XY", bytearray(b"Z"), [65, 66], b"q" * width, memoryview(b"mm")):
        same(view, ref, lambda b, v=values: b.__setitem__(index, v))
    same(view, ref, lambda b: b.__setitem__(index, [300] * width))  # ValueError
    same(view, ref, lambda b: b.__setitem__(index, "s"))  # TypeError
    same(view, ref, lambda b: b.__delitem__(index))


def test_bytearray_has_no_buffer(a_bytearray):
    view, _ = a_bytearray  # declined: a buffer over a copy would drop writes made through it
    with pytest.raises(TypeError):
        memoryview(view)
    assert not hasattr(view, "fromhex")
