"""Random view ops against python models, on every backend.

Five views live in one storage: ``EagerDictView``, ``EagerLogIndexedDictView``,
``EagerFractionalIndexedDictView``, ``EagerListView`` and ``SetView``, each
mirrored on a python dict, insertion-ordered dict, ordered list of
``(id, value)``, list and set. Every rule runs inside the current transaction;
the transaction ends by commit (models promoted), abort or drop (models
reverted), and after each end a fresh snapshot is checked against the
committed models. After every op every view is checked: ``len``, forward and
reverse iteration order, ``keys``, ``items``, ``in`` and ``get`` for every id
in the alphabet, and every raw storage row under the view (child rows and
compound children, the log view's ``__keys__`` and ``__log__`` rows, the
fractional view's ``__order__`` and ``__pos__`` rows, the ``__len__``
metadata of list and set).

Pinned write rules are modelled: a kind change on set raises
``ContainerTypeError`` and changes nothing, ``update`` applies keys in order
up to the first kind change, pop of an absent key raises ``KeyError``, insert
of an existing id raises ``ValueError``.

Two tiers, see ``tests/README.md``: ``fast`` (default suite, fixed seed, a few
short examples) and ``slow`` (``-m slow``, about half a minute per backend).
``CONFORMANCE_MODEL_EXAMPLES`` and ``CONFORMANCE_MODEL_SEED`` drive the slow
tier.
"""

from __future__ import annotations

import gc
import os
import shutil
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import hypothesis.strategies as st
import pytest
from hypothesis import HealthCheck, Phase, seed, settings
from hypothesis.stateful import RuleBasedStateMachine, rule, run_state_machine_as_test
from tests.conformance.backends import BACKENDS, BackendSpec

from virtuals._views import (
    EagerDictView,
    EagerFractionalIndexedDictView,
    EagerListView,
    EagerLogIndexedDictView,
    SetView,
)
from virtuals._views.fractional_index import validate_order_key
from virtuals.container import ContainerTypeError
from virtuals.navigator import Navigator
from virtuals.tkv.filter import PrefixFilter
from virtuals.tkv.storage import StorageScanOptions
from virtuals.types import is_empty


# ---------------------------------------------------------------------------- tiers


@dataclass(frozen=True)
class Tier:
    name: str
    examples: int
    steps: int
    seed: int


FAST = Tier("fast", examples=10, steps=50, seed=424242)
SLOW = Tier(
    "slow",
    examples=int(os.environ.get("CONFORMANCE_MODEL_EXAMPLES", 150)),
    steps=80,
    seed=int(os.environ.get("CONFORMANCE_MODEL_SEED", 424242)),
)

TIERS = [
    pytest.param(FAST, id="fast", marks=pytest.mark.timeout(120)),
    pytest.param(SLOW, id="slow", marks=[pytest.mark.slow, pytest.mark.timeout(900)]),
]


def hypothesis_settings(tier: Tier) -> settings:
    return settings(
        max_examples=tier.examples,
        stateful_step_count=tier.steps,
        deadline=None,
        database=None,
        suppress_health_check=list(HealthCheck),
        phases=[Phase.generate, Phase.shrink],
        print_blob=True,
        report_multiple_bugs=False,  # one shrunk repro, well inside the case timeout
    )


@pytest.fixture(params=list(BACKENDS), ids=list(BACKENDS))
def backend(request) -> BackendSpec:
    return BACKENDS[request.param]


# ---------------------------------------------------------------------------- strategies and model helpers

IDS = ["a", "b", "c", "d", 1, 2]
ids_st = st.sampled_from(IDS)
prim_st = st.one_of(st.integers(-3, 3), st.sampled_from(["x", "y", ""]), st.none(), st.booleans())
compound_st = st.dictionaries(st.sampled_from(["p", "q"]), prim_st, max_size=2)
value_st = st.one_of(prim_st, prim_st, compound_st)
# Booleans are drawn on purpose: SetView keys members by type and value, so 1
# and True are two members (see test_true_and_1_are_distinct_set_members).
set_prim_st = st.one_of(st.integers(-3, 3), st.sampled_from(["x", "y"]), st.none(), st.booleans())
# Equal members built differently: frozenset([1, 9]) and frozenset([9, 1])
# iterate in different orders (1 and 9 collide in the 8 slot table), so only a
# canonical member key makes them one row. A tuple holding one does the same.
COLLIDING = [frozenset([1, 9]), frozenset([9, 1]), (0, frozenset([9, 1])), (0, frozenset([1, 9]))]
set_compound_st = st.one_of(
    st.tuples(st.integers(0, 2)), st.just(frozenset({1})), st.sampled_from(COLLIDING)
)
setval_st = st.one_of(
    set_prim_st, set_compound_st
)  # a tuple or frozenset is a member like any other
unhashable_st = st.sampled_from([[1], {"p": 1}, {1}])  # refused like a python set refuses them

MISSING = object()


def kind(v) -> str:
    return "c" if isinstance(v, dict) else "p"


def typed(v):
    """A value with its type, recursively, so 1, True and 1.0 compare as different."""
    if isinstance(v, dict):
        return ("dict", tuple(sorted((typed(k), typed(x)) for k, x in v.items())))
    if isinstance(v, tuple):
        return ("tuple", tuple(typed(x) for x in v))
    if isinstance(v, frozenset):
        return ("frozenset", frozenset(typed(x) for x in v))
    return (type(v).__name__, v)


def same(a, b) -> bool:
    return typed(a) == typed(b)


def same_seq(a, b) -> bool:
    return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=True))


# Set members are modelled by typed value: SetView keys a member by its pickle,
# so 1 and True are distinct members (python says equal). Pinned in
# test_true_and_1_are_distinct_set_members; listed as a proposal in the lane V
# progress file (True/1 key collision).
def member(v):
    return typed(v)


def expect(exc, fn, what: str) -> None:
    """``fn`` must raise ``exc``. An AssertionError (not a pytest outcome) keeps hypothesis shrinking."""
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"{what} did not raise {exc}")


def set_or_kind_error(setter, stored, v) -> bool:
    """Apply ``setter``; a kind change raises ContainerTypeError and changes nothing."""
    if stored is not MISSING and kind(stored) != kind(v):
        expect(ContainerTypeError, setter, "kind change")
        return False
    setter()
    return True


def update_model(model_get, apply_one, m: dict):
    """Mirror ``view.update(m)``: keys apply in order, the first kind change raises."""
    for k, v in m.items():
        stored = model_get(k)
        if stored is not MISSING and kind(stored) != kind(v):
            return k
        apply_one(k, v)
    return None


def rows_under(ctx, prefix: tuple) -> dict:
    opts = StorageScanOptions(start=prefix, break_filter=PrefixFilter(prefix=prefix))
    return dict(ctx.scan(opts).items())


def child_keys(rows: dict, prefix: tuple) -> dict:
    """Immediate children of ``prefix`` and, for compound ones, their children."""
    n = len(prefix)
    out: dict = {}
    for k in rows:
        if len(k) == n + 1:
            out.setdefault(k[n], set())
        elif len(k) == n + 2:
            out.setdefault(k[n], set()).add(k[n + 1])
        elif len(k) > n + 2:
            raise AssertionError(f"unexpected deep row {k}")
    return out


def expect_children(model: dict) -> dict:
    return {k: (set(v) if isinstance(v, dict) else set()) for k, v in model.items()}


# ---------------------------------------------------------------------------- machine


class ViewsMachine(RuleBasedStateMachine):
    """Five views on one storage against their models. Subclassed per backend and tier."""

    spec: BackendSpec
    scratch: Path
    ops: Counter

    def __init__(self) -> None:
        super().__init__()
        self.root = Path(tempfile.mkdtemp(prefix=f"vm-{self.spec.name}-", dir=self.scratch))
        self.storage = self.spec.make(self.root)
        self.storage.open()
        self.codec = self.storage.codec
        self.nav = Navigator(self.storage)
        # the model starts from five existing, empty containers: reads on a
        # never-written container raise, which the models do not track
        with self.storage.transaction() as tx:
            r = self.nav.root(tx)
            for name, cls in (
                ("d", EagerDictView),
                ("l", EagerLogIndexedDictView),
                ("f", EagerFractionalIndexedDictView),
                ("ls", EagerListView),
                ("s", SetView),
            ):
                r.ensure_child(name, cls)
        # committed models
        self.c_dict: dict = {}
        self.c_log: dict = {}  # insertion ordered
        self.c_frac: list = []  # [(id, value)] ordered
        self.c_list: list = []
        self.c_set: set = set()
        self.tx = None
        self.begin()

    def count(self, name: str) -> None:
        self.ops[name] += 1

    # ------------------------------------------------------------ txn lifecycle
    def begin(self) -> None:
        self.tx = self.storage.begin_transaction()
        r = self.nav.root(self.tx)
        self.d = r.open_child("d", EagerDictView)
        self.l = r.open_child("l", EagerLogIndexedDictView)
        self.f = r.open_child("f", EagerFractionalIndexedDictView)
        self.ls = r.open_child("ls", EagerListView)
        self.s = r.open_child("s", SetView)
        self.m_dict = dict(self.c_dict)
        self.m_log = dict(self.c_log)
        self.m_frac = list(self.c_frac)
        self.m_list = list(self.c_list)
        self.m_set = set(self.c_set)

    @rule(how=st.sampled_from(["commit", "commit", "abort", "drop"]))
    def end_txn(self, how):
        self.count("end_" + how)
        if how == "commit":
            self.tx.commit()
            self.c_dict, self.c_log = dict(self.m_dict), dict(self.m_log)
            self.c_frac, self.c_list = list(self.m_frac), list(self.m_list)
            self.c_set = set(self.m_set)
        elif how == "abort":
            self.tx.abort()
        else:
            self.tx = None
            self.d = self.l = self.f = self.ls = self.s = None
            gc.collect()
        # committed state as a fresh snapshot sees it
        with self.storage.snapshot() as snap:
            r = self.nav.root(snap)
            self.check_dict(snap, r.open_child("d", EagerDictView), self.c_dict)
            self.check_log(snap, r.open_child("l", EagerLogIndexedDictView), self.c_log)
            self.check_frac(snap, r.open_child("f", EagerFractionalIndexedDictView), self.c_frac)
            self.check_list(snap, r.open_child("ls", EagerListView), self.c_list)
            self.check_set(snap, r.open_child("s", SetView), self.c_set)
        self.begin()

    # ------------------------------------------------------------ checks
    def order_key(self, prefix: tuple):
        return lambda k: self.codec.encode_key((*prefix, k))

    def check_dict(self, ctx, v, model: dict) -> None:
        assert len(v) == len(model), (len(v), model)
        exp_order = sorted(model, key=self.order_key(("/", "d")))
        assert list(v) == exp_order, (list(v), exp_order)
        assert list(v.keys()) == exp_order
        assert [k for k, _ in v.items()] == exp_order
        for k in IDS:
            assert (k in v) == (k in model), k
            got = v.get(k)
            if k in model:
                assert same(got, model[k]), (k, got, model[k])
            else:
                assert got is None, (k, got)  # get of an absent key is None, as on a dict
        got_items = dict(v.items())
        assert list(got_items) == list(model) or set(got_items) == set(model)
        assert all(same(got_items[k], model[k]) for k in model), (got_items, model)
        rows = rows_under(ctx, ("/", "d"))
        if model or rows:
            assert ("/", "d") in rows or not model, "dict has children but no marker"
            rows.pop(("/", "d"), None)
            assert child_keys(rows, ("/", "d")) == expect_children(model), (
                child_keys(rows, ("/", "d")),
                expect_children(model),
            )

    def check_log(self, ctx, v, model: dict) -> None:
        assert len(v) == len(model), (len(v), model)
        assert list(v) == list(model), (list(v), list(model))
        assert list(reversed(v)) == list(model)[::-1]
        assert [k for k, _ in v.items()] == list(model)
        assert list(v.keys()) == list(model)
        for k in IDS:
            assert (k in v) == (k in model), k
            if k in model:
                assert same(v[k], model[k]), (k, v[k], model[k])
                assert same(v.get(k), model[k])
            else:
                assert v.get(k) is None, (k, v.get(k))  # get of an absent key is None, as on a dict
                expect(KeyError, lambda k=k: v[k], "getitem of an absent key")
        rows = rows_under(ctx, ("/", "l"))
        keys_rows = {k[3]: val for k, val in rows.items() if len(k) == 4 and k[2] == "__keys__"}
        log_rows = {k[3]: val for k, val in rows.items() if len(k) == 4 and k[2] == "__log__"}
        data = {k: val for k, val in rows.items() if len(k) >= 4 and k[2] == "__data__"}
        assert len(keys_rows) == len(model), (keys_rows, model)
        assert [keys_rows[lk] for lk in sorted(keys_rows)] == list(model), (keys_rows, list(model))
        assert set(log_rows) == set(model), (set(log_rows), set(model))
        for k, lk in log_rows.items():
            assert keys_rows.get(lk) == k, (k, lk, keys_rows.get(lk))
        assert child_keys(data, ("/", "l", "__data__")) == expect_children(model)

    def check_frac(self, ctx, v, model: list) -> None:
        ids = [i for i, _ in model]
        vals = dict(model)
        assert len(v) == len(model), (len(v), ids)
        assert list(v) == ids, (list(v), ids)
        assert list(reversed(v)) == ids[::-1]
        assert list(v.keys()) == ids
        assert [k for k, _ in v.items()] == ids
        assert same_seq([x for _, x in v.items()], [vals[i] for i in ids])
        if ids:
            assert (v.first(), v.last()) == (ids[0], ids[-1])
        else:
            for edge in (v.first, v.last):
                with pytest.raises(IndexError):
                    edge()
        for k in IDS:
            assert (k in v) == (k in vals), k
            if k in vals:
                assert same(v[k], vals[k]), (k, v[k], vals[k])
                assert same(v.get(k), vals[k])
                i = ids.index(k)
                assert v.next_after(k) == (ids[i + 1] if i + 1 < len(ids) else None)
                assert v.prev_before(k) == (ids[i - 1] if i > 0 else None)
            else:
                assert v.get(k) is None, (k, v.get(k))  # get of an absent key is None, as on a dict
                expect(KeyError, lambda k=k: v[k], "getitem of an absent id")
        pos = list(v.positions())
        okeys = [o for o, _ in pos]
        assert okeys == sorted(okeys) and len(set(okeys)) == len(okeys), okeys
        assert [i for _, i in pos] == ids
        for o in okeys:
            validate_order_key(o)
        rows = rows_under(ctx, ("/", "f"))
        order_rows = {k[3]: val for k, val in rows.items() if len(k) == 4 and k[2] == "__order__"}
        pos_rows = {k[3]: val for k, val in rows.items() if len(k) == 4 and k[2] == "__pos__"}
        data = {k: val for k, val in rows.items() if len(k) >= 4 and k[2] == "__data__"}
        assert order_rows == dict(pos), (order_rows, pos)
        assert pos_rows == {i: o for o, i in pos}, (pos_rows, pos)
        assert child_keys(data, ("/", "f", "__data__")) == expect_children(vals)

    def check_list(self, ctx, v, model: list) -> None:
        assert len(v) == len(model), (len(v), model)
        assert same_seq(list(v), model), (list(v), model)
        assert same_seq(list(reversed(v)), model[::-1])
        for i in range(len(model)):
            assert same(v[i], model[i]), (i, v[i], model[i])
        for x in [0, 1, "x", None, {"p": 1}]:
            assert (x in v) == (x in model), x
        rows = rows_under(ctx, ("/", "ls"))
        rows.pop(("/", "ls"), None)
        exp = {i: (set(x) if isinstance(x, dict) else set()) for i, x in enumerate(model)}
        assert child_keys(rows, ("/", "ls")) == exp, (child_keys(rows, ("/", "ls")), exp)
        meta = ctx.get(("/m", "ls", "__len__"))
        if is_empty(meta):
            assert len(model) == 0, "len meta absent with items"
        else:
            assert meta == len(model), (meta, len(model))

    def check_set(self, ctx, v, model: set) -> None:
        """``model`` holds typed members (see ``member``)."""
        assert len(v) == len(model), (len(v), model)
        got = list(v)
        assert len(got) == len(model) and {member(x) for x in got} == model, (got, model)
        for x in [0, 1, True, False, "x", (1,), frozenset({1}), *COLLIDING]:
            assert (x in v) == (member(x) in model), x
        # extract() builds a python set, where True and 1 (False and 0) collapse
        # into one member; the view keeps them distinct (pinned by
        # test_true_and_1_are_distinct_set_members), so compare under python's rule.
        extracted = v.extract()
        assert extracted == set(got), (extracted, got)
        rows = rows_under(ctx, ("/", "s"))
        rows.pop(("/", "s"), None)
        assert len(rows) == len(model), (rows, model)
        assert {member(x) for x in rows.values()} == model
        meta = ctx.get(("/m", "s", "__len__"))
        if is_empty(meta):
            assert len(model) == 0, "len meta absent with items"
        else:
            assert meta == len(model), (meta, len(model))

    def check_all(self) -> None:
        self.check_dict(self.tx, self.d, self.m_dict)
        self.check_log(self.tx, self.l, self.m_log)
        self.check_frac(self.tx, self.f, self.m_frac)
        self.check_list(self.tx, self.ls, self.m_list)
        self.check_set(self.tx, self.s, self.m_set)

    # ------------------------------------------------------------ dict rules
    @rule(k=ids_st, v=value_st)
    def dict_set(self, k, v):
        self.count("dict_set")
        if set_or_kind_error(lambda: self.d.__setitem__(k, v), self.m_dict.get(k, MISSING), v):
            self.m_dict[k] = v
        self.check_all()

    @rule(k=ids_st)
    def dict_del(self, k):
        self.count("dict_del")
        if k in self.m_dict:
            del self.d[k]
            del self.m_dict[k]
        else:
            expect(KeyError, lambda: self.d.__delitem__(k), "dict del of absent key")
        self.check_all()

    @rule(k=ids_st)
    def dict_pop(self, k):
        self.count("dict_pop")
        if k in self.m_dict:
            assert self.d.pop(k) == self.m_dict.pop(k)
        else:
            expect(KeyError, lambda: self.d.pop(k), "pop of absent key")
            assert self.d.pop(k, 7) == 7
        self.check_all()

    @rule(m=st.dictionaries(ids_st, value_st, max_size=3))
    def dict_update(self, m):
        self.count("dict_update")
        bad = update_model(lambda k: self.m_dict.get(k, MISSING), self.m_dict.__setitem__, m)
        if bad is None:
            self.d.update(m)
        else:
            expect(ContainerTypeError, lambda: self.d.update(m), "update with kind change")
        self.check_all()

    @rule()
    def dict_clear(self):
        self.count("dict_clear")
        self.d.clear()
        self.m_dict.clear()
        self.check_all()

    @rule(m=st.dictionaries(ids_st, value_st, max_size=3))
    def dict_store(self, m):
        self.count("dict_store")
        self.d.store(m)
        self.m_dict = dict(m)
        self.check_all()

    # ------------------------------------------------------------ log indexed dict rules
    @rule(k=ids_st, v=value_st)
    def log_set(self, k, v):
        self.count("log_set")
        if set_or_kind_error(lambda: self.l.__setitem__(k, v), self.m_log.get(k, MISSING), v):
            self.m_log[k] = v  # existing keeps its place
        self.check_all()

    @rule(k=ids_st)
    def log_del(self, k):
        self.count("log_del")
        if k in self.m_log:
            del self.l[k]
            del self.m_log[k]
        else:
            expect(KeyError, lambda: self.l.__delitem__(k), "log del of absent key")
        self.check_all()

    @rule(k=ids_st)
    def log_pop(self, k):
        self.count("log_pop")
        if k in self.m_log:
            assert self.l.pop(k) == self.m_log.pop(k)
        else:
            expect(KeyError, lambda: self.l.pop(k), "pop of absent key")
            assert self.l.pop(k, 7) == 7
        self.check_all()

    @rule(v=value_st)
    def log_append(self, v):
        self.count("log_append")
        k = self.l.append(v)
        assert k not in self.m_log
        self.m_log[k] = v
        self.check_all()

    @rule(m=st.dictionaries(ids_st, value_st, max_size=3))
    def log_update(self, m):
        self.count("log_update")
        bad = update_model(lambda k: self.m_log.get(k, MISSING), self.m_log.__setitem__, m)
        if bad is None:
            self.l.update(m)
        else:
            expect(ContainerTypeError, lambda: self.l.update(m), "update with kind change")
        self.check_all()

    @rule()
    def log_clear(self):
        self.count("log_clear")
        self.l.clear()
        self.m_log.clear()
        self.check_all()

    @rule(m=st.dictionaries(ids_st, value_st, max_size=3))
    def log_store(self, m):
        self.count("log_store")
        self.l.store(m)
        self.m_log = dict(m)
        self.check_all()

    # ------------------------------------------------------------ fractional rules
    def _fids(self) -> list:
        return [i for i, _ in self.m_frac]

    @rule(k=ids_st, v=value_st)
    def frac_set(self, k, v):
        self.count("frac_set")
        ids = self._fids()
        stored = dict(self.m_frac).get(k, MISSING)
        if set_or_kind_error(lambda: self.f.__setitem__(k, v), stored, v):
            if k in ids:
                self.m_frac[ids.index(k)] = (k, v)
            else:
                self.m_frac.append((k, v))
        self.check_all()

    @rule(k=ids_st, v=value_st, where=st.sampled_from(["append", "prepend"]))
    def frac_append_prepend(self, k, v, where):
        self.count("frac_" + where)
        fn = getattr(self.f, where)
        if k in self._fids():
            expect(ValueError, lambda: fn(k, v), f"{where} of existing id")
        else:
            fn(k, v)
            if where == "append":
                self.m_frac.append((k, v))
            else:
                self.m_frac.insert(0, (k, v))
        self.check_all()

    @rule(ref=ids_st, k=ids_st, v=value_st, before=st.booleans())
    def frac_insert(self, ref, k, v, before):
        self.count("frac_insert")
        ids = self._fids()
        fn = self.f.insert_before if before else self.f.insert_after
        if ref not in ids:
            expect(KeyError, lambda: fn(ref, k, v), "insert at an absent ref")  # checked first
        elif k in ids:
            expect(ValueError, lambda: fn(ref, k, v), "insert of an existing id")
        else:
            fn(ref, k, v)
            i = ids.index(ref)
            self.m_frac.insert(i if before else i + 1, (k, v))
        self.check_all()

    @rule(k=ids_st, ref=ids_st, before=st.booleans())
    def frac_move(self, k, ref, before):
        self.count("frac_move")
        ids = self._fids()
        fn = self.f.move_before if before else self.f.move_after
        if k not in ids or ref not in ids:
            expect(KeyError, lambda: fn(k, ref), "move with absent id, self move too")
        elif k == ref:
            fn(k, ref)  # a present id moved next to itself: a no-op, order unchanged
        else:
            fn(k, ref)
            item = self.m_frac.pop(ids.index(k))
            j = self._fids().index(ref)
            self.m_frac.insert(j if before else j + 1, item)
        self.check_all()

    @rule(k=ids_st, first=st.booleans())
    def frac_move_edge(self, k, first):
        self.count("frac_move_edge")
        ids = self._fids()
        fn = self.f.move_first if first else self.f.move_last
        if k not in ids:
            expect(KeyError, lambda: fn(k), "move_first/last of absent id")
        else:
            fn(k)
            item = self.m_frac.pop(ids.index(k))
            if first:
                self.m_frac.insert(0, item)
            else:
                self.m_frac.append(item)
        self.check_all()

    @rule(k=ids_st)
    def frac_del(self, k):
        self.count("frac_del")
        ids = self._fids()
        if k in ids:
            del self.f[k]
            self.m_frac.pop(ids.index(k))
        else:
            expect(KeyError, lambda: self.f.__delitem__(k), "frac del of absent id")
        self.check_all()

    @rule(k=ids_st)
    def frac_pop(self, k):
        self.count("frac_pop")
        ids = self._fids()
        if k in ids:
            assert self.f.pop(k) == self.m_frac.pop(ids.index(k))[1]
        else:
            expect(KeyError, lambda: self.f.pop(k), "pop of absent id")
            assert self.f.pop(k, 7) == 7
        self.check_all()

    @rule()
    def frac_clear(self):
        self.count("frac_clear")
        self.f.clear()
        self.m_frac.clear()
        self.check_all()

    @rule()
    def frac_respace(self):
        self.count("frac_respace")
        self.f.respace()
        self.check_all()

    @rule(m=st.dictionaries(ids_st, value_st, max_size=3))
    def frac_update(self, m):
        self.count("frac_update")

        def apply_one(k, v) -> None:
            ids = self._fids()
            if k in ids:
                self.m_frac[ids.index(k)] = (k, v)
            else:
                self.m_frac.append((k, v))

        bad = update_model(lambda k: dict(self.m_frac).get(k, MISSING), apply_one, m)
        if bad is None:
            self.f.update(m)
        else:
            expect(ContainerTypeError, lambda: self.f.update(m), "update with kind change")
        self.check_all()

    @rule(m=st.dictionaries(ids_st, value_st, max_size=3))
    def frac_store(self, m):
        self.count("frac_store")
        self.f.store(m)
        self.m_frac = list(m.items())
        self.check_all()

    @rule(n=st.integers(1, 40))
    def frac_insert_same_gap(self, n):
        """Insert n fresh ids at one gap: key growth and window respace must keep order."""
        self.count("frac_same_gap")
        ids = self._fids()
        if not ids:
            self.f.append("g0", 0)
            self.m_frac.append(("g0", 0))
            ids = ["g0"]
        ref = ids[0]
        base = sum(self.ops.values())
        for i in range(n):
            k = f"g{base}_{i}"
            self.f.insert_after(ref, k, i)
            j = self._fids().index(ref)
            self.m_frac.insert(j + 1, (k, i))
        self.check_all()
        # trim back to keep the model small: delete what we added
        for i in range(n):
            k = f"g{base}_{i}"
            del self.f[k]
            self.m_frac.pop(self._fids().index(k))
        self.check_all()

    # ------------------------------------------------------------ list rules
    @rule(v=value_st)
    def list_append(self, v):
        self.count("list_append")
        self.ls.append(v)
        self.m_list.append(v)
        self.check_all()

    @rule(i=st.integers(-4, 6), v=value_st)
    def list_insert(self, i, v):
        self.count("list_insert")
        self.ls.insert(i, v)  # a compound over a primitive slot included
        self.m_list.insert(i, v)
        self.check_all()

    @rule(i=st.integers(-4, 6), v=value_st)
    def list_set(self, i, v):
        self.count("list_set")
        n = len(self.m_list)
        if -n <= i < n:
            if set_or_kind_error(lambda: self.ls.__setitem__(i, v), self.m_list[i], v):
                self.m_list[i] = v
        else:
            expect(IndexError, lambda: self.ls.__setitem__(i, v), "list set out of range")
        self.check_all()

    @rule(i=st.integers(-4, 6))
    def list_del(self, i):
        self.count("list_del")
        n = len(self.m_list)
        if -n <= i < n:
            del self.ls[i]
            del self.m_list[i]
        else:
            expect(IndexError, lambda: self.ls.__delitem__(i), "list del out of range")
        self.check_all()

    @rule(i=st.integers(-4, 6))
    def list_pop(self, i):
        self.count("list_pop")
        n = len(self.m_list)
        if -n <= i < n:
            assert self.ls.pop(i) == self.m_list.pop(i)
        else:
            expect(IndexError, lambda: self.ls.pop(i), "list pop out of range")
        self.check_all()

    @rule(vs=st.lists(value_st, max_size=3))
    def list_extend(self, vs):
        self.count("list_extend")
        self.ls.extend(vs)
        self.m_list.extend(vs)
        self.check_all()

    @rule(v=prim_st)
    def list_remove(self, v):
        self.count("list_remove")
        if v in self.m_list:
            self.ls.remove(v)
            self.m_list.remove(v)
        else:
            expect(ValueError, lambda: self.ls.remove(v), "list remove of absent value")
        self.check_all()

    @rule()
    def list_clear(self):
        self.count("list_clear")
        self.ls.clear()
        self.m_list.clear()
        self.check_all()

    @rule(vs=st.lists(value_st, max_size=4))
    def list_store(self, vs):
        self.count("list_store")
        self.ls.store(vs)
        self.m_list = list(vs)
        self.check_all()

    # ------------------------------------------------------------ set rules
    @rule(v=setval_st)
    def set_add(self, v):
        self.count("set_add")
        self.s.add(v)
        self.m_set.add(member(v))
        self.check_all()

    @rule(i=st.integers(0, 1))
    def set_add_colliding_pair(self, i):
        """Both spellings of one unordered member land on one row."""
        self.count("set_add_colliding")
        a, b = COLLIDING[2 * i], COLLIDING[2 * i + 1]
        self.s.add(a)
        assert b in self.s, (a, b)
        self.s.add(b)
        self.m_set.add(member(a))
        assert member(a) == member(b)
        self.check_all()

    @rule(v=unhashable_st)
    def set_add_unhashable_is_refused(self, v):
        self.count("set_add_unhashable")
        expect(TypeError, lambda: self.s.add(v), "add of an unhashable member")
        self.check_all()

    @rule(v=setval_st)
    def set_remove(self, v):
        self.count("set_remove")
        if member(v) in self.m_set:
            self.s.remove(v)
            self.m_set.remove(member(v))
        else:
            expect(KeyError, lambda: self.s.remove(v), "set remove of absent value")
        self.check_all()

    @rule(v=setval_st)
    def set_discard(self, v):
        self.count("set_discard")
        self.s.discard(v)
        self.m_set.discard(member(v))
        self.check_all()

    @rule()
    def set_clear(self):
        self.count("set_clear")
        self.s.clear()
        self.m_set.clear()
        self.check_all()

    @rule(vs=st.sets(setval_st, max_size=4))
    def set_store(self, vs):
        self.count("set_store")
        self.s.store(vs)
        self.m_set = {member(x) for x in vs}
        self.check_all()

    def teardown(self) -> None:
        try:
            if self.tx is not None and not self.tx.is_closed:
                self.tx.abort()
            self.storage.close()
        finally:
            shutil.rmtree(self.root, ignore_errors=True)


# ---------------------------------------------------------------------------- cases


def test_true_and_1_are_distinct_set_members(backend: BackendSpec, tmp_path):
    """Pins today's key collision behaviour, listed as a proposal for the owner.

    SetView keys a member by its pickle, so ``1`` and ``True`` are two members
    and ``1.0`` is a third; a python set holds one. DictView keys through the
    codec, where ``1`` and ``True`` are one key, like a python dict. See the
    True/1 key collision proposal in the lane V progress file.
    """
    st_ = backend.make(tmp_path)
    st_.open()
    try:
        with st_.transaction() as tx:
            r = Navigator(st_).root(tx)
            s = r.open_child("s", SetView)
            s.add(1)
            assert 1 in s and True not in s and 1.0 not in s
            s.add(True)
            s.add(1.0)
            assert len(s) == 3 and {member(x) for x in s} == {member(1), member(True), member(1.0)}
            d = r.open_child("d", EagerDictView)
            d[1] = "one"
            d[True] = "true"
            assert len(d) == 1 and d[1] == "true" and d[True] == "true"
    finally:
        st_.close()


@pytest.mark.parametrize("tier", TIERS)
def test_views_model(backend: BackendSpec, tier: Tier, tmp_path):
    """Random ops on five views, every view and every raw row checked against the models."""
    ops: Counter = Counter()
    cls = type(
        f"ViewsMachine_{backend.name}_{tier.name}",
        (ViewsMachine,),
        {"spec": backend, "scratch": tmp_path, "ops": ops},
    )
    t0 = time.monotonic()
    try:
        run_state_machine_as_test(seed(tier.seed)(cls), settings=hypothesis_settings(tier))
    finally:
        dt = time.monotonic() - t0
        print(
            f"[views model {backend.name} {tier.name}] seed={tier.seed} examples={tier.examples} "
            f"steps={tier.steps} ops={sum(ops.values())} time={dt:.1f}s"
        )
        print("  " + ", ".join(f"{k}={v}" for k, v in sorted(ops.items())))
