"""Server-side routing: the nu_notify_v1 Lua library and its loader.

Registration goes straight through `nu1_reg` (the observer is migrated in a
later step). Runs against `redis_url` (function mode) and, when the system
redis-server 6 binary exists, a private redis 6 (EVALSHA mode).
"""

from __future__ import annotations

import random
import time
import uuid

import pytest

from virtuals._backends.notify.keyenc import enc_key, hint_entries
from virtuals._backends.notify.router import NotifyLua
from virtuals.tkv.filter import (
    WILDCARD,
    And,
    Filter,
    LengthFilter,
    Or,
    PassAll,
    PassNone,
    PrefixFilter,
    SuffixFilter,
    WildcardFilter,
    canonicalize,
    filter_hash,
)


pytestmark = pytest.mark.redis

redis = pytest.importorskip("redis")
msgpack = pytest.importorskip("msgpack")

IID = 2**60 + 7  # above 2**53 on purpose
HDR = b"\x82" + msgpack.packb("iid") + msgpack.packb(IID) + msgpack.packb("keys")


# =============================================================================
# Fixtures and helpers
# =============================================================================


@pytest.fixture(scope="module")
def redis6():
    from tests.support.valkey import private_redis6

    server = private_redis6()
    if server is None:
        pytest.skip("system redis-server not found")
    yield server.url
    server.close()


@pytest.fixture(params=["valkey", "redis6"])
def url(request, redis_url):
    if request.param == "valkey":
        return redis_url
    return request.getfixturevalue("redis6")


class Rig:
    """One connection + loader + prefix, with helpers to register and route."""

    def __init__(self, url: str) -> None:
        self.conn = redis.from_url(url)
        self.lua = NotifyLua()
        self.lua.ensure(self.conn)
        self.P = f"vtest:{uuid.uuid4().hex[:8]}"
        self.ps = self.conn.pubsub()

    def reg(self, f: Filter, subscribe: bool = True) -> str:
        h = filter_hash(f)
        entries = hint_entries(canonicalize(f).routing_hints())
        if subscribe:
            self.sub(h)
        if entries:
            args = [self.P, h] + [x for e in entries for x in e]
            self.lua.call(self.conn, "nu1_reg", args)
        return h

    def sub(self, h: str) -> None:
        ch = f"{self.P}:notif:{h}"
        self.ps.subscribe(ch)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            m = self.ps.get_message(timeout=0.1)
            if m and m["type"] == "subscribe" and m["channel"].decode() == ch:
                return
        raise AssertionError(f"no subscribe confirmation for {ch}")

    def route(self, keys: list[tuple]) -> int:
        args: list = [self.P, HDR]
        for k in keys:
            segs = enc_key(k)
            args += [msgpack.packb(list(k), use_bin_type=True), b"%d" % len(segs), *segs]
        return self.lua.call(self.conn, "nu1_route", args)

    def collect(self, n: int, timeout: float = 3.0) -> list[tuple[str, bytes]]:
        """Read exactly n published messages: (hash, raw payload)."""
        out: list[tuple[str, bytes]] = []
        pre = f"{self.P}:notif:"
        deadline = time.monotonic() + timeout
        while len(out) < n and time.monotonic() < deadline:
            m = self.ps.get_message(timeout=0.1)
            if m and m["type"] == "message":
                out.append((m["channel"].decode()[len(pre) :], m["data"]))
        assert len(out) == n, f"expected {n} messages, got {len(out)}"
        assert self.ps.get_message(timeout=0.05) is None
        return out

    def delivered(self, keys: list[tuple]) -> dict[str, list]:
        sent = self.route(keys)
        res: dict[str, list] = {}
        for h, data in self.collect(sent):
            p = msgpack.unpackb(data, raw=False)
            assert p["iid"] == IID
            res.setdefault(h, []).extend(tuple(k) for k in p["keys"])
        return res

    def index_keys(self) -> list[bytes]:
        return sorted(self.conn.keys(f"{self.P}:nx1:*"))

    def close(self) -> None:
        for k in self.conn.keys(f"{self.P}:*"):
            self.conn.delete(k)
        self.ps.close()
        self.conn.close()


@pytest.fixture
def rig(url):
    r = Rig(url)
    yield r
    r.close()


@pytest.fixture
def vrig(redis_url):
    """Function mode only."""
    r = Rig(redis_url)
    yield r
    r.close()


# =============================================================================
# Loader
# =============================================================================


def test_loader_mode(rig, url, redis_url):
    assert rig.lua.mode == ("function" if url == redis_url else "script")


def test_reg_idempotent(rig):
    f = PrefixFilter(("users",))
    h = filter_hash(f)
    args = [rig.P, h] + [x for e in hint_entries(f.routing_hints()) for x in e]
    assert rig.lua.call(rig.conn, "nu1_reg", args) == 1
    assert rig.lua.call(rig.conn, "nu1_reg", args) == 0
    assert rig.conn.hget(f"{rig.P}:nx1:pc", "1") == b"1"
    assert rig.conn.smembers(f"{rig.P}:nx1:p:1:s5:users") == {h.encode()}


def test_reload_after_function_flush(vrig):
    h = vrig.reg(PrefixFilter(("a",)))
    vrig.conn.function_flush()
    assert vrig.delivered([("a", 1)]) == {h: [("a", 1)]}


def test_reload_after_script_flush(redis6):
    r = Rig(redis6)
    try:
        assert r.lua.mode == "script"
        h = r.reg(PrefixFilter(("a",)))
        r.conn.script_flush()
        assert r.delivered([("a", 1)]) == {h: [("a", 1)]}
    finally:
        r.close()


def test_unknown_function_raises(rig):
    with pytest.raises(redis.ResponseError):
        rig.lua.call(rig.conn, "nu1_nope", [rig.P])


# =============================================================================
# Routing per hint kind
# =============================================================================


def test_route_per_kind(rig):
    hp = rig.reg(PrefixFilter(("users",)))
    hs = rig.reg(SuffixFilter(("name",)))
    hl = rig.reg(LengthFilter(3))
    hw = rig.reg(WildcardFilter(("users", WILDCARD, "age")))
    ha = rig.reg(PassAll())
    hn = rig.reg(PassNone())
    keys = [("users", "bob", "age"), ("users", "bob", "name"), ("posts", 1), ("x", "y", "z")]
    got = rig.delivered(keys)
    assert got[hp] == keys[:2]
    assert got[hs] == [keys[1]]
    assert got[hl] == [keys[0], keys[1], keys[3]]
    assert got[hw] == [keys[0]]
    assert got[ha] == keys
    assert hn not in got
    assert set(got) == {hp, hs, hl, hw, ha}


def test_route_and_or_supersets(rig):
    h_and = rig.reg(And((PrefixFilter(("users",)), LengthFilter(3))))
    h_or = rig.reg(Or((PrefixFilter(("a",)), SuffixFilter(("z",)))))
    got = rig.delivered([("users", 1), ("users", 1, 2), ("a", 9), ("q", "z"), ("q",)])
    # And routes by its most selective child: a superset, the observer drops extras.
    assert got[h_and] == [("users", 1), ("users", 1, 2)]
    assert got[h_or] == [("a", 9), ("q", "z")]


def test_key_routed_once_per_hash(rig):
    h = rig.reg(Or((PrefixFilter(("a",)), LengthFilter(2), PassAll())))
    h2 = rig.reg(Or((PrefixFilter(("a",)), LengthFilter(2))))
    assert rig.delivered([("a", 1)]) == {h: [("a", 1)], h2: [("a", 1)]}


def test_no_interest_publishes_nothing(rig):
    assert rig.route([("a",), ("b", 1)]) == 0


def test_payload_bytes_match_todays_format(rig):
    h = rig.reg(PassAll())
    keys = [("users", "alice"), ("n", 2**63), ("neg", -5), ("ü", "日本")]
    rig.route(keys)
    [(got_h, data)] = rig.collect(1)
    assert got_h == h
    today = msgpack.packb({"iid": IID, "keys": [list(k) for k in keys]}, use_bin_type=True)
    assert data == today


@pytest.mark.parametrize("n", [0, 1, 15, 16, 65535, 65536])
def test_array_header_sizes(vrig, n):
    vrig.reg(PassAll())
    keys = [(i,) for i in range(n)] if n else [()]  # n == 0: one empty key
    vrig.route(keys)
    [(_, data)] = vrig.collect(1, timeout=10)
    want = msgpack.packb({"iid": IID, "keys": [list(k) for k in keys]}, use_bin_type=True)
    assert data == want


# =============================================================================
# Edge keys: ints above 2**53, negatives, unicode, int vs str
# =============================================================================


def test_big_ints_exact(rig):
    h = rig.reg(PrefixFilter((2**60,)))
    # msgpack (today's payload) tops out at 2**64 - 1.
    h2 = rig.reg(WildcardFilter((WILDCARD, 2**64 - 1)))
    # 2**60 + 1 is the same double as 2**60: a Lua number would confuse them.
    got = rig.delivered([(2**60,), (2**60 + 1,), (2**60, "x"), ("k", 2**64 - 1), ("k", 2**64 - 2)])
    assert got[h] == [(2**60,), (2**60, "x")]
    assert got[h2] == [("k", 2**64 - 1)]


def test_negative_ints(rig):
    h = rig.reg(SuffixFilter((-1,)))
    got = rig.delivered([("a", -1), ("a", 1), ("a", "-1"), (-(2**63), -1)])
    assert got == {h: [("a", -1), (-(2**63), -1)]}


def test_int_vs_str(rig):
    hi = rig.reg(PrefixFilter((1,)))
    hs = rig.reg(PrefixFilter(("1",)))
    got = rig.delivered([(1, "a"), ("1", "a")])
    assert got == {hi: [(1, "a")], hs: [("1", "a")]}


def test_bool_routes_like_int(rig):
    h = rig.reg(PrefixFilter((1,)))
    got = rig.delivered([(True,), (False,)])
    assert got == {h: [(True,)]}


def test_unicode(rig):
    h = rig.reg(WildcardFilter(("日本", WILDCARD, "ü")))
    h2 = rig.reg(PrefixFilter(("😀",)))
    got = rig.delivered([("日本", 1, "ü"), ("日本", 1, "u"), ("😀", "x"), ("😁",)])
    assert got == {h: [("日本", 1, "ü")], h2: [("😀", "x")]}


def test_segment_boundaries_not_confused(rig):
    # concat("ab", "c") must not equal concat("a", "bc")
    h = rig.reg(PrefixFilter(("ab", "c")))
    got = rig.delivered([("a", "bc"), ("ab", "c"), ("abc",)])
    assert got == {h: [("ab", "c")]}


# =============================================================================
# Lazy removal
# =============================================================================


def test_lazy_removal_leaves_no_index_keys(rig):
    filters = [
        PrefixFilter(("a", 1)),
        SuffixFilter(("z",)),
        LengthFilter(2),
        WildcardFilter(("a", WILDCARD)),
        PassAll(),
        Or((PrefixFilter(("b",)), WildcardFilter((WILDCARD, "z")))),
    ]
    for f in filters:
        rig.reg(f, subscribe=False)
    assert rig.index_keys()
    assert rig.route([("a", 1), ("b", "z"), ("q", "z")]) == 0
    assert rig.index_keys() == []


def test_lazy_removal_keeps_live_and_shared_counters(rig):
    live = rig.reg(PrefixFilter(("a",)))
    rig.reg(PrefixFilter(("b",)), subscribe=False)  # same depth counter
    assert rig.conn.hget(f"{rig.P}:nx1:pc", "1") == b"2"
    assert rig.delivered([("a",), ("b",)]) == {live: [("a",)]}
    assert rig.conn.hget(f"{rig.P}:nx1:pc", "1") == b"1"
    assert rig.delivered([("a",), ("b",)]) == {live: [("a",)]}


def test_reregister_after_lazy_removal(rig):
    f = PrefixFilter(("a",))
    rig.reg(f, subscribe=False)
    rig.route([("a",)])
    h = rig.reg(f)
    assert rig.delivered([("a",)]) == {h: [("a",)]}


# =============================================================================
# Server fuzz: delivered is a superset of Filter.matches
# =============================================================================

HOT = ["a", 1, True, -1, 2**60, "ü", "", "*"]
COLD = ["b", 0, 2, 2**53 + 1, -(2**63), 2**64 - 1, "日本", "1"]


def _seg(r):
    return r.choice(HOT if r.random() < 0.7 else COLD)


def _key(r):
    return tuple(_seg(r) for _ in range(r.randint(0, 4)))


def _filter(r, depth=0):
    if depth < 2 and r.random() < 0.3:
        kids = tuple(_filter(r, depth + 1) for _ in range(r.randint(0, 3)))
        return And(kids) if r.random() < 0.5 else Or(kids)
    leaf = r.randrange(6)
    if leaf == 0:
        return PrefixFilter(tuple(_seg(r) for _ in range(r.randint(0, 3))))
    if leaf == 1:
        return SuffixFilter(tuple(_seg(r) for _ in range(r.randint(0, 2))))
    if leaf == 2:
        return LengthFilter(r.randint(-1, 4))
    if leaf == 3:
        return WildcardFilter(
            tuple(WILDCARD if r.random() < 0.5 else _seg(r) for _ in range(r.randint(0, 4)))
        )
    return PassAll() if leaf == 4 else PassNone()


@pytest.mark.parametrize("seed", range(3))
def test_fuzz_server_superset(rig, seed):
    r = random.Random(seed)  # noqa: S311
    filters: dict[str, Filter] = {}
    for _ in range(80):
        f = _filter(r)
        filters.setdefault(filter_hash(f), f)
    for f in filters.values():
        rig.reg(f)
    keys = [_key(r) for _ in range(400)]
    got = rig.delivered(keys)
    matched = selective = 0
    for h, f in filters.items():
        want = [k for k in keys if f.matches(k)]
        have = got.get(h, [])
        matched += len(want)
        selective += len(want) * (
            hint_entries(canonicalize(f).routing_hints()) != [(b"a", b"", b"")]
        )
        packed_have = {msgpack.packb(list(k)) for k in have}
        for k in want:
            assert msgpack.packb(list(k)) in packed_have, (f, k)
    assert matched > 500
    assert selective > 300  # enough matches go through non-all hints
