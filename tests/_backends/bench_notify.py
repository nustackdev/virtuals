"""Bench: redis notify publish cost, server-side routing vs the old client-side cache.

Plain script (not collected by pytest). Runs against whichever virtuals is
importable, so the same file measures the old code (a worktree of the base
commit with its own venv) and the new one:

    .venv/bin/python tests/_backends/bench_notify.py --url unix:///tmp/x/vk.sock
    ../virtuals-base/.venv/bin/python tests/_backends/bench_notify.py --url ...

Without --url it starts a private valkeylite server (dev dependency).

For F registered filters (mixed prefixes, wildcards, lengths, suffixes,
And/Or) and B keys per batch it reports, per batch:
- pub: publisher time, mean / p99 us, synchronous on the calling thread
  (new: `_publish_batch` = one nu1_route call; old: client-side match +
  PUBLISH pipeline, with the registry cache populated from HGETALL first).
- vk: valkey time per batch from commandstats (new: fcall or evalsha
  usec_per_call; old: total PUBLISH usec / batches, and publishes per batch).
- obs: one observer's local SubscriptionRegistry.match over the batch keys.

A drain connection SUBSCRIBEs every channel, so PUBLISH has a receiver and
the new router's lazy removal never fires.
"""

from __future__ import annotations

import argparse
import json
import queue
import random
import tempfile
import threading
import time
from pathlib import Path

import redis

from virtuals._backends.publishers.redis_pubsub import RedisPublisher
from virtuals.tkv.filter.filter import (
    And,
    LengthFilter,
    Or,
    PrefixFilter,
    SuffixFilter,
    WildcardFilter,
    filter_hash,
)
from virtuals.tkv.observer.registry import SubscriptionRegistry
from virtuals.tkv.observer.subscription import Subscription
from virtuals.tkv.observer.types import SubscriptionOptions


try:
    from virtuals._backends.notify.keyenc import hint_entries
    from virtuals._backends.notify.router import NotifyLua
    from virtuals.tkv.filter import canonicalize

    IMPL = "new"
except ImportError:
    IMPL = "old"


def make_filters(n: int) -> list:
    fs = []
    for i in range(n):
        k = i % 20
        if k < 10:
            fs.append(PrefixFilter(prefix=("app", i)))
        elif k == 10:
            fs.append(PrefixFilter(prefix=("app", i, f"f{i % 20}")))
        elif k == 11:
            fs.append(PrefixFilter(prefix=("app", i, "f0", "x1")))
        elif k == 12:
            fs.append(WildcardFilter(pattern=("app", "*", f"f{i % 20}")))
        elif k == 13:
            fs.append(WildcardFilter(pattern=("*", i, "*")))
        elif k == 14:
            fs.append(SuffixFilter(suffix=(f"x{i % 50}",)))
        elif k == 15:
            fs.append(And(filters=(PrefixFilter(prefix=("app", i)), LengthFilter(length=3))))
        elif k == 16:
            fs.append(
                Or(
                    filters=(
                        PrefixFilter(prefix=("app", i, "f0")),
                        PrefixFilter(prefix=("other", i)),
                    )
                )
            )
        elif k == 17:
            fs.append(LengthFilter(length=5 + i % 4))
        elif k == 18:
            fs.append(
                Or(
                    filters=(
                        WildcardFilter(pattern=("app", i, "*")),
                        SuffixFilter(suffix=("tail", i)),
                    )
                )
            )
        else:
            fs.append(
                And(
                    filters=(
                        WildcardFilter(pattern=("app", "*", "*", f"x{i % 7}")),
                        PrefixFilter(prefix=("app", i)),
                    )
                )
            )
    return fs


def make_key(rnd: random.Random, n: int) -> tuple:
    r = rnd.random()
    a = rnd.randrange(max(n, 10))
    if r < 0.8:
        return ("app", a, f"f{rnd.randrange(20)}")
    if r < 0.95:
        return ("app", a, f"f{rnd.randrange(20)}", f"x{rnd.randrange(50)}")
    return ("other", a, "tail", a)


def stats(xs: list[float]) -> tuple[float, float]:
    xs = sorted(xs)
    return sum(xs) / len(xs) * 1e6, xs[int(len(xs) * 0.99)] * 1e6


class Drain:
    """SUBSCRIBE every channel and consume messages on a thread."""

    def __init__(self, url: str, channels: list[str]) -> None:
        self.ps = redis.from_url(url).pubsub()
        self.ps.subscribe(*channels)
        got = 0
        while got < len(channels):
            m = self.ps.get_message(timeout=1.0)
            if m and m["type"] == "subscribe":
                got += 1
        self.n = 0
        self.stop = threading.Event()
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _run(self) -> None:
        while not self.stop.is_set():
            if self.ps.get_message(timeout=0.05):
                self.n += 1

    def close(self) -> None:
        self.stop.set()
        self.th.join()
        self.ps.close()


def setup_new(url: str, prefix: str, hs: dict) -> tuple:
    conn = redis.from_url(url)
    lua = NotifyLua()
    lua.ensure(conn)
    for h, f in hs.items():
        args: list = [prefix, h]
        for e in hint_entries(canonicalize(f).routing_hints()):
            args.extend(e)
        lua.call(conn, "nu1_reg", args)
    pub = RedisPublisher(redis_url=url, channel_prefix=prefix)
    pub._on_connect()

    def run(batch: list) -> None:
        pub._publish_batch(batch)

    cmd = "cmdstat_fcall" if pub._lua.mode == "function" else "cmdstat_evalsha"
    return run, cmd, pub._on_disconnect


def setup_old(url: str, prefix: str, hs: dict) -> tuple:
    conn = redis.from_url(url)
    for h, f in hs.items():
        conn.hset(f"{prefix}:subs:registry", h, json.dumps(f.to_dict(), separators=(",", ":")))
    pub = RedisPublisher(redis_url=url, channel_prefix=prefix)
    pub._redis_pub = redis.from_url(url)
    pub._refresh_remote_registry()
    assert len(pub._remote_registry) == len(hs)
    pub._publish_queue = queue.Queue()

    def run(batch: list) -> None:
        pub._publish_batch(batch)
        items = []
        while not pub._publish_queue.empty():
            items.append(pub._publish_queue.get_nowait())
        pub._flush_pipeline(items)

    return run, "cmdstat_publish", lambda: pub._redis_pub.close()


def bench(url: str, fs_list: list[int], bs_list: list[int], iters: int) -> list[dict]:
    admin = redis.from_url(url)
    rows = []
    for n in fs_list:
        prefix = f"bn{IMPL}{n}"
        hs = {filter_hash(f): f for f in make_filters(n)}
        drain = Drain(url, [f"{prefix}:notif:{h}" for h in hs])
        run, cmd, close = (setup_new if IMPL == "new" else setup_old)(url, prefix, hs)
        reg = SubscriptionRegistry()
        for f in hs.values():
            reg.add(Subscription(_options=SubscriptionOptions(f), _observer=None))
        rnd = random.Random(1)  # noqa: S311
        for b in bs_list:
            it = iters if b < 1000 else max(iters // 6, 20)
            batches = [[make_key(rnd, n) for _ in range(b)] for _ in range(it)]
            for batch in batches[:10]:  # warm up
                run(batch)
            admin.config_resetstat()
            t_pub, t_obs = [], []
            for batch in batches:
                t0 = time.perf_counter()
                run(batch)
                t_pub.append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                for k in batch:
                    reg.match(k)
                t_obs.append(time.perf_counter() - t0)
            st = admin.info("commandstats").get(cmd, {"calls": 0, "usec": 0})
            pm, pp = stats(t_pub)
            om, _ = stats(t_obs)
            row = {
                "impl": IMPL,
                "F": n,
                "hashes": len(hs),
                "B": b,
                "iters": it,
                "pub_mean": pm,
                "pub_p99": pp,
                "vk_us_batch": st["usec"] / it,
                "vk_calls_batch": st["calls"] / it,
                "obs_mean": om,
            }
            rows.append(row)
            print(
                f"{IMPL} F={n:4d} ({len(hs):4d} h) B={b:4d} | pub {pm:8.0f} / {pp:8.0f} us"
                f" | vk {row['vk_us_batch']:7.1f} us/batch"
                f" ({row['vk_calls_batch']:5.1f} {cmd[8:]}/batch) | obs {om:6.0f} us",
                flush=True,
            )
        close()
        time.sleep(0.2)
        print(f"  drained messages: {drain.n}", flush=True)
        drain.close()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="valkey/redis url; default starts a private valkeylite")
    ap.add_argument("--filters", default="1,10,100,1000")
    ap.add_argument("--batches", default="1,10,100")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--json", help="write rows to this file")
    a = ap.parse_args()
    srv = None
    url = a.url
    if url is None:
        from valkeylite import ValkeyServer

        d = tempfile.mkdtemp(prefix="vkb-", dir="/tmp")
        sock = f"{d}/vk.sock"
        srv = ValkeyServer(port=0, data_dir=d, unix_socket_path=sock, save="", appendonly="no")
        srv.start()
        url = f"unix://{sock}"
    try:
        rows = bench(
            url,
            [int(x) for x in a.filters.split(",")],
            [int(x) for x in a.batches.split(",")],
            a.iters,
        )
    finally:
        if srv is not None:
            srv.stop()
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
