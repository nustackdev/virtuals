"""Kh57View - sparse int-keyed map with kh57 range reservoir sampling."""

import random

from virtuals._views import EagerKh57View
from virtuals.codecs import NoOpCodec
from virtuals.navigator import Navigator
from virtuals.storages.mem import InMemoryStorage


N = 100_000
SAMPLE_N = 500
RANGE_BEGIN = 30_000
RANGE_END = 70_000

nav = Navigator(InMemoryStorage(codec=NoOpCodec()))

with nav.storage as storage, storage.transaction() as tx:
    root = nav.root(tx)
    events = root.open_child("events", EagerKh57View)

    # Sparse: put N items with non-contiguous int keys
    for i in range(N):
        events[i * 3] = f"event-{i}"

    print("len:", len(events))

    # Iteration yields keys in original int order
    first_10 = []
    for k in events:
        first_10.append(k)
        if len(first_10) >= 10:
            break
    print("first 10 keys in original order:", first_10)

    # Sample 500 items from a range
    rng = random.Random(0)  # noqa: S311
    picks = events.sample(SAMPLE_N, begin=RANGE_BEGIN, end=RANGE_END, rng=rng)
    print("sampled:", len(picks))
    sorted_keys = sorted(k for k, _v in picks)
    print("first 10 sampled keys (sorted):", sorted_keys[:10])
    assert all(RANGE_BEGIN <= k < RANGE_END for k, _v in picks)
