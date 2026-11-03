# virtuals test suite

```sh
.venv/bin/python -m pytest -q -p no:cacheprovider tests            # everything
.venv/bin/python -m pytest -q -p no:cacheprovider tests -n 8       # parallel (xdist)
.venv/bin/python -m pytest -q -p no:cacheprovider tests/views      # one area
.venv/bin/python -m pytest -q -p no:cacheprovider tests -k lmdb      # one backend
```

The full suite takes about 3 minutes serial and under a minute with `-n 8`. Every test runs on the project venv; there is no other configuration.

## Layout

| directory | what it covers | backends |
|---|---|---|
| `conformance/` | the kv contract as executable cases, one run per backend. Transactions, snapshots, errors, process model, notifications, cost. See its [README](conformance/README.md) | sqlite, lmdb, rocksdb, mem; notify on the in-memory and redis transports |
| `_backends/storages/` | per backend behaviour below the contract: locks, threads, fork, close races, map growth, secondaries. `_write_slot_cases.py` is the shared writer slot suite each backend runs | one file per backend |
| `_backends/publishers/`, `_backends/observers/`, `_backends/notify/` | publisher queue, retry, drops and resync; observer readiness, reconnect, backlog; the redis routing function | in-memory and redis |
| `_backends/test_notify_*.py` | notify end to end: gap after subscribe, backpressure, reconnect | in-memory and redis |
| `views/` | every view against storage: behaviour on every backend, op counts on the seeking backends (sqlite, lmdb, rocksdb). `test_dict_views_hardening.py` pins the contract rules for dict views: EMPTY, existence, kind change, cost | all four |
| `functional/` | container and view behaviour through the public API, watches on subtrees and absent containers | mem |
| `unit/` | pure units: key and path navigation, container types and markers, view registry and capabilities, fractional index keys | none |
| `tkv/` | the tkv protocols: codecs, filters, transport, compliance suites a backend can import | compliance on sqlite, lmdb, rocksdb, mem; the rest none |
| `support/` | shared helpers: `counting.py` (op counter for cost tests, `open_backend`), `valkey.py` (a private valkey for redis tests), `dict_view.py` | |

## Conventions

- Every test that touches a file backend uses `tmp_path`. Nothing is left on disk.
- Redis tests use `TEST_REDIS_URL` when set, else start a private valkey under `tmp_path` and skip if none can be started.
- Cost tests count storage ops, not time. They assert that an op costs the same at 10 and at 1000 siblings, or that `len` reads children only. See `support/counting.py`.
- Concurrency tests force the race (a pause between read and write, a barrier before the write) and retry only the two retryable error classes. A stalled thread fails the case; the conformance suite joins such threads after the storage closed.
- Timing budgets are generous (100ms to 300ms) and hold under 3x load. A failure on time is a real stall, not noise.
- Known contract failures are strict xfails in `conformance/known_failures.py`. An unexpected pass fails the run, so a fix deletes the row in the same commit.
- Notify tests wait on `publisher.flush()` and `observer.flush()` or an event, not on sleeps, except for a few negative checks in `conformance/test_s5_notify.py` that wait a quiet period to assert nothing arrived.
- Multi-process cases spawn workers (`spawn` start method) and skip on single-process backends (rocksdb, mem).

## Adding a backend

Add a spec to `conformance/backends.py` (factory, process model, writer slot flag) and a line to `support/counting.open_backend`. Run `tests/conformance` and `tests/views`. Known gaps go into `known_failures.py` with an owner, as strict xfails.
