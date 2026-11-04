# Storage Layer: Notifications

Writes are published after they commit. Observers wake subscribers whose filter matches a written key. This page says what a subscriber may rely on. The class docstrings under `virtuals._backends.publishers` and `virtuals._backends.observers` carry the mechanics.

## Pieces

- `Publisher`: attached to a storage. Every commit hands its written keys to `notify()`, which queues them. A worker thread drains the queue and sends batches over the transport.
- `Observer`: owns the receiving side. `subscribe(options)` returns a subscription; `bind(callback)` attaches a receiver. Callbacks run on the observer's worker thread.
- Transports: in-memory (`InMemoryTransport`, one process) and redis pubsub (valkey or redis, any number of processes).
- Views build the filter: `view.on_change()`, `view.on_child_change(addr)`, `view.on_children_change()`, `view.on_descendants_change(*pattern)` return `SubscriptionOptions`. See [filtering](filtering.md).

## Guarantees

**Notify after commit.** Keys are published once the transaction committed. A subscriber woken by a change reads that commit or a later one when it opens a snapshot. A rolled back transaction, an exception inside the block and an empty transaction wake nobody. Write batches notify like transactions.

**Subscribe is ready on return.** Once `subscribe()` returns, every later commit matching the filter is delivered. On redis the observer registers the filter in the server-side routing index after the server confirmed the subscribe, so there is no window. On the in-memory transport the subscription is live at once.

**A watch covers what it names.** `on_child_change(addr)` matches the child's own row and every row under it. A dict, list or shape stored at an untyped address fires the watch on every write inside it. A sibling whose name only starts the same does not match.

**One wake per subscription per key per commit, at most.** A key matching two subscriptions wakes each once. Delivery is per key today: one commit that writes three matching keys may call the receiver up to three times, and a transport may coalesce them. Receivers re-read state on wake, so repeats and coalescing are harmless.

**At least once.** A redis chunk whose reply is lost is sent again, so a receiver may see a key twice.

**Resync on reconnect.** Pub/sub cannot replay writes made while an observer was disconnected. When the redis connection drops, each filter loses its readiness until it is subscribed and registered again. Writes committed in that window are not delivered. Once a filter is re-registered, every local subscription on it gets one keyless wake: the receiver is called with `None`. That means re-read state. Level flows recover exactly. Event flows must treat `None` as a re-read or skip it.

**Drops become resync wakes.** The publisher never drops silently. A failing send is retried with backoff for `publish_retry_seconds` (default 5s). Meanwhile the queue fills and `notify()` blocks (backpressure) for up to `notify_timeout` (default 1s on redis, unbounded on the in-memory transport), then drops the rest of its keys. Every drop is logged at ERROR, counted in `dropped_keys`, raised once by the next `flush()`, and followed by a resync wake to every subscriber on the prefix the next time the worker can publish. A drop costs readers one re-read, never a missed state. A commit while the publisher is disconnected is a drop too: the storage logs it, the keys are counted and raised once by the next `flush()`, and the first connection that sends again carries the resync wake.

**Subscriptions on an absent container work.** A view whose container does not exist yet builds the same filter as an existing one. The first write under it fires the watch. Taking the watch writes nothing.

## Timeouts

| knob | where | default | what it bounds |
|---|---|---|---|
| `notify_timeout` | publisher | 1s redis, none in-memory | how long a committing writer blocks on a full queue before dropping keys |
| `publish_retry_seconds` | redis publisher | 5s | retry window for one failing chunk |
| `socket_timeout` | redis publisher and observer | 2s | one socket call; connect timeout too. One send attempt is bounded by about twice this (one transparent redis-py retry) |
| `subscribe_timeout` | redis observer | 2s | how long `subscribe()` waits for the server to confirm and register |
| `queue_maxsize` | publisher | 100000 keys | queue capacity before backpressure |

A stalled server bounds `subscribe()` at about the subscribe timeout plus twice the socket timeout. A slow receiver never stalls a `subscribe()` from another thread: the listener parks inbound keys in an overflow buffer and keeps serving commands, and while a subscribe awaits its confirmation it reads the socket whatever the backlog, so the confirmation is never behind it. Past the overflow cap with no subscribe pending it stops reading the socket; with one pending the overflow is unbounded, and the real cap is valkey's client output buffer limit, past which the client is disconnected, reconnects and resyncs.

## Lifecycle

- `disconnect()` drains: keys queued before it are sent once each, no retry window. Keys that cannot be sent are counted and surfaced the same way as drops.
- A publisher or observer dropped without `disconnect()` is collected. Its threads exit and its connections are closed by a finalizer.
- The in-memory transport outlives observers. An observer it references is held weakly.
- Self-echo is not suppressed. A process sees its own writes.

## Testing

`tests/conformance/test_s5_notify.py` runs the guarantees above on every backend and both transports. `publisher.flush()` and `observer.flush()` wait for the queues to drain, so a test can assert on delivery without sleeping.
