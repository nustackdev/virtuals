"""Watch a view's notifications on mem with the in-memory transport.

``observed_storage()`` opens a mem storage wired to an in-memory publisher
and observer. ``Hooks`` subscribes named watches built from a view's
``on_*`` options and returns, per write, the keys each watch saw.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

from virtuals.codecs import NoOpCodec
from virtuals.navigator import Navigator
from virtuals.observers.mem import InMemoryObserver
from virtuals.publishers.mem import InMemoryPublisher
from virtuals.storages.mem import InMemoryStorage
from virtuals.tkv.transport import InMemoryTransport


if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from virtuals.tkv.observer import SubscriptionOptions


__all__ = ["Hooks", "observed_storage"]


@contextmanager
def observed_storage() -> Generator[tuple[InMemoryStorage, InMemoryPublisher, InMemoryObserver]]:
    transport = InMemoryTransport()
    publisher = InMemoryPublisher(transport)
    observer = InMemoryObserver(transport)
    storage = InMemoryStorage(codec=NoOpCodec(), publisher=publisher)
    publisher.connect()
    observer.connect()
    storage.open()
    try:
        yield storage, publisher, observer
    finally:
        storage.close()
        observer.disconnect()
        publisher.disconnect()


class Hooks:
    """Named watches on the view at ``root/<address>``, opened as ``view_cls``."""

    def __init__(
        self,
        storage: InMemoryStorage,
        publisher: InMemoryPublisher,
        observer: InMemoryObserver,
        view_cls: type,
        address: str = "v",
    ) -> None:
        self.storage = storage
        self.publisher = publisher
        self.observer = observer
        self.view_cls = view_cls
        self.address = address
        self.nav = Navigator(storage)
        self.hits: dict[str, list] = {}
        self.subs: list = []

    def view(self, ctx: object) -> object:
        return self.nav.root(ctx).open_child(self.address, self.view_cls)

    def write(self, fn: Callable[[object], object]) -> None:
        with self.storage.transaction() as tx:
            fn(self.view(tx))

    def watch(self, name: str, options: SubscriptionOptions) -> None:
        hits: list = []
        self.hits[name] = hits
        sub = self.observer.subscribe(options)
        sub.bind(hits.append)
        self.subs.append(sub)

    def watch_view(self, **builders: Callable[[object], SubscriptionOptions]) -> None:
        """Subscribe each ``name=lambda view: view.on_...()`` against a snapshot view."""
        with self.storage.snapshot() as snap:
            view = self.view(snap)
            for name, build in builders.items():
                self.watch(name, build(view))

    def settle(self) -> None:
        self.publisher.flush(timeout=2.0)
        self.observer.flush(timeout=2.0)

    def run(self, fn: Callable[[object], object]) -> dict[str, list]:
        """Run ``fn`` on the view in one transaction; the keys each watch saw, sorted."""
        self.settle()
        for hits in self.hits.values():
            hits.clear()
        self.write(fn)
        self.settle()
        return {name: sorted(hits, key=repr) for name, hits in self.hits.items()}
