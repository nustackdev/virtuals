"""IndexedDictView — dict view with hierarchical internal layout.

Layout under the view's container::

    site/
    ├── __keys__/    # FlatListView — ordered key index
    │   ├── 0: "alice"
    │   ├── 1: "bob"
    │   └── ...
    └── __data__/    # actual dict children (same as EagerDictView)
        ├── alice: {...}
        └── bob: {...}

No /m tree.  Length = len(__keys__).  Data ops route to ``__data__/``,
key ops route to ``__keys__/``.

Provides three classes following the eager/lazy facet pattern:
- IndexedDictViewBase: shared mutations, keys, lifecycle
- EagerIndexedDictView: reads return extracted Python values
- LazyIndexedDictView: reads return child Views for containers, values for primitives
"""

from __future__ import annotations

from collections.abc import ItemsView, KeysView, MutableMapping, ValuesView
from typing import TYPE_CHECKING, ClassVar, cast

from virtuals.container import Container, ContainerProtocol, ContainerStructure, NodeType
from virtuals.types import EMPTY, Empty, Value
from virtuals.view import (
    ChildNavigationBase,
    ChildNestedGetBase,
    ChildNestedSetBase,
    ChildObservableBase,
    ChildPrimitiveSetBase,
    LazyChildReadBase,
    ObservableBase,
    UnsafePrimitiveOpsBase,
    ViewBase,
)

from .flat_list_view import FlatListView


if TYPE_CHECKING:
    from collections.abc import Generator
    from collections.abc import Mapping as PyMapping

    from virtuals.view import View


__all__ = [
    "EagerIndexedDictView",
    "IndexedDictViewBase",
    "LazyIndexedDictView",
]

_KEYS = "__keys__"
_DATA = "__data__"


# =============================================================================
# BASE — shared by eager and lazy facets
# =============================================================================


class IndexedDictViewBase(
    ObservableBase,
    ChildObservableBase[str | int],
    ChildNavigationBase[str | int],
    ChildNestedGetBase,
    ChildNestedSetBase,
    ChildPrimitiveSetBase,
    LazyChildReadBase,
    UnsafePrimitiveOpsBase,
    ViewBase,
):
    """Dict view base with hierarchical internal layout for O(1) key access.

    Stores data in ``__data__/`` child container and key index in ``__keys__/``
    child container (FlatListView). Shared by eager and lazy facets.

    Example::

        >>> dct = nav.root(ctx)
        >>> dct.store({"a": 1, "b": 2, "c": 3})
        >>> list(dct.keys())   # reads from __keys__/
        ['a', 'b', 'c']
    """

    STRUCTURE: ClassVar[ContainerStructure] = ContainerStructure(13)
    PROTOCOL: ClassVar[ContainerProtocol] = ContainerProtocol.MAPPING | ContainerProtocol.MUTABLE

    @classmethod
    def is_address_static(cls, address: object) -> bool:
        # Must be False: data lives under __data__/ so navigate_view
        # must go through open_child() to insert the __data__ segment.
        return False

    def normalize_address(self, address: str | int) -> str | int:
        return address

    # -- Internal containers -----------------------------------------------

    def _data_container(self) -> Container:
        """Container for actual dict data children."""
        return Container(
            ctx=self.container.ctx,
            site=(*self.container.site, _DATA),
        )

    def _keys_view(self) -> FlatListView:
        """FlatListView over ``__keys__/`` child container."""
        container = Container(
            ctx=self.container.ctx,
            site=(*self.container.site, _KEYS),
        )
        return FlatListView(container, self.registry)

    def _ensure_layout(self) -> None:
        """Ensure both child containers exist."""
        self.ensure_created()
        self.container.create_child_container(
            _DATA,
            structure=ContainerStructure(1),
            protocol=ContainerProtocol.MAPPING | ContainerProtocol.MUTABLE,
            validate=False,
        )
        self.container.create_child_container(
            _KEYS,
            structure=FlatListView.STRUCTURE,
            protocol=FlatListView.PROTOCOL,
            validate=False,
        )

    # -- Membership & keys (same for both facets) ---------------------------

    def __len__(self) -> int:
        return len(self._keys_view())

    def __contains__(self, obj: str | int) -> bool:
        return self._data_container().exists_child(obj)

    def __iter__(self) -> Generator[str | int, None, None]:
        self._ensure_keys_synced()
        yield from self._keys_view()

    def key_at(self, idx: int) -> str | int:
        """Get key at index — single O(1) read from __keys__ FlatListView."""
        return self._keys_view()[idx]

    def _ensure_keys_synced(self) -> None:
        """Sync __keys__ from __data__ if empty."""
        kv = self._keys_view()
        if len(kv) == 0:
            dc = self._data_container()
            if dc.exists():
                data_keys = list(dc.iter_child_keys(validate=False))
                if data_keys:
                    self._ensure_layout()
                    kv = self._keys_view()
                    kv.store(data_keys)

    def keys(self) -> KeysView[str | int]:
        """Get all keys as a set-like view."""
        self._ensure_keys_synced()
        return KeysView(self)  # type: ignore[arg-type]

    def get(self, address: str | int, default: object | Empty = EMPTY) -> object | Empty:
        try:
            return self[address]
        except (KeyError, Exception):
            return default

    # -- Mutations (same for both facets) -----------------------------------

    def __setitem__(self, address: str | int, value: object) -> None:
        self._ensure_layout()
        dc = self._data_container()
        is_new = not dc.exists_child(address)
        # Write to __data__
        if self.registry.is_container_type(value):
            from virtuals.collections import Initializable

            value_type = value.__class__
            view_class = self.registry.get_view_for_type(value_type)
            structure_id = view_class.get_structure()
            protocol_hints = view_class.get_protocol()
            child_container = dc.create_child_container(
                address,
                structure=ContainerStructure(structure_id),
                protocol=protocol_hints,
            )
            child_view = view_class(container=child_container, registry=self.registry)
            if not isinstance(child_view, Initializable):
                raise TypeError(f"Child view {view_class.__name__} does not support initialization")
            child_view.store(value)
        else:
            dc.put_child_primitive(address, cast("Value", value))
        if is_new:
            self._keys_view().append(address)

    def __delitem__(self, address: str | int) -> None:
        self._ensure_layout()
        dc = self._data_container()
        if not dc.exists_child(address):
            raise KeyError(address)
        dc.delete_child(address)
        # Rebuild key index
        kv = self._keys_view()
        remaining = [k for k in kv if k != address]
        kv.store(remaining)

    def clear(self) -> None:
        self._ensure_layout()
        self._data_container().clear_children(validate=False)
        self._keys_view().clear()

    def update(self, other: PyMapping[str | int, object] | None = None, **kwargs: object) -> None:
        if other:
            for key, value in other.items():
                self[key] = value
        for key, value in kwargs.items():
            self[key] = value  # type: ignore[assignment]

    def store(self, value: PyMapping[str | int, object], *, replace: bool = True) -> None:
        self._ensure_layout()
        if replace and len(self) > 0:
            self.clear()

        keys: list[str | int] = []
        dc = self._data_container()
        for key, val in value.items():
            if self.registry.is_container_type(val):
                from virtuals.collections import Initializable

                value_type = val.__class__
                view_class = self.registry.get_view_for_type(value_type)
                structure_id = view_class.get_structure()
                protocol_hints = view_class.get_protocol()
                child_container = dc.create_child_container(
                    key,
                    structure=ContainerStructure(structure_id),
                    protocol=protocol_hints,
                )
                child_view = view_class(container=child_container, registry=self.registry)
                if not isinstance(child_view, Initializable):
                    raise TypeError(
                        f"Child view {view_class.__name__} does not support initialization"
                    )
                child_view.store(val)
            else:
                dc.put_child_primitive(key, cast("Value", val))
            keys.append(key)

        self._keys_view().store(keys)

    # -- Navigation --------------------------------------------------------

    def open_child(self, address: str | int, view: type) -> View:  # type: ignore[override]
        """Open child view under __data__/."""
        normalized = self.normalize_address(address)
        dc = self._data_container()
        child_site = (*dc.site, normalized)
        child_container = Container(ctx=dc.ctx, site=child_site)
        return view(child_container, self.registry)

    # -- Internal read helpers ---------------------------------------------

    def _read_child_from_data(self, address: str | int) -> tuple[object, bool]:
        """Read node info from __data__ container.

        Returns:
            (node_info, is_container) — the raw node info and whether it's a container
        """
        dc = self._data_container()
        child_site = (*dc.site, address)
        from virtuals.container import node_ops

        node_info = node_ops.get_node_info(child_site, dc.ctx)
        if not node_info.exists or node_info.node_type == NodeType.NOT_FOUND:
            raise KeyError(address)
        return node_info, node_info.node_type == NodeType.CONTAINER

    def _extract_data_child(self, address: str | int, node_info: object) -> object:
        """Extract a container child from __data__ using the registry."""
        from virtuals.collections import Convertible

        dc = self._data_container()
        child_site = (*dc.site, address)
        child_container = Container(ctx=dc.ctx, site=child_site)
        if node_info.structure is None:  # type: ignore[union-attr]
            raise ValueError(f"Child container '{address}' has no structure ID")
        view_class = self.registry.get_view_for_structure(node_info.structure)  # type: ignore[union-attr]
        child_view = view_class(container=child_container, registry=self.registry)
        if not isinstance(child_view, Convertible):
            raise TypeError(f"Child view {view_class.__name__} does not support extraction")
        return child_view.extract()

    def _view_data_child(self, address: str | int, node_info: object) -> object:
        """Return a View for a container child from __data__ (no extraction)."""
        dc = self._data_container()
        child_site = (*dc.site, address)
        child_container = Container(ctx=dc.ctx, site=child_site)
        if node_info.structure is None:  # type: ignore[union-attr]
            raise ValueError(f"Child container '{address}' has no structure ID")
        view_class = self.registry.get_view_for_structure(node_info.structure)  # type: ignore[union-attr]
        return view_class(container=child_container, registry=self.registry)


# =============================================================================
# EAGER FACET — reads return extracted Python values
# =============================================================================


class EagerIndexedDictView(IndexedDictViewBase):
    """Eager indexed dict view — reads return extracted Python values.

    Cross-navigate to lazy facet via .lazy property.
    """

    CONTAINER_CLS: ClassVar[type] = dict

    def __getitem__(self, address: str | int) -> object | Empty:
        node_info, is_container = self._read_child_from_data(address)
        if not is_container:
            from virtuals.types import is_empty

            if is_empty(node_info.primitive_value):  # type: ignore[union-attr]
                raise KeyError(address)
            return node_info.primitive_value  # type: ignore[union-attr]
        return self._extract_data_child(address, node_info)

    def values(self) -> ValuesView[object]:
        """Get all values as a collection view."""
        return ValuesView(self)  # type: ignore[arg-type]

    def items(self) -> ItemsView[str | int, object]:
        """Get all items as a set-like view."""
        return ItemsView(self)  # type: ignore[arg-type]

    def pop(self, address: str | int, default: object | Empty = EMPTY) -> object | Empty:
        try:
            value = self[address]
            del self[address]
            return value
        except KeyError:
            from virtuals.types import is_empty

            if is_empty(default):
                raise
            return default

    def extract(self) -> dict[str | int, object]:
        return {k: self[k] for k in self}

    # -- Facet navigation --------------------------------------------------

    @property
    def lazy(self) -> LazyIndexedDictView:
        """Switch to lazy facet — reads return child Views."""
        return LazyIndexedDictView(container=self.container, registry=self.registry)

    @property
    def eager(self) -> EagerIndexedDictView:
        """Identity — already eager."""
        return self


# =============================================================================
# LAZY FACET — reads return child Views
# =============================================================================


class LazyIndexedDictView(IndexedDictViewBase):
    """Lazy indexed dict view — reads return child Views for containers.

    Cross-navigate to eager facet via .eager property.
    """

    def __getitem__(self, address: str | int) -> object:
        node_info, is_container = self._read_child_from_data(address)
        if not is_container:
            from virtuals.types import is_empty

            if is_empty(node_info.primitive_value):  # type: ignore[union-attr]
                raise KeyError(address)
            return node_info.primitive_value  # type: ignore[union-attr]
        return self._view_data_child(address, node_info)

    def values(self) -> ValuesView[object]:
        """Get all values as a collection view."""
        return ValuesView(self)  # type: ignore[arg-type]

    def items(self) -> ItemsView[str | int, object]:
        """Get all items as a set-like view."""
        return ItemsView(self)  # type: ignore[arg-type]

    # -- Facet navigation --------------------------------------------------

    @property
    def eager(self) -> EagerIndexedDictView:
        """Switch to eager facet — reads return extracted values."""
        return EagerIndexedDictView(container=self.container, registry=self.registry)

    @property
    def lazy(self) -> LazyIndexedDictView:
        """Identity — already lazy."""
        return self


MutableMapping.register(EagerIndexedDictView)
