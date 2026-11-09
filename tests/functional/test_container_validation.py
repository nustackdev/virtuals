"""Functional tests for parent-chain validation: a missing parent names its site."""

import pytest

from virtuals.container import (
    ContainerNotFoundError,
    ContainerProtocol,
    ContainerStructure,
    create_container,
    validate_parents_chain,
    validate_parents_exist,
)
from virtuals.tkv.storage import TransactionProtocol


@pytest.mark.parametrize("validate", [validate_parents_exist, validate_parents_chain])
def test_a_missing_parent_names_the_topmost_missing_site(tx: TransactionProtocol, validate) -> None:
    create_container(("users",), ContainerStructure(1), ContainerProtocol.MUTABLE, tx)
    with pytest.raises(ContainerNotFoundError) as info:
        validate(("users", "alice", "profile", "name"), tx)
    assert info.value.site == ("users", "alice")


@pytest.mark.parametrize("validate", [validate_parents_exist, validate_parents_chain])
def test_a_present_chain_passes(tx: TransactionProtocol, validate) -> None:
    create_container(("users",), ContainerStructure(1), ContainerProtocol.MUTABLE, tx)
    validate(("users", "alice"), tx)
