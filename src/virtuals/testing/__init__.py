"""Testing utilities for tkv.tkv.

This module provides reusable test infrastructure for packages that implement
tkv protocols. Import compliance test suites to verify your implementations.

Example:
    ```python
    from tkv.testing import (
        StorageProtocolCompliance,
        ObserverCompliance,
        KeyCodecCompliance,
    )


    class TestMyStorage(StorageProtocolCompliance):
        @pytest.fixture
        def storage(self):
            db = MyStorage()
            db.open()
            yield db
            db.close()


    class TestMyObserver(ObserverCompliance):
        @pytest.fixture
        def observable_storage(self):
            return MyObservableStorage()


    class TestMyCodec(KeyCodecCompliance):
        @pytest.fixture
        def codec(self):
            return MyKeyCodec()
    ```
"""

from .codec_compliance import (
    KeyCodecCompliance,
    ValueCodecCompliance,
)
from .observer_compliance import (
    ObserverCompliance,
    RegistryCompliance,
    SubscriptionCompliance,
)
from .storage_compliance import StorageProtocolCompliance


__all__ = [
    "KeyCodecCompliance",
    "ObserverCompliance",
    "RegistryCompliance",
    "StorageProtocolCompliance",
    "SubscriptionCompliance",
    "ValueCodecCompliance",
]
