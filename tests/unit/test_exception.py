"""Tests for pv._exception module."""

import pytest

from pv._exception import PVError


class TestPVError:
    """Test cases for PVError exception class."""

    def test_exception_can_be_raised(self):
        """Test that PVError can be raised."""
        with pytest.raises(PVError):
            raise PVError()

    def test_exception_can_be_caught(self):
        """Test that PVError can be caught."""
        try:
            raise PVError("test error")
        except PVError:
            # Exception was caught successfully
            pass
        else:
            pytest.fail("PVError was not caught")

    def test_exception_message_handling(self):
        """Test that exception message is stored and retrieved correctly."""
        error_message = "This is a test error message"
        exc = PVError(error_message)
        assert str(exc) == error_message

    def test_exception_with_empty_message(self):
        """Test exception with empty message."""
        exc = PVError()
        assert str(exc) == ""

    def test_exception_is_instance_of_exception(self):
        """Test that PVError is an instance of Exception."""
        exc = PVError()
        assert isinstance(exc, Exception)
        assert isinstance(exc, PVError)

    def test_exception_inheritance_chain(self):
        """Test the inheritance chain of PVError."""
        exc = PVError("test")
        assert issubclass(PVError, Exception)
        assert isinstance(exc, BaseException)
