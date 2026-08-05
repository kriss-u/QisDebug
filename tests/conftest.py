"""Shared fixtures: every test gets its own unpatched, empty QisDebug instance."""

import pytest

import qisdebug


@pytest.fixture
def debug():
    """A fresh QisDebug instance, ended automatically after the test."""
    instance = qisdebug.QisDebug()
    yield instance
    instance.end()
