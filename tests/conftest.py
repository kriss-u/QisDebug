"""Shared fixtures: every test starts unpatched, empty, and default-configured."""

import pytest

import qdebug
from qdebug._config import QDebugConfig


@pytest.fixture(autouse=True)
def clean_qdebug():
    """Reset activation state, telemetry, and configuration around each test."""
    qdebug.deactivate()
    qdebug.clear()
    qdebug.configure(**QDebugConfig().model_dump())
    yield
    qdebug.deactivate()
    qdebug.clear()
    qdebug.configure(**QDebugConfig().model_dump())
