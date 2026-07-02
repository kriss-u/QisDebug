"""Shared fixtures: every test starts unpatched, empty, and default-configured."""

import pytest

import qdebug
from qdebug._config import QDebugConfig
from qdebug._events import Timeline


def get_timeline(circuit=None) -> Timeline:
    """Return the timeline for ``circuit``, asserting it was captured.

    Narrows ``qdebug.timeline()``'s ``Timeline | None`` return for tests that
    know instrumentation was active and must have recorded something.
    """
    timeline = qdebug.timeline(circuit)
    assert timeline is not None
    return timeline


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
