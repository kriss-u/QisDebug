"""Shared fixtures: every test starts unpatched, empty, and default-configured."""

import pytest

import qisdebug
from qisdebug._config import QisDebugConfig
from qisdebug._events import Timeline


def get_timeline(circuit=None) -> Timeline:
    """Return the timeline for ``circuit``, asserting it was captured.

    Narrows ``qisdebug.timeline()``'s ``Timeline | None`` return for tests that
    know instrumentation was active and must have recorded something.
    """
    timeline = qisdebug.timeline(circuit)
    assert timeline is not None
    return timeline


@pytest.fixture(autouse=True)
def clean_qisdebug():
    """Reset activation state, telemetry, and configuration around each test."""
    qisdebug.deactivate()
    qisdebug.clear()
    qisdebug.configure(**QisDebugConfig().model_dump())
    yield
    qisdebug.deactivate()
    qisdebug.clear()
    qisdebug.configure(**QisDebugConfig().model_dump())
