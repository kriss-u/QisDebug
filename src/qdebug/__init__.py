"""QDebug: zero-touch instrumentation and timeline debugging for Qiskit.

Attach full circuit and execution visibility to existing Qiskit code without
changing it::

    import qdebug

    qdebug.activate()

    # ... build, transpile and run circuits exactly as usual ...

    print(qdebug.timeline(circuit).summary())

Importing this module has no side effects; nothing is patched until
:func:`activate` is called, and :func:`deactivate` restores every original
method.
"""

from __future__ import annotations

from typing import Any

from ._activation import ActivationReport, activate, deactivate, is_active
from ._config import QDebugConfig, configure, get_config
from ._events import (
    EventType,
    ExecutionFinishedEvent,
    ExecutionSubmittedEvent,
    GateEvent,
    NoteEvent,
    StateSnapshotEvent,
    TelemetryEvent,
    Timeline,
    TranspileFinishedEvent,
    TranspilePassEvent,
    TranspileStartedEvent,
)
from ._store import STORE

__version__ = "0.1.0"

__all__ = [
    "ActivationReport",
    "EventType",
    "ExecutionFinishedEvent",
    "ExecutionSubmittedEvent",
    "GateEvent",
    "NoteEvent",
    "QDebugConfig",
    "StateSnapshotEvent",
    "TelemetryEvent",
    "Timeline",
    "TranspileFinishedEvent",
    "TranspilePassEvent",
    "TranspileStartedEvent",
    "activate",
    "clear",
    "configure",
    "deactivate",
    "get_config",
    "is_active",
    "timeline",
    "timelines",
]


def timeline(circuit: Any = None) -> Timeline | None:
    """Return the timeline for ``circuit``, or the most recent one when None.

    ``circuit`` may be a QuantumCircuit that was touched while QDebug was
    active, or a timeline id string.
    """
    if circuit is None:
        return STORE.latest()
    return STORE.get(circuit)


def timelines() -> list[Timeline]:
    """Return all tracked timelines, most recently touched first."""
    return STORE.all_timelines()


def clear() -> None:
    """Drop all recorded telemetry without touching the active patches."""
    STORE.clear()
