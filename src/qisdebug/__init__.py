"""QisDebug: zero-touch instrumentation and timeline debugging for Qiskit.

Attach full circuit and execution visibility to existing Qiskit code without
changing it::

    import qisdebug

    debug = qisdebug.QisDebug()
    debug.start()

    # ... build, transpile and run circuits exactly as usual ...

    print(debug.get(circuit).summary())

    debug.end()

Importing this module has no side effects; nothing is patched until
:meth:`QisDebug.start` is called, and :meth:`QisDebug.end` restores every
original method this instance was using. Multiple independently scoped
``QisDebug()`` instances can be started at once.
"""

from __future__ import annotations

from ._breakpoints import (
    AnomalyCondition,
    BreakpointCondition,
    BreakpointHit,
    GateMatchCondition,
    SeqThresholdCondition,
    WatchThresholdCondition,
)
from ._config import QisDebugConfig
from ._debug import ActivationReport, QisDebug
from ._errors import QisDebugError, TimelineNotFoundError
from ._events import (
    EventType,
    ExecutionFinishedEvent,
    ExecutionSubmittedEvent,
    GateEvent,
    NoteEvent,
    PropertySetChanges,
    StateSnapshotEvent,
    TelemetryEvent,
    Timeline,
    TranspileFinishedEvent,
    TranspilePassEvent,
    TranspileStartedEvent,
)

__version__ = "0.1.1"

__all__ = [
    "ActivationReport",
    "AnomalyCondition",
    "BreakpointCondition",
    "BreakpointHit",
    "EventType",
    "ExecutionFinishedEvent",
    "ExecutionSubmittedEvent",
    "GateEvent",
    "GateMatchCondition",
    "NoteEvent",
    "PropertySetChanges",
    "QisDebug",
    "QisDebugConfig",
    "QisDebugError",
    "SeqThresholdCondition",
    "StateSnapshotEvent",
    "TelemetryEvent",
    "Timeline",
    "TimelineNotFoundError",
    "TranspileFinishedEvent",
    "TranspilePassEvent",
    "TranspileStartedEvent",
    "WatchThresholdCondition",
]
