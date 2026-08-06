"""Conditional-breakpoint predicate model: a closed set of condition kinds
and the pure matcher that evaluates one against a single event.

Deliberately not an expression language -- MCP tool arguments have to stay
plain JSON with no code-injection surface, so a breakpoint condition is one
of a fixed set of Pydantic models rather than an arbitrary ``eval``'d string.
Mirrors the split already used elsewhere: pure computation lives here (no
presentation, no I/O, no Qiskit import), with three call sites building on
the exact same :func:`matches`:

- the live pause path in :mod:`qisdebug._store` (``TelemetryStore`` blocks
  the calling thread when an event matches a registered condition)
- the MCP background-run poll loop (:mod:`qisdebug._mcp.tools`), which
  reports the :class:`BreakpointHit` that paused a run
- the frontends' "jump to next hit" navigation over an already-captured
  timeline (:mod:`qisdebug._frontends._breakpoints`), which never blocks
  anything live
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from ._analysis import (
    Anomaly,
    anomaly_for_event,
    clbit_ones_total,
    qubit_probability_one,
)
from ._events import (
    ExecutionFinishedEvent,
    GateEvent,
    StateSnapshotEvent,
    TelemetryEvent,
    Timeline,
)

Severity = Literal["info", "warning", "error"]
_SEVERITY_RANK: dict[str, int] = {"info": 0, "warning": 1, "error": 2}
Comparator = Literal["gt", "lt", "ge", "le", "eq", "ne"]


class GateMatchCondition(BaseModel):
    """Break when a gate with this name (and, if given, exactly these
    qubits) is added to a circuit.

    ``circuit_id=None`` matches the gate on any circuit; ``qubits=None``
    matches the gate regardless of which qubits it touches.
    """

    kind: Literal["gate"] = "gate"
    circuit_id: str | None = None
    gate: str
    qubits: tuple[int, ...] | None = None


class SeqThresholdCondition(BaseModel):
    """Break once any event reaches this store-wide sequence number.

    ``seq`` is a :class:`~qisdebug._events.TelemetryEvent.seq` (the store's
    monotonic counter, shared across every circuit it tracks), typically
    read off an earlier captured event to say "stop me right after this
    point next time."
    """

    kind: Literal["seq"] = "seq"
    circuit_id: str | None = None
    seq: int


class WatchThresholdCondition(BaseModel):
    """Break when a watched qubit/clbit/register's numeric value crosses a
    threshold.

    Reuses the same numeric primitives the watch panel computes
    (:func:`~qisdebug._analysis.qubit_probability_one`,
    :func:`~qisdebug._analysis.clbit_ones_total`): marginal P(1) for a qubit
    (checked against each new state snapshot), measured P(1) for a clbit
    (checked against each new execution result). ``watch_kind="register"``
    resolves ``ref`` against the circuit's qregs/cregs the same way a watch
    does, and is treated as whichever bit kind that register turns out to
    be. Unlike gate/seq conditions this always needs one specific circuit,
    since P(1) is meaningless without knowing which circuit's state it's
    read from.
    """

    kind: Literal["watch"] = "watch"
    circuit_id: str
    watch_kind: Literal["qubit", "clbit", "register"]
    ref: str
    comparator: Comparator
    threshold: float


class AnomalyCondition(BaseModel):
    """Break the first time :func:`~qisdebug._analysis.find_anomalies` would
    flag a newly recorded event at or above this severity.

    Covers "some law of quantum mechanics was violated" style conditions --
    a statevector that stopped being normalized, NaN/inf amplitudes, a
    failed or empty execution, a transpiler pass that blew up circuit
    depth -- without the caller having to name the specific check.
    """

    kind: Literal["anomaly"] = "anomaly"
    circuit_id: str | None = None
    min_severity: Severity = "warning"


BreakpointCondition = Annotated[
    Union[
        GateMatchCondition,
        SeqThresholdCondition,
        WatchThresholdCondition,
        AnomalyCondition,
    ],
    Field(discriminator="kind"),
]


class BreakpointHit(BaseModel):
    """One condition matching one event, as reported by the live pause path,
    MCP's ``poll_run``, and frontend "jump to next hit" navigation alike."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    handle: str
    condition: BreakpointCondition
    circuit_id: str
    seq: int
    event_summary: str


def _resolve_watch_bits(
    condition: WatchThresholdCondition, timeline: Timeline
) -> tuple[Literal["qubit", "clbit"], tuple[int, ...]]:
    """Resolve a watch-threshold condition to concrete bit kind and indices.

    Mirrors :func:`qisdebug._frontends._render._watch_bits` for
    :class:`~qisdebug._frontends._render.WatchTarget`, kept as a small
    separate copy rather than a shared import since the input shape differs
    (a Pydantic condition here, a frozen dataclass there) and pulling this
    root module into an import relationship with ``_frontends`` would invert
    the project's presentation-depends-on-core layering.
    """
    if condition.watch_kind in ("qubit", "clbit"):
        return condition.watch_kind, (int(condition.ref),)
    if condition.ref in timeline.qregs:
        return "qubit", tuple(timeline.qregs[condition.ref])
    return "clbit", tuple(timeline.cregs.get(condition.ref, ()))


def _compare(value: float, comparator: Comparator, threshold: float) -> bool:
    if comparator == "gt":
        return value > threshold
    if comparator == "lt":
        return value < threshold
    if comparator == "ge":
        return value >= threshold
    if comparator == "le":
        return value <= threshold
    if comparator == "eq":
        return value == threshold
    return value != threshold  # "ne"


def _matches_qubit_threshold(
    condition: WatchThresholdCondition, event: TelemetryEvent, qubits: tuple[int, ...]
) -> bool:
    if not isinstance(event, StateSnapshotEvent):
        return False
    amplitudes = event.amplitudes()
    if not amplitudes:
        return False
    return any(
        _compare(
            qubit_probability_one(amplitudes, qubit),
            condition.comparator,
            condition.threshold,
        )
        for qubit in qubits
    )


def _matches_clbit_threshold(
    condition: WatchThresholdCondition, event: TelemetryEvent, clbits: tuple[int, ...]
) -> bool:
    if not isinstance(event, ExecutionFinishedEvent) or not event.counts:
        return False
    for clbit in clbits:
        result = clbit_ones_total(event.counts, clbit)
        if result is not None:
            ones, total = result
            if _compare(ones / total, condition.comparator, condition.threshold):
                return True
    return False


def _matches_watch_threshold(
    condition: WatchThresholdCondition, event: TelemetryEvent, timeline: Timeline
) -> bool:
    bit_kind, indices = _resolve_watch_bits(condition, timeline)
    if not indices:
        return False
    if bit_kind == "qubit":
        return _matches_qubit_threshold(condition, event, indices)
    return _matches_clbit_threshold(condition, event, indices)


def _matches_anomaly(
    condition: AnomalyCondition, event: TelemetryEvent, *, prev_depth: int | None
) -> Anomaly | None:
    anomaly, _ = anomaly_for_event(event, prev_depth=prev_depth)
    if anomaly is None:
        return None
    if _SEVERITY_RANK[anomaly.severity] < _SEVERITY_RANK[condition.min_severity]:
        return None
    return anomaly


def matches(
    condition: BreakpointCondition,
    event: TelemetryEvent,
    timeline: Timeline,
    *,
    prev_depth: int | None = None,
) -> bool:
    """Return True when ``event`` (just recorded on ``timeline``) satisfies
    ``condition``.

    ``prev_depth`` is only meaningful for an :class:`AnomalyCondition`
    checking a transpile-track event -- the depth-blowup check needs the
    previous transpile event's depth (see
    :func:`~qisdebug._analysis.anomaly_for_event`); omit it when scanning
    from the start of a timeline, or track it incrementally the way
    :mod:`qisdebug._store` does for the live pause path.
    """
    if condition.circuit_id is not None and condition.circuit_id != timeline.circuit_id:
        return False
    if isinstance(condition, GateMatchCondition):
        if not isinstance(event, GateEvent) or event.gate != condition.gate:
            return False
        return condition.qubits is None or set(condition.qubits) == set(event.qubits)
    if isinstance(condition, SeqThresholdCondition):
        return event.seq >= condition.seq
    if isinstance(condition, WatchThresholdCondition):
        return _matches_watch_threshold(condition, event, timeline)
    return _matches_anomaly(condition, event, prev_depth=prev_depth) is not None
