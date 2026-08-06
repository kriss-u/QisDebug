"""Breakpoints as a navigation aid over an already-captured timeline.

"Jump to next hit" reuses the exact same :func:`~qisdebug._breakpoints.matches`
the live pause path in :mod:`qisdebug._store` checks against -- the only
difference is this walks a merged, already-recorded event list instead of
blocking anything live, so a breakpoint set while replaying a captured run
means the same thing it would have meant live.
"""

from __future__ import annotations

from typing import Literal, cast

from pydantic import ValidationError

from .._breakpoints import (
    AnomalyCondition,
    BreakpointCondition,
    Comparator,
    GateMatchCondition,
    Severity,
    SeqThresholdCondition,
    WatchThresholdCondition,
    matches,
)
from .._events import TelemetryEvent, Timeline

Merged = list[tuple[Timeline, TelemetryEvent]]


class FrontendBreakpoints:
    """A frontend session's set of breakpoint conditions plus handle
    assignment -- purely for "jump to next hit" navigation, no live pause
    (see :class:`qisdebug._store.BreakpointRegistry` for that). Bundling
    the condition dict and handle counter into one object, shared by
    ``cli.py`` and ``gui.py``, keeps each frontend's session class down to
    one attribute for this feature instead of two, and means the two
    frontends can't drift on how handles are assigned.
    """

    def __init__(self) -> None:
        self._conditions: dict[str, BreakpointCondition] = {}
        self._next_id = 0

    def add(self, condition: BreakpointCondition) -> str:
        """Register a condition, returning a handle for :meth:`remove`."""
        self._next_id += 1
        handle = str(self._next_id)
        self._conditions[handle] = condition
        return handle

    def remove(self, handle: str) -> None:
        """Unregister a breakpoint by its handle. No-op if already removed."""
        self._conditions.pop(handle, None)

    def clear(self) -> None:
        """Remove every registered breakpoint."""
        self._conditions.clear()

    def items(self) -> list[tuple[str, BreakpointCondition]]:
        """Return every active ``(handle, condition)`` pair."""
        return list(self._conditions.items())

    def as_dict(self) -> dict[str, BreakpointCondition]:
        """Return a plain ``{handle: condition}`` copy, for :func:`find_next_hit`."""
        return dict(self._conditions)

    def get(self, handle: str) -> BreakpointCondition | None:
        """Return the condition for ``handle``, or ``None`` if not registered."""
        return self._conditions.get(handle)

    def __bool__(self) -> bool:
        return bool(self._conditions)

    def __len__(self) -> int:
        return len(self._conditions)

    def __contains__(self, handle: str) -> bool:
        return handle in self._conditions


CONDITION_USAGE = (
    "gate NAME [Q1 Q2 ...] | seq N | "
    "watch qubit|clbit|register REF gt|lt|ge|le|eq|ne THRESHOLD | "
    "anomaly [info|warning|error]"
)
_COMPARATORS: tuple[Comparator, ...] = ("gt", "lt", "ge", "le", "eq", "ne")
_SEVERITIES: tuple[Severity, ...] = ("info", "warning", "error")


def parse_condition_args(
    kind: str, args: list[str], circuit_id: str
) -> BreakpointCondition:
    """Parse one condition's kind and remaining tokens into a
    :class:`~qisdebug._breakpoints.BreakpointCondition`, scoped to
    ``circuit_id``.

    A small fixed per-kind tokenizer, not an expression parser -- breakpoint
    conditions are a closed set (see :mod:`qisdebug._breakpoints`), never
    arbitrary code. Shared by both frontends (CLI's ``break`` command line,
    GUI's condition-builder row) so the two accept exactly the same syntax
    and never drift apart on what's valid. Raises ``ValueError`` (including
    on a wrapped :class:`pydantic.ValidationError`) with a message safe to
    show the user directly.
    """
    try:
        if kind == "gate":
            if not args:
                raise ValueError(f"usage: gate NAME [Q1 Q2 ...]  ({CONDITION_USAGE})")
            gate, qubit_args = args[0], args[1:]
            qubits = tuple(int(q) for q in qubit_args) if qubit_args else None
            return GateMatchCondition(circuit_id=circuit_id, gate=gate, qubits=qubits)
        if kind == "seq":
            if len(args) != 1:
                raise ValueError(f"usage: seq N  ({CONDITION_USAGE})")
            return SeqThresholdCondition(circuit_id=circuit_id, seq=int(args[0]))
        if kind == "watch":
            usage = (
                "usage: watch qubit|clbit|register REF gt|lt|ge|le|eq|ne THRESHOLD"
                f"  ({CONDITION_USAGE})"
            )
            if len(args) != 4 or args[0] not in ("qubit", "clbit", "register"):
                raise ValueError(usage)
            watch_kind_arg, ref, comparator_arg, threshold_arg = args
            if comparator_arg not in _COMPARATORS:
                raise ValueError(usage)
            watch_kind = cast(Literal["qubit", "clbit", "register"], watch_kind_arg)
            return WatchThresholdCondition(
                circuit_id=circuit_id,
                watch_kind=watch_kind,
                ref=ref,
                comparator=cast(Comparator, comparator_arg),
                threshold=float(threshold_arg),
            )
        if kind == "anomaly":
            if len(args) > 1 or (args and args[0] not in _SEVERITIES):
                raise ValueError(
                    f"usage: anomaly [info|warning|error]  ({CONDITION_USAGE})"
                )
            min_severity = cast(Severity, args[0]) if args else "warning"
            return AnomalyCondition(circuit_id=circuit_id, min_severity=min_severity)
        raise ValueError(f"unknown breakpoint kind {kind!r}  ({CONDITION_USAGE})")
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc


def find_next_hit(
    merged: Merged,
    conditions: dict[str, BreakpointCondition],
    start_index: int,
) -> tuple[int, str] | None:
    """Return ``(index, handle)`` of the first event after ``start_index``
    that satisfies any registered condition, or ``None`` if none do.

    ``conditions`` is a ``{handle: condition}`` mapping, as returned by
    :meth:`~qisdebug.QisDebug.list_breakpoints`. An :class:`~qisdebug._breakpoints.AnomalyCondition`
    checking transpile depth needs the previous transpile event's depth (see
    :func:`~qisdebug._analysis.anomaly_for_event`), so this replays depth
    per circuit from the start of ``merged`` up through ``start_index`` to
    seed that state correctly before searching forward, the same way
    :class:`~qisdebug._store.TelemetryStore` tracks it incrementally live.
    """
    prev_depth: dict[str, int | None] = {}
    for i in range(start_index + 1):
        timeline, event = merged[i]
        depth = getattr(event, "depth", None)
        if depth is not None:
            prev_depth[timeline.circuit_id] = depth
    for i in range(start_index + 1, len(merged)):
        timeline, event = merged[i]
        for handle, condition in conditions.items():
            if matches(
                condition,
                event,
                timeline,
                prev_depth=prev_depth.get(timeline.circuit_id),
            ):
                return i, handle
        depth = getattr(event, "depth", None)
        if depth is not None:
            prev_depth[timeline.circuit_id] = depth
    return None


def describe_condition(condition: BreakpointCondition) -> str:
    """Return a compact one-line label for a breakpoint condition, for
    listing widgets in both frontends."""
    if condition.kind == "gate":
        qubits = f" q{list(condition.qubits)}" if condition.qubits is not None else ""
        scope = f" on {condition.circuit_id}" if condition.circuit_id else ""
        return f"gate {condition.gate}{qubits}{scope}"
    if condition.kind == "seq":
        scope = f" on {condition.circuit_id}" if condition.circuit_id else ""
        return f"seq >= {condition.seq}{scope}"
    if condition.kind == "watch":
        return (
            f"{condition.circuit_id} {condition.watch_kind} {condition.ref} "
            f"{condition.comparator} {condition.threshold}"
        )
    scope = f" on {condition.circuit_id}" if condition.circuit_id else ""
    return f"anomaly >= {condition.min_severity}{scope}"
