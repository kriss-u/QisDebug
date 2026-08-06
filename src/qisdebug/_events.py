"""Event models and the per-circuit timeline that the visualizer reads."""

from __future__ import annotations

import time
from enum import Enum
from typing import TYPE_CHECKING, Any, Iterable, Literal, cast, overload

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

if TYPE_CHECKING:
    from ._store import TelemetryStore


class EventType(str, Enum):
    """Kinds of telemetry events QisDebug records."""

    GATE_ADDED = "gate_added"
    TRANSPILE_STARTED = "transpile_started"
    TRANSPILE_PASS = "transpile_pass"
    TRANSPILE_FINISHED = "transpile_finished"
    EXECUTION_SUBMITTED = "execution_submitted"
    STATE_SNAPSHOT = "state_snapshot"
    EXECUTION_FINISHED = "execution_finished"
    NOTE = "note"


class TelemetryEvent(BaseModel):
    """Base class for all timeline events.

    ``seq`` is a process-wide monotonic counter, so events from different
    layers interleave in the order they actually happened even when their
    wall-clock timestamps collide.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: EventType
    seq: int
    timestamp: float = Field(default_factory=time.time)

    def describe(self) -> str:
        """Return a one-line human-readable rendering of the event."""
        return self.type.value


class GateEvent(TelemetryEvent):
    """An instruction appended to a circuit during construction."""

    type: EventType = EventType.GATE_ADDED
    gate: str
    qubits: tuple[int, ...] = ()
    clbits: tuple[int, ...] = ()
    params: tuple[float | str, ...] = ()
    operation: Any = Field(default=None, exclude=True, repr=False)

    def describe(self) -> str:
        """Return e.g. ``h q[0]`` or ``cx q[0, 1]``."""
        text = f"{self.gate} q{list(self.qubits)}"
        if self.clbits:
            text += f" -> c{list(self.clbits)}"
        if self.params:
            text += f" ({', '.join(str(p) for p in self.params)})"
        return text


class TranspileStartedEvent(TelemetryEvent):
    """A pass manager began running on the circuit.

    ``phase`` distinguishes user-driven transpilation from pass manager runs
    a backend performs internally while executing the circuit.
    """

    type: EventType = EventType.TRANSPILE_STARTED
    pass_manager: str
    depth: int | None = None
    size: int | None = None
    phase: Literal["user", "backend"] = "user"

    def describe(self) -> str:
        """Return the pass manager name with the input circuit shape."""
        verb = (
            "backend compile started"
            if self.phase == "backend"
            else "transpile started"
        )
        return f"{verb} ({self.pass_manager}, depth={self.depth}, size={self.size})"


class PropertySetChanges(BaseModel):
    """Added/removed/changed keys in a pass's property_set vs. the previous pass."""

    added: dict[str, str] = Field(default_factory=dict)
    removed: dict[str, str] = Field(default_factory=dict)
    changed: dict[str, tuple[str, str]] = Field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.added or self.removed or self.changed)


class TranspilePassEvent(TelemetryEvent):
    """One transpiler pass completed."""

    type: EventType = EventType.TRANSPILE_PASS
    pass_name: str
    duration: float = 0.0
    depth: int | None = None
    size: int | None = None
    ops: dict[str, int] = Field(default_factory=dict)
    changed: bool = False
    phase: Literal["user", "backend"] = "user"
    circuit: Any = Field(default=None, exclude=True, repr=False)
    property_changes: PropertySetChanges | None = None
    property_set: dict[str, str] | None = Field(default=None, exclude=True, repr=False)

    def describe(self) -> str:
        """Return the pass name and whether it changed the circuit or property_set."""
        marker = "*" if self.changed else " "
        prefix = "backend pass" if self.phase == "backend" else "pass"
        text = (
            f"{prefix}{marker} {self.pass_name} (depth={self.depth}, size={self.size})"
        )
        if self.property_changes:
            text += (
                f" [props +{len(self.property_changes.added)}"
                f"/-{len(self.property_changes.removed)}"
                f"~{len(self.property_changes.changed)}]"
            )
        return text


class TranspileFinishedEvent(TelemetryEvent):
    """A pass manager finished running on the circuit."""

    type: EventType = EventType.TRANSPILE_FINISHED
    duration: float = 0.0
    depth: int | None = None
    size: int | None = None
    phase: Literal["user", "backend"] = "user"
    circuit: Any = Field(default=None, exclude=True, repr=False)

    def describe(self) -> str:
        """Return the total duration and output circuit shape."""
        verb = (
            "backend compile finished"
            if self.phase == "backend"
            else "transpile finished"
        )
        return f"{verb} in {self.duration:.4f}s (depth={self.depth}, size={self.size})"


class ExecutionSubmittedEvent(TelemetryEvent):
    """The circuit was submitted to a backend or primitive."""

    type: EventType = EventType.EXECUTION_SUBMITTED
    backend: str | None = None
    shots: int | None = None
    job_id: str | None = None
    instrumented: bool = False

    def describe(self) -> str:
        """Return the target backend and shot count."""
        extra = " with statevector capture" if self.instrumented else ""
        return f"submitted to {self.backend} (shots={self.shots}){extra}"


class StateSnapshotEvent(TelemetryEvent):
    """The simulator state captured after one instruction."""

    type: EventType = EventType.STATE_SNAPSHOT
    step: int
    gate_index: int | None = None
    gate: str | None = None
    num_qubits: int | None = None
    statevector: Any = Field(default=None, exclude=True, repr=False)

    def amplitudes(self) -> list[complex]:
        """Return the statevector as a plain list of complex amplitudes."""
        if self.statevector is None:
            return []
        return [complex(z) for z in np.asarray(self.statevector).ravel()]

    def probabilities(self) -> list[float]:
        """Return measurement probabilities per basis state."""
        return [abs(z) ** 2 for z in self.amplitudes()]

    def describe(self) -> str:
        """Return which instruction the snapshot follows."""
        if self.gate is None:
            return "state snapshot (initial state)"
        return f"state snapshot after {self.gate} (instruction {self.gate_index})"


class ExecutionFinishedEvent(TelemetryEvent):
    """Execution results became available."""

    type: EventType = EventType.EXECUTION_FINISHED
    backend: str | None = None
    duration: float | None = None
    success: bool | None = None
    shots: int | None = None
    counts: dict[str, int] | None = None

    def describe(self) -> str:
        """Return the outcome, duration and counts summary."""
        text = f"finished on {self.backend}"
        if self.duration is not None:
            text += f" in {self.duration:.4f}s"
        if self.counts is not None:
            text += f", counts={self.counts}"
        return text


class NoteEvent(TelemetryEvent):
    """A diagnostic message from QisDebug itself, e.g. why capture was skipped."""

    type: EventType = EventType.NOTE
    message: str

    def describe(self) -> str:
        """Return the note text."""
        return self.message


class Timeline(BaseModel):
    """Everything QisDebug observed about one circuit, ordered by sequence.

    A timeline follows a circuit through construction, transpilation and
    execution. The transpile hook propagates the circuit id onto transpiled
    outputs, so all three phases land on the same timeline.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    circuit_id: str
    name: str | None = None
    num_qubits: int = 0
    num_clbits: int = 0
    qregs: dict[str, list[int]] = Field(default_factory=dict)
    cregs: dict[str, list[int]] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)
    events: list[TelemetryEvent] = Field(default_factory=list)

    _owning_store: "TelemetryStore | None" = PrivateAttr(default=None)

    def bind_store(self, store: "TelemetryStore") -> None:
        """Record which store owns this timeline, for internal suspension use."""
        self._owning_store = store

    @overload
    def filter(  # pyright: ignore[reportOverlappingOverload]
        self, event_type: Literal[EventType.GATE_ADDED], /
    ) -> list[GateEvent]: ...
    @overload
    def filter(
        self, event_type: Literal[EventType.TRANSPILE_STARTED], /
    ) -> list[TranspileStartedEvent]: ...
    @overload
    def filter(
        self, event_type: Literal[EventType.TRANSPILE_PASS], /
    ) -> list[TranspilePassEvent]: ...
    @overload
    def filter(
        self, event_type: Literal[EventType.TRANSPILE_FINISHED], /
    ) -> list[TranspileFinishedEvent]: ...
    @overload
    def filter(
        self, event_type: Literal[EventType.EXECUTION_SUBMITTED], /
    ) -> list[ExecutionSubmittedEvent]: ...
    @overload
    def filter(
        self, event_type: Literal[EventType.STATE_SNAPSHOT], /
    ) -> list[StateSnapshotEvent]: ...
    @overload
    def filter(
        self, event_type: Literal[EventType.EXECUTION_FINISHED], /
    ) -> list[ExecutionFinishedEvent]: ...
    @overload
    def filter(self, event_type: Literal[EventType.NOTE], /) -> list[NoteEvent]: ...
    @overload
    def filter(self, *types: EventType | str) -> list[TelemetryEvent]: ...
    def filter(self, *types: EventType | str) -> list[Any]:
        """Return events whose type matches any of ``types``."""
        wanted = {EventType(t) for t in types}
        return [e for e in self.events if e.type in wanted]

    @property
    def gate_events(self) -> list[GateEvent]:
        """Construction-time instruction events, in append order."""
        return [e for e in self.events if isinstance(e, GateEvent)]

    @property
    def transpile_events(
        self,
    ) -> list[TranspileStartedEvent | TranspilePassEvent | TranspileFinishedEvent]:
        """All transpilation-related events."""
        return cast(
            "list[TranspileStartedEvent | TranspilePassEvent | TranspileFinishedEvent]",
            self.filter(
                EventType.TRANSPILE_STARTED,
                EventType.TRANSPILE_PASS,
                EventType.TRANSPILE_FINISHED,
            ),
        )

    @property
    def execution_events(self) -> list[TelemetryEvent]:
        """All execution-related events, including state snapshots."""
        return self.filter(
            EventType.EXECUTION_SUBMITTED,
            EventType.STATE_SNAPSHOT,
            EventType.EXECUTION_FINISHED,
        )

    @property
    def snapshots(self) -> list[StateSnapshotEvent]:
        """State snapshot events ordered by step."""
        found = [e for e in self.events if isinstance(e, StateSnapshotEvent)]
        return sorted(found, key=lambda e: (e.seq, e.step))

    def snapshot_at_seq(self, seq: int) -> StateSnapshotEvent | None:
        """Return the state snapshot captured right after the gate event at ``seq``.

        Snapshots are numbered by ``step`` (position in the instrumented
        replay), while gates are numbered by ``seq`` (position in the whole
        timeline, interleaved with transpile/execution events) -- resolving
        one into the other otherwise means a caller has to count gate
        events by hand. Returns None if ``seq`` isn't a gate_added event's
        seq, or no snapshot was captured for it.
        """
        gates = self.gate_events
        index = next((i for i, e in enumerate(gates) if e.seq == seq), None)
        if index is None:
            return None
        step = index + 1
        return next((s for s in self.snapshots if s.step == step), None)

    def snapshot_upto(self, upto_seq: int | None) -> StateSnapshotEvent | None:
        """Return the most recent state snapshot at or before ``upto_seq``.

        ``None`` means "the final captured state" (no bound). Used both by
        the watch panel and by :func:`~qisdebug._analysis.diff_timelines` to
        resolve "the state as of this point in the replay".
        """
        candidates = (
            self.snapshots
            if upto_seq is None
            else [e for e in self.snapshots if e.seq <= upto_seq]
        )
        return candidates[-1] if candidates else None

    def execution_finished_upto(
        self, upto_seq: int | None
    ) -> ExecutionFinishedEvent | None:
        """Return the most recent execution-finished event at or before ``upto_seq``.

        ``None`` means "the final execution result" (no bound).
        """
        result = None
        for event in self.events:
            if isinstance(event, ExecutionFinishedEvent) and (
                upto_seq is None or event.seq <= upto_seq
            ):
                result = event
        return result

    def circuit_at(self, step: int | None = None) -> Any | None:
        """Rebuild the circuit as it looked after the first ``step`` gates.

        Returns None when Qiskit is unavailable or the recorded operation
        objects are missing (e.g. on a timeline restored from an export).
        """
        try:
            from qiskit.circuit import QuantumCircuit
        except ImportError:
            return None
        gates = self.gate_events
        if step is not None:
            gates = gates[:step]
        if any(e.operation is None for e in gates):
            return None

        def build() -> Any:
            circuit = QuantumCircuit(
                self.num_qubits, self.num_clbits, name=self.name or "qisdebug"
            )
            for event in gates:
                circuit.append(
                    event.operation,
                    [circuit.qubits[i] for i in event.qubits],
                    [circuit.clbits[i] for i in event.clbits],
                )
            return circuit

        try:
            if self._owning_store is not None:
                with self._owning_store.suspended():
                    return build()
            return build()
        except Exception:
            return None

    def summary(self) -> str:
        """Return a compact plain-text overview of the timeline."""
        lines = [
            f"Timeline '{self.name}' (id={self.circuit_id}, "
            f"{self.num_qubits} qubits, {self.num_clbits} clbits, {len(self.events)} events)"
        ]
        order = [t for t in EventType]
        counts = {t: 0 for t in order}
        for event in self.events:
            counts[event.type] = counts.get(event.type, 0) + 1
        for event_type in order:
            if counts.get(event_type):
                lines.append(f"  {event_type.value:<22} {counts[event_type]}")
        return "\n".join(lines)

    def export(self) -> dict:
        """Return a JSON-serializable dict of the full timeline.

        Statevectors are expanded into ``[[re, im], ...]`` pairs and circuit
        snapshots into their text drawings, so the result round-trips through
        ``json.dumps`` without custom encoders.
        """
        payload = self.model_dump(mode="json", exclude={"events"})
        payload["events"] = self.export_events()
        return payload

    def export_events(
        self,
        types: Iterable[EventType | str] | None = None,
        since_seq: int | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        """Return a filtered slice of events as JSON-serializable dicts.

        ``types`` restricts to the given event types, ``since_seq`` to events
        with a sequence number strictly greater than it (for incremental
        reads), and ``limit`` caps how many are returned, in timeline order.
        Serialization matches :meth:`export`.
        """
        wanted = {EventType(t) for t in types} if types is not None else None
        selected = [
            e
            for e in self.events
            if (wanted is None or e.type in wanted)
            and (since_seq is None or e.seq > since_seq)
        ]
        if limit is not None:
            selected = selected[:limit]
        return [self._export_event(e) for e in selected]

    def _export_event(self, event: TelemetryEvent) -> dict:
        """Serialize one event, expanding non-JSON fields."""
        data = event.model_dump(mode="json")
        if isinstance(event, StateSnapshotEvent) and event.statevector is not None:
            data["statevector"] = [[z.real, z.imag] for z in event.amplitudes()]
        if (
            isinstance(event, (TranspilePassEvent, TranspileFinishedEvent))
            and event.circuit is not None
        ):
            try:
                data["circuit_text"] = str(event.circuit.draw(output="text", fold=-1))
            except Exception:
                data["circuit_text"] = None
        return data

    def __repr__(self) -> str:
        return f"Timeline(name={self.name!r}, id={self.circuit_id!r}, events={len(self.events)})"
