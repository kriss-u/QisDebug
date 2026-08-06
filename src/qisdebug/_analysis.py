"""Comparison arithmetic and cheap sanity checks over captured telemetry.

QisDebug deliberately does not try to decide what a quantum program *should*
do -- that judgment belongs to whoever is debugging (a human, or an agent
via :mod:`qisdebug._mcp`). What this module provides is the reliable
arithmetic for testing a caller-supplied expectation against what was
actually captured (:func:`compare_statevector`, :func:`compare_counts`),
plus :func:`find_anomalies`, a small fixed set of expectation-free checks
for states no correct program produces (non-normalized or non-finite
statevectors, failed executions, transpile blow-ups).

Pure functions over :class:`~qisdebug._events.Timeline` data, with no
presentation concerns, so any consumer (the MCP tool layer, the CLI, the
GUI) calls the same code instead of re-deriving fidelity math or heuristics
independently. Mirrors the split already used for rendering: computation
lives here, presentation lives in the caller.
"""

from __future__ import annotations

import difflib
import itertools
from typing import Literal, Mapping, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ._events import EventType, GateEvent, StateSnapshotEvent, Timeline

Severity = str  # "info" | "warning" | "error"


class BasisMismatch(BaseModel):
    """One basis state whose amplitude differs from what was expected."""

    basis: int
    label: str
    actual: tuple[float, float]
    expected: tuple[float, float]
    delta: float


class StatevectorDiff(BaseModel):
    """Result of comparing a captured statevector against an expected one."""

    matched: bool
    fidelity: float
    max_amplitude_delta: float
    mismatches: list[BasisMismatch] = Field(default_factory=list)

    def describe(self) -> str:
        """Return a one-paragraph human-readable summary."""
        if self.matched:
            return f"statevector matches expected (fidelity={self.fidelity:.6f})"
        lines = [
            f"statevector MISMATCH (fidelity={self.fidelity:.6f}, "
            f"max amplitude delta={self.max_amplitude_delta:.6f})"
        ]
        for m in self.mismatches[:16]:
            lines.append(
                f"  |{m.label}>: actual={m.actual} expected={m.expected} delta={m.delta:.4f}"
            )
        if len(self.mismatches) > 16:
            lines.append(f"  ... and {len(self.mismatches) - 16} more")
        return "\n".join(lines)


class OutcomeDelta(BaseModel):
    """One measurement outcome whose observed frequency differs from expected."""

    outcome: str
    observed_probability: float
    expected_probability: float
    delta: float


class CountsDiff(BaseModel):
    """Result of comparing observed shot counts against an expected distribution."""

    matched: bool
    total_variation_distance: float
    deltas: list[OutcomeDelta] = Field(default_factory=list)

    def describe(self) -> str:
        """Return a one-paragraph human-readable summary."""
        if self.matched:
            return f"counts match expected (TVD={self.total_variation_distance:.4f})"
        lines = [f"counts MISMATCH (TVD={self.total_variation_distance:.4f})"]
        for d in sorted(self.deltas, key=lambda d: -abs(d.delta))[:16]:
            lines.append(
                f"  {d.outcome}: observed={d.observed_probability:.4f} "
                f"expected={d.expected_probability:.4f} delta={d.delta:+.4f}"
            )
        return "\n".join(lines)


class Anomaly(BaseModel):
    """One heuristic finding surfaced by :func:`find_anomalies`."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    severity: Severity
    seq: int
    message: str


def expand_sparse_statevector(
    expected: Mapping[str, Sequence[float]], num_qubits: int
) -> list[complex]:
    """Expand a sparse bitstring-labeled amplitude map into a dense statevector.

    ``expected`` maps a basis state's binary label (qubit 0 least
    significant, the same label :class:`BasisMismatch` reports) to its
    ``[re, im]`` amplitude; every label left out is assumed to be zero. This
    mirrors the shape :func:`compare_counts` already takes for outcomes, so
    a caller only names the basis states it has an expectation about
    instead of supplying all ``2 ** num_qubits`` amplitudes by hand.
    """
    dense: list[complex] = [0j] * (2**num_qubits)
    for label, amplitude in expected.items():
        if len(label) != num_qubits or any(c not in "01" for c in label):
            raise ValueError(
                f"label {label!r} must be a {num_qubits}-character binary string"
            )
        re, im = amplitude
        dense[int(label, 2)] = complex(re, im)
    return dense


def compare_statevector(
    snapshot: StateSnapshotEvent, expected: Sequence[complex], atol: float = 1e-6
) -> StatevectorDiff:
    """Compare a captured statevector snapshot against an expected one.

    ``expected`` must be the same length as the captured statevector, i.e.
    ``2 ** num_qubits`` amplitudes in the same qubit ordering Qiskit uses.
    Fidelity is ``|<expected|actual>|**2``; a basis state is flagged as a
    mismatch when its amplitude differs by more than ``atol``.
    """
    actual = np.asarray(snapshot.amplitudes(), dtype=complex)
    expected_arr = np.asarray(expected, dtype=complex)
    if actual.shape != expected_arr.shape:
        raise ValueError(
            f"expected {expected_arr.shape[0]} amplitudes, "
            f"snapshot has {actual.shape[0]} (mismatched qubit count)"
        )
    fidelity = float(abs(np.vdot(expected_arr, actual)) ** 2)
    deltas = np.abs(actual - expected_arr)
    max_delta = float(deltas.max()) if deltas.size else 0.0
    num_qubits = snapshot.num_qubits or int(np.log2(len(actual))) if len(actual) else 0
    mismatches = [
        BasisMismatch(
            basis=i,
            label=format(i, f"0{num_qubits}b"),
            actual=(float(actual[i].real), float(actual[i].imag)),
            expected=(float(expected_arr[i].real), float(expected_arr[i].imag)),
            delta=float(deltas[i]),
        )
        for i in range(len(actual))
        if deltas[i] > atol
    ]
    return StatevectorDiff(
        matched=not mismatches,
        fidelity=fidelity,
        max_amplitude_delta=max_delta,
        mismatches=mismatches,
    )


def compare_counts(
    observed: dict[str, int], expected: dict[str, float], atol: float = 0.02
) -> CountsDiff:
    """Compare observed shot counts against an expected probability distribution.

    ``expected`` maps bitstring outcomes to probabilities (need not sum to
    exactly 1.0 to allow for hand-rounded values). Outcomes present in only
    one side are treated as probability 0 on the other. Reports the total
    variation distance and flags outcomes whose delta exceeds ``atol``.
    """
    total = sum(observed.values())
    observed_probs = {k: v / total for k, v in observed.items()} if total else {}
    outcomes = set(observed_probs) | set(expected)
    deltas = []
    tvd = 0.0
    for outcome in outcomes:
        obs_p = observed_probs.get(outcome, 0.0)
        exp_p = expected.get(outcome, 0.0)
        delta = obs_p - exp_p
        tvd += abs(delta)
        if abs(delta) > atol:
            deltas.append(
                OutcomeDelta(
                    outcome=outcome,
                    observed_probability=obs_p,
                    expected_probability=exp_p,
                    delta=delta,
                )
            )
    tvd /= 2
    return CountsDiff(matched=not deltas, total_variation_distance=tvd, deltas=deltas)


def find_anomalies(timeline: Timeline) -> list[Anomaly]:
    """Scan a timeline for heuristic signs of a semantic or numerical bug.

    Catches things an agent would otherwise have to notice by eye: a
    statevector that stopped being normalized (a sign of a bad custom gate
    or unitary), NaN/inf amplitudes, a failed or empty execution, and a
    transpiler pass that blew up circuit depth. This is a fixed set of
    cheap, general checks, not a substitute for comparing against a
    known-expected state or distribution -- use :func:`compare_statevector`
    or :func:`compare_counts` for that.
    """
    anomalies: list[Anomaly] = []
    for event in timeline.snapshots:
        amplitudes = np.asarray(event.amplitudes(), dtype=complex)
        if amplitudes.size == 0:
            continue
        if not np.all(np.isfinite(amplitudes)):
            anomalies.append(
                Anomaly(
                    severity="error",
                    seq=event.seq,
                    message=f"{event.describe()}: non-finite amplitude",
                )
            )
            continue
        norm = float(np.sum(np.abs(amplitudes) ** 2))
        if abs(norm - 1.0) > 1e-6:
            anomalies.append(
                Anomaly(
                    severity="error",
                    seq=event.seq,
                    message=f"{event.describe()}: statevector norm is {norm:.6f}, expected 1.0",
                )
            )
    for event in timeline.filter(EventType.EXECUTION_FINISHED):
        if event.success is False:
            anomalies.append(
                Anomaly(
                    severity="error",
                    seq=event.seq,
                    message=f"{event.describe()}: execution failed",
                )
            )
        elif event.counts is not None and not event.counts:
            anomalies.append(
                Anomaly(
                    severity="warning",
                    seq=event.seq,
                    message=f"{event.describe()}: empty counts",
                )
            )
    prev_depth: int | None = None
    for event in timeline.transpile_events:
        depth = getattr(event, "depth", None)
        depth_spiked = (
            depth is not None
            and prev_depth is not None
            and depth > prev_depth * 3
            and depth - prev_depth > 8
        )
        if depth_spiked:
            anomalies.append(
                Anomaly(
                    severity="warning",
                    seq=event.seq,
                    message=f"{event.describe()}: depth jumped from {prev_depth} to {depth}",
                )
            )
        if depth is not None:
            prev_depth = depth
    return sorted(anomalies, key=lambda a: a.seq)


class GateDiffEntry(BaseModel):
    """One aligned step in a gate-sequence diff between two timelines.

    ``a``/``b`` are the two timelines' own :class:`GateEvent`, present
    depending on ``op``: both for ``"equal"``/``"replace"``, only ``a`` for
    ``"delete"`` (present in ``a``, missing from ``b``), only ``b`` for
    ``"insert"``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    op: Literal["equal", "insert", "delete", "replace"]
    a: GateEvent | None = None
    b: GateEvent | None = None


class TimelineDiff(BaseModel):
    """Result of comparing two points in time: gate alignment plus state deltas.

    Each side is "timeline as of seq_a/seq_b" -- ``None`` means the whole
    timeline / final state, so this covers both diffing two different
    circuits (typically both seqs ``None``, or matching points in each) and
    diffing two moments of the *same* circuit's replay (``circuit_id_a ==
    circuit_id_b``, different seqs).
    """

    circuit_id_a: str
    circuit_id_b: str
    name_a: str | None = None
    name_b: str | None = None
    seq_a: int | None = None
    seq_b: int | None = None
    gate_diff: list[GateDiffEntry] = Field(default_factory=list)
    statevector_diff: StatevectorDiff | None = None
    counts_diff: CountsDiff | None = None
    depth_a: int | None = None
    depth_b: int | None = None
    size_a: int | None = None
    size_b: int | None = None

    def summary_line(self) -> str:
        """Return a compact one-line fidelity/TVD/depth-delta summary."""
        parts = [f"{self._op_counts_text()}"]
        if self.statevector_diff is not None:
            parts.append(f"fidelity={self.statevector_diff.fidelity:.6f}")
        if self.counts_diff is not None:
            parts.append(f"TVD={self.counts_diff.total_variation_distance:.4f}")
        if self.depth_a is not None or self.depth_b is not None:
            parts.append(f"depth={self.depth_a}->{self.depth_b}")
        if self.size_a is not None or self.size_b is not None:
            parts.append(f"size={self.size_a}->{self.size_b}")
        return "  ".join(parts)

    def _op_counts_text(self) -> str:
        counts = {"equal": 0, "insert": 0, "delete": 0, "replace": 0}
        for entry in self.gate_diff:
            counts[entry.op] += 1
        return (
            f"gates: {counts['equal']} equal, {counts['insert']} inserted, "
            f"{counts['delete']} deleted, {counts['replace']} replaced"
        )

    def side_label(self, name: str | None, circuit_id: str, seq: int | None) -> str:
        """Return e.g. ``"'bell' @ seq 12"`` or ``"'bell' (final)"`` for one side."""
        bound = f"@ seq {seq}" if seq is not None else "(final)"
        return f"'{name or circuit_id}' {bound}"

    def describe(self) -> str:
        """Return a multi-paragraph human-readable summary."""
        lines = [
            f"diff {self.side_label(self.name_a, self.circuit_id_a, self.seq_a)} vs "
            f"{self.side_label(self.name_b, self.circuit_id_b, self.seq_b)}",
            self._op_counts_text(),
        ]
        if self.statevector_diff is not None:
            lines.append(self.statevector_diff.describe())
        if self.counts_diff is not None:
            lines.append(self.counts_diff.describe())
        if self.depth_a is not None or self.depth_b is not None:
            lines.append(f"depth: {self.depth_a} -> {self.depth_b}")
        if self.size_a is not None or self.size_b is not None:
            lines.append(f"size: {self.size_a} -> {self.size_b}")
        return "\n".join(lines)


def _gate_key(event: GateEvent) -> tuple[str, tuple[int, ...]]:
    """Key a gate event by name and qubits for sequence alignment."""
    return (event.gate, event.qubits)


def _gate_events_upto(timeline: Timeline, upto_seq: int | None) -> list[GateEvent]:
    """Return a timeline's gate events, optionally truncated to ``seq <= upto_seq``."""
    if upto_seq is None:
        return timeline.gate_events
    return [e for e in timeline.gate_events if e.seq <= upto_seq]


def _diff_gate_events(
    a_gates: list[GateEvent], b_gates: list[GateEvent]
) -> list[GateDiffEntry]:
    """Align two gate sequences with a standard sequence diff, keyed on gate + qubits."""
    matcher = difflib.SequenceMatcher(
        a=[_gate_key(e) for e in a_gates],
        b=[_gate_key(e) for e in b_gates],
        autojunk=False,
    )
    entries: list[GateDiffEntry] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for ai, bi in zip(range(i1, i2), range(j1, j2)):
                entries.append(GateDiffEntry(op="equal", a=a_gates[ai], b=b_gates[bi]))
        elif tag == "delete":
            entries.extend(
                GateDiffEntry(op="delete", a=a_gates[ai]) for ai in range(i1, i2)
            )
        elif tag == "insert":
            entries.extend(
                GateDiffEntry(op="insert", b=b_gates[bi]) for bi in range(j1, j2)
            )
        else:  # "replace"
            for ai, bi in itertools.zip_longest(range(i1, i2), range(j1, j2)):
                if ai is not None and bi is not None:
                    entries.append(
                        GateDiffEntry(op="replace", a=a_gates[ai], b=b_gates[bi])
                    )
                elif ai is not None:
                    entries.append(GateDiffEntry(op="delete", a=a_gates[ai]))
                elif bi is not None:
                    entries.append(GateDiffEntry(op="insert", b=b_gates[bi]))
    return entries


def _last_transpile_shape(
    timeline: Timeline, upto_seq: int | None
) -> tuple[int | None, int | None]:
    """Return (depth, size) from the most recent transpile event at/before ``upto_seq``
    that recorded a depth. ``None`` means no bound (the last such event overall).
    """
    for event in reversed(timeline.transpile_events):
        if event.depth is not None and (upto_seq is None or event.seq <= upto_seq):
            return event.depth, event.size
    return None, None


def _diff_statevectors(
    a: Timeline, b: Timeline, seq_a: int | None, seq_b: int | None, atol: float
) -> StatevectorDiff | None:
    """Compare the state snapshot at/before each side's seq bound, if both captured one."""
    a_snapshot = a.snapshot_upto(seq_a)
    b_snapshot = b.snapshot_upto(seq_b)
    if a_snapshot is None or b_snapshot is None:
        return None
    return compare_statevector(b_snapshot, a_snapshot.amplitudes(), atol=atol)


def _diff_counts(
    a: Timeline, b: Timeline, seq_a: int | None, seq_b: int | None, atol: float = 0.02
) -> CountsDiff | None:
    """Compare the execution counts at/before each side's seq bound, if both captured any."""
    a_finished = a.execution_finished_upto(seq_a)
    b_finished = b.execution_finished_upto(seq_b)
    a_counts = a_finished.counts if a_finished else None
    b_counts = b_finished.counts if b_finished else None
    if not a_counts or not b_counts:
        return None
    a_total = sum(a_counts.values())
    a_probs = {k: v / a_total for k, v in a_counts.items()} if a_total else {}
    return compare_counts(b_counts, a_probs, atol=atol)


def diff_timelines(
    a: Timeline,
    b: Timeline,
    *,
    seq_a: int | None = None,
    seq_b: int | None = None,
    atol: float = 1e-6,
) -> TimelineDiff:
    """Compare two points in time: gate-sequence alignment plus state deltas.

    ``seq_a``/``seq_b`` bound each side to "as of that event's seq" (a
    ``TelemetryEvent.seq``, e.g. from a currently-displayed event during
    playback, or from :func:`~qisdebug._mcp.tools.get_events`); ``None``
    (the default) means that side's whole timeline / final state. Passing
    the same :class:`Timeline` for ``a`` and ``b`` with different seqs diffs
    two moments of one circuit's own replay; passing two different
    timelines (with matching, differing, or unbounded seqs) diffs two
    circuits at a chosen or final point.

    Aligns each side's ``gate_added`` events up to its bound with a standard
    sequence diff (insert/delete/replace, keyed on gate name + qubits), so a
    construction-code refactor that reordered equivalent gates reads clearly
    instead of as a wall of mismatches. Separately reuses
    :func:`compare_statevector` (``atol`` is its per-amplitude mismatch
    threshold) and :func:`compare_counts` (at its own default per-outcome
    tolerance) on the state snapshot/counts at or before each bound when
    both sides have one (``a`` is treated as the expected/baseline side),
    plus a depth/size delta pulled from the last transpile event at or
    before each bound that recorded one.
    """
    depth_a, size_a = _last_transpile_shape(a, seq_a)
    depth_b, size_b = _last_transpile_shape(b, seq_b)
    return TimelineDiff(
        circuit_id_a=a.circuit_id,
        circuit_id_b=b.circuit_id,
        name_a=a.name,
        name_b=b.name,
        seq_a=seq_a,
        seq_b=seq_b,
        gate_diff=_diff_gate_events(
            _gate_events_upto(a, seq_a), _gate_events_upto(b, seq_b)
        ),
        statevector_diff=_diff_statevectors(a, b, seq_a, seq_b, atol),
        counts_diff=_diff_counts(a, b, seq_a, seq_b),
        depth_a=depth_a,
        depth_b=depth_b,
        size_a=size_a,
        size_b=size_b,
    )
