"""Comparison arithmetic and cheap sanity checks over captured telemetry.

QDebug deliberately does not try to decide what a quantum program *should*
do -- that judgment belongs to whoever is debugging (a human, or an agent
via :mod:`qdebug._mcp`). What this module provides is the reliable
arithmetic for testing a caller-supplied expectation against what was
actually captured (:func:`compare_statevector`, :func:`compare_counts`),
plus :func:`find_anomalies`, a small fixed set of expectation-free checks
for states no correct program produces (non-normalized or non-finite
statevectors, failed executions, transpile blow-ups).

Pure functions over :class:`~qdebug._events.Timeline` data, with no
presentation concerns, so any consumer (the MCP tool layer, the CLI, the
GUI) calls the same code instead of re-deriving fidelity math or heuristics
independently. Mirrors the split already used for rendering: computation
lives here, presentation lives in the caller.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ._events import EventType, ExecutionFinishedEvent, StateSnapshotEvent, Timeline

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
        assert isinstance(event, ExecutionFinishedEvent)
        if event.success is False:
            anomalies.append(
                Anomaly(
                    severity="error", seq=event.seq, message=f"{event.describe()}: execution failed"
                )
            )
        elif event.counts is not None and not event.counts:
            anomalies.append(
                Anomaly(
                    severity="warning", seq=event.seq, message=f"{event.describe()}: empty counts"
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
