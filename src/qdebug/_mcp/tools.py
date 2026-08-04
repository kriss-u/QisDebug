"""Agent-facing query functions over the live QDebug telemetry store.

Plain functions with JSON-serializable arguments and return values, and no
MCP protocol code, so :mod:`qdebug._mcp.server` can wrap each one as an MCP
tool and the test suite can call them directly without a protocol client.
All the actual work -- timeline lookup, rendering, statevector/counts
comparison, anomaly detection -- is delegated to :mod:`qdebug._store`,
:mod:`qdebug._events`, :mod:`qdebug._frontends._render` and
:mod:`qdebug._analysis`; nothing here re-implements that logic, the same
way :mod:`qdebug._frontends.cli` delegates to those modules instead of
recomputing what they already compute.
"""

from __future__ import annotations

import runpy
from typing import Any

import qdebug

from .. import _analysis
from .._events import EventType, Timeline
from .._frontends._render import format_statevector
from .._store import STORE


def _require_timeline(circuit_id: str) -> Timeline:
    """Look up a timeline by id, or raise a message an agent can act on."""
    timeline = STORE.get(circuit_id)
    if timeline is None:
        raise LookupError(f"no timeline for circuit id {circuit_id!r}")
    return timeline


def run_script(path: str) -> dict[str, Any]:
    """Run a Python script with QDebug instrumentation active and report results.

    Executes ``path`` as ``__main__`` with :func:`qdebug.activate` in
    effect, so any circuits it builds, transpiles or runs get recorded.
    Instrumentation is deactivated afterwards regardless of outcome. Returns
    the id/name of every timeline touched during the run plus anomalies
    found in each, so an agent can call this once and immediately see
    whether anything looks wrong instead of needing a second round trip.
    """
    before = {t.circuit_id for t in qdebug.timelines()}
    qdebug.activate()
    error: str | None = None
    try:
        runpy.run_path(path, run_name="__main__")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        qdebug.deactivate()
    touched = [t for t in qdebug.timelines() if t.circuit_id not in before]
    return {
        "error": error,
        "timelines": [
            {
                "circuit_id": t.circuit_id,
                "name": t.name,
                "num_qubits": t.num_qubits,
                "event_count": len(t.events),
                "anomalies": [a.model_dump() for a in _analysis.find_anomalies(t)],
            }
            for t in touched
        ],
    }


def list_timelines() -> list[dict[str, Any]]:
    """List every tracked timeline, most recently touched first."""
    return [
        {
            "circuit_id": t.circuit_id,
            "name": t.name,
            "num_qubits": t.num_qubits,
            "num_clbits": t.num_clbits,
            "event_count": len(t.events),
        }
        for t in qdebug.timelines()
    ]


def get_timeline_summary(circuit_id: str) -> str:
    """Return the compact plain-text event-count summary for one timeline."""
    return _require_timeline(circuit_id).summary()


def get_timeline(circuit_id: str) -> dict[str, Any]:
    """Return the full JSON-serializable export of one timeline.

    Includes every event (gates, transpile passes, state snapshots,
    execution results), with statevectors expanded to ``[[re, im], ...]``
    pairs and circuit snapshots to their text drawings.
    """
    return _require_timeline(circuit_id).export()


def get_circuit_text(circuit_id: str, step: int | None = None) -> str:
    """Return the circuit's text drawing after ``step`` gates, or fully built if omitted."""
    timeline = _require_timeline(circuit_id)
    circuit = timeline.circuit_at(step)
    if circuit is None:
        raise ValueError(f"circuit for {circuit_id!r} could not be rebuilt at step {step!r}")
    return str(circuit.draw(output="text", fold=-1))


def get_state_snapshot(circuit_id: str, step: int) -> str:
    """Return the formatted basis-state/amplitude breakdown for one snapshot step."""
    timeline = _require_timeline(circuit_id)
    for snapshot in timeline.snapshots:
        if snapshot.step == step:
            return format_statevector(snapshot)
    raise ValueError(f"no state snapshot at step {step} for circuit {circuit_id!r}")


def compare_statevector(
    circuit_id: str, step: int, expected_real: list[float], expected_imag: list[float]
) -> dict[str, Any]:
    """Compare a captured statevector snapshot against an expected one.

    ``expected_real``/``expected_imag`` give the real and imaginary parts of
    the expected amplitude for each basis state, ordered the way Qiskit
    orders statevectors (qubit 0 least significant).
    """
    timeline = _require_timeline(circuit_id)
    for snapshot in timeline.snapshots:
        if snapshot.step == step:
            expected = [complex(r, i) for r, i in zip(expected_real, expected_imag)]
            return _analysis.compare_statevector(snapshot, expected).model_dump()
    raise ValueError(f"no state snapshot at step {step} for circuit {circuit_id!r}")


def compare_counts(
    circuit_id: str, expected: dict[str, float], atol: float = 0.02
) -> dict[str, Any]:
    """Compare a circuit's measured counts against an expected probability distribution.

    ``atol`` is the per-outcome probability tolerance; widen it for
    low-shot-count runs where sampling noise alone can exceed the default.
    """
    timeline = _require_timeline(circuit_id)
    finished = timeline.filter(EventType.EXECUTION_FINISHED)
    if not finished or not finished[-1].counts:
        raise ValueError(f"no execution counts recorded for circuit {circuit_id!r}")
    return _analysis.compare_counts(finished[-1].counts, expected, atol=atol).model_dump()


def find_anomalies(circuit_id: str) -> list[dict[str, Any]]:
    """Scan a timeline for heuristic signs of a semantic or numerical bug."""
    timeline = _require_timeline(circuit_id)
    return [a.model_dump() for a in _analysis.find_anomalies(timeline)]


def clear() -> str:
    """Drop all recorded telemetry, without touching active instrumentation patches."""
    qdebug.clear()
    return "cleared"
