"""Agent-facing debugging session over the live QisDebug telemetry store.

Plain functions with JSON-serializable arguments and return values, and no
MCP protocol code, so :mod:`qisdebug._mcp.server` can wrap each one as an MCP
tool and the test suite can call them directly without a protocol client.

The division of labor is deliberate: QisDebug extracts ground truth from a
run (gate history, mid-execution statevectors, transpile snapshots, counts)
and does the arithmetic of checking a supplied expectation against it; the
agent decides where to scope instrumentation, which qubits and registers to
watch, what the program *should* have produced, and what the discrepancy
means. All the actual work -- timeline lookup, rendering, comparison,
anomaly scanning -- is delegated to :mod:`qisdebug._store`,
:mod:`qisdebug._events`, :mod:`qisdebug._frontends._render` and
:mod:`qisdebug._analysis`; nothing here re-implements that logic, the same
way :mod:`qisdebug._frontends.cli` delegates to those modules instead of
recomputing what they already compute.
"""

from __future__ import annotations

import json
import runpy
import sys
from typing import Any, Literal, Mapping, Sequence

from .. import _analysis
from .._debug import QisDebug
from .._errors import TimelineNotFoundError
from .._events import EventType, StateSnapshotEvent, Timeline
from .._frontends._render import WatchTarget, format_statevector, render_watch

WatchKind = Literal["qubit", "clbit", "register", "track"]

# One QisDebug instance scopes the whole MCP server process, matching the
# documented agent workflow: run_script's auto_activate controls whether
# *this* instance's start()/end() bracket the script, or whether the script
# brackets itself.
_debug = QisDebug()
_watches: list[WatchTarget] = []


def _require_timeline(circuit_id: str) -> Timeline:
    """Look up a timeline by id, or raise a message an agent can act on."""
    timeline = _debug.store.get(circuit_id)
    if timeline is None:
        raise TimelineNotFoundError(f"no timeline for circuit id {circuit_id!r}")
    return timeline


def run_script(
    path: str, argv: list[str] | None = None, auto_activate: bool = True
) -> dict[str, Any]:
    """Run a Python script under QisDebug instrumentation and report what was captured.

    Executes ``path`` as ``__main__`` with ``sys.argv`` set to ``[path,
    *argv]``. With ``auto_activate`` (the default) this server's
    instrumentation session covers the whole script. To scope debugging to a
    suspect region instead, call this with ``auto_activate=False`` -- this
    tool leaves its own session untouched, and the target script is
    responsible for its own scoped ``qisdebug.QisDebug()`` instance (its
    timelines won't appear in this call's return value; have the script
    export or print what it captured).
    """
    before = {t.circuit_id for t in _debug.timelines()}
    if auto_activate:
        _debug.start()
    error: str | None = None
    saved_argv = sys.argv
    sys.argv = [path, *(argv or [])]
    try:
        runpy.run_path(path, run_name="__main__")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        sys.argv = saved_argv
        if auto_activate:
            _debug.end()
    touched = [t for t in _debug.timelines() if t.circuit_id not in before]
    return {
        "error": error,
        "timelines": [
            {
                "circuit_id": t.circuit_id,
                "name": t.name,
                "num_qubits": t.num_qubits,
                "num_clbits": t.num_clbits,
                "qregs": t.qregs,
                "cregs": t.cregs,
                "event_count": len(t.events),
                "anomalies": [a.model_dump() for a in _analysis.find_anomalies(t)],
            }
            for t in touched
        ],
    }


def list_timelines() -> list[dict[str, Any]]:
    """List every tracked timeline, most recently touched first.

    Includes register layouts (name to flat bit indices) so watch targets
    can be chosen without a separate lookup.
    """
    return [
        {
            "circuit_id": t.circuit_id,
            "name": t.name,
            "num_qubits": t.num_qubits,
            "num_clbits": t.num_clbits,
            "qregs": t.qregs,
            "cregs": t.cregs,
            "event_count": len(t.events),
        }
        for t in _debug.timelines()
    ]


def get_timeline_summary(circuit_id: str) -> str:
    """Return the compact plain-text event-count summary for one timeline."""
    return _require_timeline(circuit_id).summary()


def get_timeline(circuit_id: str) -> dict[str, Any]:
    """Return the full JSON-serializable export of one timeline.

    Includes every event (gates, transpile passes, state snapshots,
    execution results), with statevectors expanded to ``[[re, im], ...]``
    pairs and circuit snapshots to their text drawings. For large timelines
    prefer :func:`get_events`, which filters and pages.
    """
    return _require_timeline(circuit_id).export()


def get_events(
    circuit_id: str,
    types: list[str] | None = None,
    since_seq: int | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Return a filtered slice of one timeline's events as structured dicts.

    ``types`` restricts to the given event type names (``"gate_added"``,
    ``"transpile_pass"``, ``"state_snapshot"``, ``"execution_finished"``,
    ...), ``since_seq`` to events after that sequence number (for
    incremental reads while narrowing in on a region), and ``limit`` caps
    the result size.
    """
    return _require_timeline(circuit_id).export_events(
        types=types, since_seq=since_seq, limit=limit
    )


def get_circuit_text(circuit_id: str, step: int | None = None) -> str:
    """Return the circuit's text drawing after ``step`` gates, or fully built if omitted."""
    timeline = _require_timeline(circuit_id)
    circuit = timeline.circuit_at(step)
    if circuit is None:
        raise ValueError(
            f"circuit for {circuit_id!r} could not be rebuilt at step {step!r}"
        )
    return str(circuit.draw(output="text", fold=-1))


def get_property_set(circuit_id: str, seq: int) -> dict[str, str]:
    """Return the full property_set snapshot captured at one transpile pass.

    ``seq`` is a transpile_pass event's seq (from :func:`get_events`), for
    the property_set as it stood right after that pass ran. Requires that
    pass to have captured a snapshot -- governed by ``capture_property_set``
    ("changed" only keeps snapshots for passes whose property_set changed;
    call ``set_configuration(overrides={"capture_property_set": "all"})``
    before a run to guarantee every pass is available).
    """
    timeline = _require_timeline(circuit_id)
    event = next(
        (e for e in timeline.filter(EventType.TRANSPILE_PASS) if e.seq == seq), None
    )
    if event is None:
        raise ValueError(
            f"no transpile_pass event at seq {seq} for circuit {circuit_id!r}"
        )
    if event.property_set is None:
        raise ValueError(
            f"no property_set snapshot captured for the pass at seq {seq}; "
            "set capture_property_set='all' before the run to guarantee capture"
        )
    return event.property_set


def _resolve_snapshot(
    timeline: Timeline, step: int | None, seq: int | None
) -> StateSnapshotEvent:
    """Look up one state snapshot by ``step`` or by the ``seq`` of the gate it follows.

    Exactly one of ``step``/``seq`` may be given; neither means "the final
    captured state". ``seq`` saves the caller from converting a
    ``get_events`` seq into the corresponding snapshot step by hand.
    """
    if step is not None and seq is not None:
        raise ValueError("pass step or seq, not both")
    snapshots = timeline.snapshots
    if not snapshots:
        raise ValueError(
            f"no state snapshots recorded for circuit {timeline.circuit_id!r}"
        )
    if seq is not None:
        snapshot = timeline.snapshot_at_seq(seq)
        if snapshot is None:
            raise ValueError(
                f"no state snapshot for the gate at seq {seq} on circuit "
                f"{timeline.circuit_id!r}; seq must be a gate_added event's seq"
            )
        return snapshot
    if step is None:
        return snapshots[-1]
    matches = [s for s in snapshots if s.step == step]
    if not matches:
        available = sorted(s.step for s in snapshots)
        raise ValueError(
            f"no state snapshot at step {step} for circuit {timeline.circuit_id!r}; "
            f"available steps: {available}"
        )
    return matches[0]


def get_state_snapshot(
    circuit_id: str, step: int | None = None, seq: int | None = None
) -> dict[str, Any]:
    """Return the captured simulator state at one point mid-execution.

    Select the snapshot with ``step`` (0 is the initial state, N the state
    after the Nth instrumented instruction) or ``seq`` (the seq of a
    gate_added event from :func:`get_events`, for the state right after
    that gate); omit both for the final captured state. Returns the raw
    amplitudes as ``[[re, im], ...]`` pairs, per-basis-state probabilities,
    and a human-readable rendering.
    """
    timeline = _require_timeline(circuit_id)
    snapshot = _resolve_snapshot(timeline, step, seq)
    amplitudes = snapshot.amplitudes()
    return {
        "step": snapshot.step,
        "seq": snapshot.seq,
        "gate": snapshot.gate,
        "gate_index": snapshot.gate_index,
        "num_qubits": snapshot.num_qubits,
        "amplitudes": [[z.real, z.imag] for z in amplitudes],
        "probabilities": snapshot.probabilities(),
        "formatted": format_statevector(snapshot),
    }


def add_watch(circuit_id: str, kind: WatchKind, ref: int | str) -> str:
    """Pin a qubit, clbit, register, or circuit track to watch across the
    timeline.

    ``ref`` is a bit index for ``"qubit"``/``"clbit"``, a register name for
    ``"register"`` (register layouts appear in :func:`list_timelines`), or a
    track key for ``"track"`` (e.g. ``"transpiled"``, ``"backend"`` --
    available tracks appear in :func:`get_timeline`/:func:`get_events` via
    the transpile events, or can be discovered with
    :func:`~qisdebug._analysis.track_keys_upto`). A qubit/clbit/register
    watch reports, at any point in the timeline, which gates have touched it
    on *every* circuit lineage (construction, transpiled, backend-compiled)
    and its latest known value -- marginal P(1) and Bloch vector for qubits,
    measured P(1) for clbits. A track watch instead reports that lineage's
    latest transpile pass stats (depth/size/ops/property changes). Read them
    with :func:`read_watches`.
    """
    timeline = _require_timeline(circuit_id)
    ref_text = str(ref)
    if kind in ("qubit", "clbit"):
        if not ref_text.isdigit():
            raise ValueError(f"{kind} ref must be a bit index, got {ref!r}")
        width = timeline.num_qubits if kind == "qubit" else timeline.num_clbits
        if int(ref_text) >= width:
            raise ValueError(
                f"{kind} {ref_text} out of range for circuit {circuit_id!r} "
                f"({width} {kind}s)"
            )
    elif kind == "register":
        known = set(timeline.qregs) | set(timeline.cregs)
        if ref_text not in known:
            raise ValueError(
                f"no register {ref_text!r} on circuit {circuit_id!r}; "
                f"available: {sorted(known)}"
            )
    else:
        known_tracks = set(_analysis.track_keys_upto(timeline, None))
        if ref_text not in known_tracks:
            raise ValueError(
                f"no track {ref_text!r} on circuit {circuit_id!r}; "
                f"available: {sorted(known_tracks)}"
            )
    target = WatchTarget(circuit_id=circuit_id, kind=kind, ref=ref_text)
    if target not in _watches:
        _watches.append(target)
    return f"watching {target.label(timeline)}"


def remove_watch(circuit_id: str, kind: WatchKind, ref: int | str) -> str:
    """Remove one previously added watch."""
    target = WatchTarget(circuit_id=circuit_id, kind=kind, ref=str(ref))
    if target not in _watches:
        raise ValueError(f"not watching {kind} {ref!r} on circuit {circuit_id!r}")
    _watches.remove(target)
    return f"unwatched {kind} {target.ref}"


def list_watches() -> list[dict[str, str]]:
    """List every active watch target."""
    return [
        {
            "circuit_id": t.circuit_id,
            "kind": t.kind,
            "ref": t.ref,
            "label": t.label(_debug.store.get(t.circuit_id)),
        }
        for t in _watches
    ]


def read_watches(upto_seq: int | None = None) -> list[dict[str, str]]:
    """Report every watched target's state as of a point in the timeline.

    ``upto_seq`` bounds the read to events at or before that sequence
    number, so watch values can be inspected mid-execution ("what was qubit
    1 just before the cx at seq 12?"); omit it for the latest state.
    """
    reports = []
    for target in _watches:
        timeline = _debug.store.get(target.circuit_id)
        if timeline is None:
            text = "(timeline no longer tracked)"
        else:
            bound = upto_seq
            if bound is None:
                bound = max((e.seq for e in timeline.events), default=0)
            text = render_watch(target, timeline, bound)
        reports.append(
            {
                "circuit_id": target.circuit_id,
                "kind": target.kind,
                "ref": target.ref,
                "text": text,
            }
        )
    return reports


def compare_statevector(
    circuit_id: str,
    expected: Mapping[str, Sequence[float]],
    step: int | None = None,
    seq: int | None = None,
) -> dict[str, Any]:
    """Compare a captured statevector snapshot against a supplied expected state.

    Select the snapshot with ``step`` (0 is the initial state, N the state
    after the Nth instrumented instruction) or ``seq`` (the seq of a
    gate_added event from :func:`get_events`, for the state right after
    that gate); omit both for the final captured state.

    ``expected`` maps a basis state's binary label (qubit 0 least
    significant, e.g. ``"01"``, matching the ``label`` field in the
    returned mismatches) to its ``[re, im]`` amplitude; any label left out
    is assumed to be zero, so only the basis states under suspicion need
    to be named. Returns the fidelity and every basis state whose
    amplitude differs, so the caller judges whether the discrepancy is the
    bug or noise.
    """
    timeline = _require_timeline(circuit_id)
    snapshot = _resolve_snapshot(timeline, step, seq)
    dense = _analysis.expand_sparse_statevector(expected, snapshot.num_qubits or 0)
    return _analysis.compare_statevector(snapshot, dense).model_dump()


def compare_counts(
    circuit_id: str, expected: dict[str, float], atol: float = 0.02
) -> dict[str, Any]:
    """Compare a circuit's measured counts against a supplied expected distribution.

    ``expected`` maps bitstring outcomes to probabilities. ``atol`` is the
    per-outcome tolerance; widen it for low-shot-count runs where sampling
    noise alone can exceed the default.
    """
    timeline = _require_timeline(circuit_id)
    finished = timeline.filter(EventType.EXECUTION_FINISHED)
    if not finished or not finished[-1].counts:
        raise ValueError(f"no execution counts recorded for circuit {circuit_id!r}")
    return _analysis.compare_counts(
        finished[-1].counts, expected, atol=atol
    ).model_dump()


def diff_timelines(
    circuit_id_a: str,
    circuit_id_b: str,
    seq_a: int | None = None,
    seq_b: int | None = None,
    atol: float = 1e-6,
) -> dict[str, Any]:
    """Compare two points in time: gate-sequence alignment plus state deltas.

    ``seq_a``/``seq_b`` bound each side to "as of that event's seq" (from
    :func:`get_events`); omit either to mean that circuit's whole timeline /
    final state. Passing the same circuit_id for both with different seqs
    diffs two moments of one circuit's own replay (e.g. the state right
    after gate X vs. the state right after gate Y); passing two different
    circuit_ids diffs two circuits, at a chosen or final point each.

    Aligns each side's gate_added events (up to its seq bound) with a
    standard sequence diff (insert/delete/replace, keyed on gate name +
    qubits) -- useful for confirming a construction-code refactor is
    equivalent, or spotting where it isn't. When both sides have a state
    snapshot or execution counts at or before their bound, also reports
    fidelity (``atol`` is the per-amplitude mismatch threshold) or total
    variation distance (at compare_counts's own default per-outcome
    tolerance), treating side A as the baseline/expected side. Also reports
    each side's transpiled depth/size as of its bound, when available.
    """
    timeline_a = _require_timeline(circuit_id_a)
    timeline_b = _require_timeline(circuit_id_b)
    return _analysis.diff_timelines(
        timeline_a, timeline_b, seq_a=seq_a, seq_b=seq_b, atol=atol
    ).model_dump()


def find_anomalies(circuit_id: str) -> list[dict[str, Any]]:
    """Run the expectation-free sanity scan over one timeline.

    Flags states no correct program produces: non-normalized or non-finite
    statevectors, failed or empty executions, and transpile depth
    blow-ups. A clean scan does not mean the program is correct -- semantic
    bugs need an expectation, via :func:`compare_statevector` or
    :func:`compare_counts`.
    """
    timeline = _require_timeline(circuit_id)
    return [a.model_dump() for a in _analysis.find_anomalies(timeline)]


def save_timeline(circuit_id: str, path: str) -> dict[str, Any]:
    """Write one timeline's full JSON export to a file.

    Persists the collected evidence past the server session (and past the
    store's LRU eviction), so a debugging finding can be kept alongside the
    code it explains or diffed against a later run.
    """
    timeline = _require_timeline(circuit_id)
    payload = timeline.export()
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return {"path": path, "events": len(payload["events"])}


def get_configuration() -> dict[str, Any]:
    """Return the current instrumentation configuration."""
    return _debug.config.model_dump(mode="json")


def set_configuration(overrides: dict[str, Any]) -> dict[str, Any]:
    """Adjust instrumentation limits and switches before a run.

    Accepts any :class:`~qisdebug.QisDebugConfig` field, e.g. raising
    ``max_statevector_qubits`` for a wide circuit, or setting
    ``transpile_snapshots`` to ``"all"`` to snapshot every pass while
    hunting a transpiler-introduced bug. Invalid keys or values raise a
    validation error and leave the configuration unchanged.
    """
    return _debug.configure(**overrides).model_dump(mode="json")


def clear() -> str:
    """Drop all recorded telemetry and watches, without touching active patches."""
    _debug.clear()
    _watches.clear()
    return "cleared"
