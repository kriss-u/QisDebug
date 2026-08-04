"""Frontend-agnostic data prep shared by every :class:`TimelineFrontend`.

Nothing here depends on ipywidgets, tkinter, or the terminal: it only turns
:class:`~qisdebug._events.Timeline` data into plain text/PNG bytes that each
frontend then presents in its own way.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Any, Iterable, Literal

import numpy as np

from .._events import (
    ExecutionFinishedEvent,
    GateEvent,
    StateSnapshotEvent,
    TelemetryEvent,
    Timeline,
    TranspileFinishedEvent,
    TranspilePassEvent,
)

TRACK_LABELS = {
    "circuit": "Circuit (construction)",
    "transpiled": "Circuit (transpiled)",
    "backend": "Circuit (backend compiled)",
}


def as_timelines(timelines: Timeline | Iterable[Timeline]) -> list[Timeline]:
    """Normalize a single timeline or an iterable of them into a list."""
    if isinstance(timelines, Timeline):
        return [timelines]
    return list(timelines)


def circuit_tracks(timeline: Timeline) -> tuple[list[dict[str, str]], list[str | None]]:
    """Return, per event, the latest known circuit drawing for each track.

    A "track" is a distinct circuit lineage (the circuit as constructed, the
    circuit as last seen mid-transpile, ...). Once a track has a value it is
    carried forward to every later event until a newer snapshot replaces it,
    so the circuit panel always shows the most recent known state instead of
    only appearing on the event that happened to produce it.

    Also returns, per event, which track was most recently touched (carried
    forward the same way), so callers can highlight/scroll to the circuit
    that is actually relevant to the currently selected event.
    """
    current: dict[str, str] = {}
    series: list[dict[str, str]] = []
    active_series: list[str | None] = []
    active_key: str | None = None
    gate_count = 0
    for event in timeline.events:
        if isinstance(event, GateEvent):
            gate_count += 1
            circuit = timeline.circuit_at(gate_count)
            if circuit is not None:
                try:
                    current["circuit"] = str(circuit.draw(output="text", fold=-1))
                    active_key = "circuit"
                except Exception:
                    pass
        elif (
            isinstance(event, (TranspilePassEvent, TranspileFinishedEvent))
            and event.circuit is not None
        ):
            key = "backend" if getattr(event, "phase", "user") == "backend" else "transpiled"
            try:
                current[key] = str(event.circuit.draw(output="text", fold=-1))
                active_key = key
            except Exception:
                pass
        series.append(dict(current))
        active_series.append(active_key)
    return series, active_series


def merge_timelines(
    timelines: list[Timeline],
) -> tuple[list[tuple[Timeline, TelemetryEvent]], list[dict[str, str]], list[str | None]]:
    """Interleave events from several timelines in global sequence order.

    ``seq`` is a process-wide monotonic counter (see
    :class:`~qisdebug._events.TelemetryEvent`), so sorting by it produces the
    order things actually happened across circuits, not just within one.
    Circuit tracks are computed independently per timeline (each carries
    forward its own state) and merged under a per-circuit-prefixed key so
    the panel can show every circuit's latest known state side by side.
    Also returns, per merged event, the display label of the track most
    relevant to that event (or ``None``), for scroll/highlight focus.
    """
    per_timeline_tracks = {t.circuit_id: circuit_tracks(t) for t in timelines}
    merged = sorted(
        ((t, event) for t in timelines for event in t.events),
        key=lambda pair: pair[1].seq,
    )
    cursors = {t.circuit_id: 0 for t in timelines}
    combined: dict[str, str] = {}
    series: list[dict[str, str]] = []
    active_labels: list[str | None] = []
    for source, _event in merged:
        prefix = source.name or source.circuit_id
        local, active_key = _advance_track_cursor(
            per_timeline_tracks[source.circuit_id], cursors, source.circuit_id
        )
        combined.update({f"{prefix}:{key}": text for key, text in local.items()})
        series.append(dict(combined))
        active_labels.append(track_label(f"{prefix}:{active_key}") if active_key else None)
    return merged, series, active_labels


def _advance_track_cursor(
    tracks_for_source: tuple[list[dict[str, str]], list[str | None]],
    cursors: dict[str, int],
    circuit_id: str,
) -> tuple[dict[str, str], str | None]:
    """Consume one step of a timeline's track series, advancing its cursor.

    Each timeline's ``circuit_tracks()`` series only has one entry per event
    *within that timeline*, so :func:`merge_timelines` must track how far
    into each timeline's own series it has read separately from its
    position in the interleaved, cross-timeline ``merged`` sequence.
    """
    series_for_source, active_for_source = tracks_for_source
    idx = cursors[circuit_id]
    cursors[circuit_id] += 1
    return series_for_source[idx], active_for_source[idx]


def track_label(combined_key: str) -> str:
    """Turn a ``circuit_name:track`` key into a display label."""
    prefix, _, key = combined_key.rpartition(":")
    label = TRACK_LABELS.get(key, key)
    return f"{prefix} — {label}" if prefix else label


def render_tracks_text(tracks: dict[str, str]) -> str:
    """Render the circuit panel as plain text for one point in the timeline."""
    if not tracks:
        return "(no circuit recorded yet)"
    sections = []
    for key, text in tracks.items():
        sections.append(f"-- {track_label(key)} --\n{text}")
    return "\n\n".join(sections)


def render_tracks_text_with_active(
    tracks: dict[str, str], active_label: str | None
) -> tuple[str, tuple[int, int] | None]:
    """Like :func:`render_tracks_text`, but also locates the active section.

    Returns the rendered text plus the 1-indexed ``(start_line, end_line)``
    span of the section matching ``active_label`` (the track most recently
    touched by the current event, per :func:`merge_timelines`), so a caller
    with a line-oriented widget can scroll to and highlight it. ``None`` if
    there are no tracks or no section matches.
    """
    if not tracks:
        return "(no circuit recorded yet)", None
    sections = []
    active_span = None
    line = 1
    for key, text in tracks.items():
        label = track_label(key)
        section = f"-- {label} --\n{text}"
        section_line_count = section.count("\n") + 1
        if active_label is not None and label == active_label:
            active_span = (line, line + section_line_count - 1)
        sections.append(section)
        line += section_line_count + 1
    return "\n\n".join(sections), active_span


def render_event(_timeline: Timeline, event: TelemetryEvent) -> str:
    """Return the detail panel text for one event.

    The circuit itself is rendered separately by the always-on circuit panel
    (see :func:`circuit_tracks`), so this only covers per-event metadata that
    panel doesn't show. ``_timeline`` is accepted for interface symmetry with
    callers that always have both on hand, even though this particular
    rendering doesn't need it.
    """
    if isinstance(event, GateEvent):
        return event.describe()
    if isinstance(event, StateSnapshotEvent):
        return format_statevector(event)
    if isinstance(event, TranspilePassEvent):
        return render_pass(event)
    lines = []
    for key, value in event.model_dump(mode="json").items():
        if key not in ("type", "seq", "timestamp"):
            lines.append(f"{key}: {value}")
    return "\n".join(lines) or event.describe()


def render_pass(event: TranspilePassEvent) -> str:
    """Show pass statistics for one transpiler pass."""
    suffix = "   (backend-internal)" if event.phase == "backend" else ""
    return (
        f"pass:     {event.pass_name}{suffix}\n"
        f"duration: {event.duration:.6f}s\n"
        f"depth:    {event.depth}   size: {event.size}   changed: {event.changed}\n"
        f"ops:      {event.ops}"
    )


def format_statevector(event: StateSnapshotEvent, max_terms: int = 16) -> str:
    """Render a snapshot as basis states with amplitudes and probabilities."""
    amplitudes = event.amplitudes()
    if not amplitudes:
        return event.describe()
    num_qubits = event.num_qubits or int(np.log2(len(amplitudes)))
    lines = [event.describe(), ""]
    shown = 0
    for basis, amplitude in enumerate(amplitudes):
        probability = abs(amplitude) ** 2
        if probability < 1e-12:
            continue
        if shown >= max_terms:
            lines.append("...")
            break
        label = format(basis, f"0{num_qubits}b")
        amplitude_text = f"{amplitude.real:+.4f}{amplitude.imag:+.4f}j"
        lines.append(f"|{label}>  {amplitude_text}   p={probability:.4f}")
        shown += 1
    return "\n".join(lines)


def render_histogram_ascii(counts: dict[str, int], width: int = 40) -> str:
    """Render execution counts as a simple text bar chart."""
    if not counts:
        return "(no counts)"
    total = sum(counts.values())
    peak = max(counts.values())
    lines = []
    for outcome, count in sorted(counts.items()):
        bar_len = max(1, round(count / peak * width)) if peak else 0
        filled = "#" * bar_len
        lines.append(f"{outcome:>12}  {filled:<{width}}  {count} ({count / total:.1%})")
    return "\n".join(lines)


def _figure_to_png_bytes(fig: Any) -> bytes:
    """Render a matplotlib figure to raw PNG bytes, then close it."""
    import matplotlib.pyplot as plt

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def _figure_to_data_uri(fig: Any) -> str:
    """Encode a matplotlib figure as a base64 PNG data URI."""
    encoded = base64.b64encode(_figure_to_png_bytes(fig)).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_histogram(event: ExecutionFinishedEvent) -> str | None:
    """Render an execution's counts as a histogram data URI, or None if unavailable.

    Requires matplotlib (``pip install "qisdebug[viz]"``); degrades to None
    rather than raising if it, or the counts themselves, are missing.
    """
    if not event.counts:
        return None
    try:
        from qiskit.visualization import plot_histogram

        return _figure_to_data_uri(plot_histogram(event.counts))
    except Exception:
        return None


def render_histogram_bytes(event: ExecutionFinishedEvent) -> bytes | None:
    """Like :func:`render_histogram` but returns raw PNG bytes for GUI use."""
    if not event.counts:
        return None
    try:
        from qiskit.visualization import plot_histogram

        return _figure_to_png_bytes(plot_histogram(event.counts))
    except Exception:
        return None


def render_bloch(event: StateSnapshotEvent) -> str | None:
    """Render a state snapshot as one Bloch sphere per qubit, or None if unavailable.

    Requires matplotlib (``pip install "qisdebug[viz]"``); degrades to None
    rather than raising if it, or the statevector itself, are missing.
    """
    if event.statevector is None:
        return None
    try:
        from qiskit.visualization import plot_bloch_multivector

        return _figure_to_data_uri(plot_bloch_multivector(np.asarray(event.statevector)))
    except Exception:
        return None


def render_bloch_bytes(event: StateSnapshotEvent) -> bytes | None:
    """Like :func:`render_bloch` but returns raw PNG bytes for GUI use."""
    if event.statevector is None:
        return None
    try:
        from qiskit.visualization import plot_bloch_multivector

        return _figure_to_png_bytes(plot_bloch_multivector(np.asarray(event.statevector)))
    except Exception:
        return None


def render_extra_visual(event: TelemetryEvent) -> tuple[str, str | None]:
    """Return (label, data_uri) for an event's supplementary visualization, if any."""
    if isinstance(event, ExecutionFinishedEvent) and event.counts:
        image = render_histogram(event)
        return ("Counts histogram", image) if image else ("", None)
    if isinstance(event, StateSnapshotEvent):
        image = render_bloch(event)
        return ("Bloch sphere", image) if image else ("", None)
    return "", None


def render_extra_visual_bytes(event: TelemetryEvent) -> tuple[str, bytes | None]:
    """Return (label, png_bytes) for an event's supplementary visualization, if any."""
    if isinstance(event, ExecutionFinishedEvent) and event.counts:
        image = render_histogram_bytes(event)
        return ("Counts histogram", image) if image else ("", None)
    if isinstance(event, StateSnapshotEvent):
        image = render_bloch_bytes(event)
        return ("Bloch sphere", image) if image else ("", None)
    return "", None


@dataclass(frozen=True)
class WatchTarget:
    """One qubit, clbit, or register a user pinned to watch during playback.

    Set interactively (watch/unwatch) while stepping through a timeline in
    any frontend; not persisted on the :class:`~qisdebug._events.Timeline`
    itself. ``ref`` is a qubit/clbit index as text for those two kinds, or a
    register name for ``"register"``.
    """

    circuit_id: str
    kind: Literal["qubit", "clbit", "register"]
    ref: str

    def label(self, timeline: Timeline | None) -> str:
        """Return a human-readable label, e.g. ``"bell — qubit 0"``."""
        name = timeline.name or timeline.circuit_id if timeline is not None else self.circuit_id
        return f"{name} — {self.kind} {self.ref}"


_WatchBitKind = Literal["qubit", "clbit"]


def _watch_bits(target: WatchTarget, timeline: Timeline) -> tuple[_WatchBitKind, tuple[int, ...]]:
    """Resolve a watch target to the concrete bit kind and indices it covers."""
    if target.kind == "qubit":
        return "qubit", (int(target.ref),)
    if target.kind == "clbit":
        return "clbit", (int(target.ref),)
    if target.ref in timeline.qregs:
        return "qubit", tuple(timeline.qregs[target.ref])
    return "clbit", tuple(timeline.cregs.get(target.ref, ()))


def render_watches(
    targets: list[WatchTarget], timelines_by_id: dict[str, Timeline], upto_seq: int
) -> str:
    """Render the tracked-value panel for every active watch, up through ``upto_seq``."""
    blocks = []
    for target in targets:
        timeline = timelines_by_id.get(target.circuit_id)
        if timeline is not None:
            blocks.append(render_watch(target, timeline, upto_seq))
    return "\n\n".join(blocks)


def render_watch(target: WatchTarget, timeline: Timeline, upto_seq: int) -> str:
    """Render one watch: gates that touched it so far, plus its latest known value."""
    bit_kind, indices = _watch_bits(target, timeline)
    lines = [f"* {target.label(timeline)}"]
    if not indices:
        lines.append("  (not found on this circuit)")
        return "\n".join(lines)
    index_set = set(indices)
    touches = [
        event
        for event in timeline.gate_events
        if event.seq <= upto_seq
        and index_set & set(event.qubits if bit_kind == "qubit" else event.clbits)
    ]
    lines.append(f"  gates touched: {len(touches)}")
    for event in touches[-5:]:
        lines.append(f"    seq {event.seq}: {event.describe()}")
    if bit_kind == "qubit":
        snapshot = _latest_snapshot(timeline, upto_seq)
        if snapshot is not None:
            lines.append(f"  {_render_qubit_watch(snapshot, indices)}")
    else:
        finished = _latest_execution_finished(timeline, upto_seq)
        if finished is not None and finished.counts:
            lines.append(f"  {_render_clbit_watch(finished, indices)}")
    return "\n".join(lines)


def _latest_snapshot(timeline: Timeline, upto_seq: int) -> StateSnapshotEvent | None:
    """Return the most recent state snapshot at or before ``upto_seq``, if any."""
    candidates = [e for e in timeline.snapshots if e.seq <= upto_seq]
    return candidates[-1] if candidates else None


def _latest_execution_finished(timeline: Timeline, upto_seq: int) -> ExecutionFinishedEvent | None:
    """Return the most recent execution-finished event at or before ``upto_seq``, if any."""
    result = None
    for event in timeline.events:
        if isinstance(event, ExecutionFinishedEvent) and event.seq <= upto_seq:
            result = event
    return result


def _render_qubit_watch(snapshot: StateSnapshotEvent, qubits: tuple[int, ...]) -> str:
    """Render P(1) and the reduced Bloch vector for each watched qubit."""
    amplitudes = snapshot.amplitudes()
    if not amplitudes:
        return "(no state captured yet)"
    num_qubits = snapshot.num_qubits or int(np.log2(len(amplitudes)))
    lines = [f"state {snapshot.describe()}:"]
    for qubit in qubits:
        probability_one = _qubit_probability_one(amplitudes, qubit)
        x, y, z = (_clean(v) for v in _reduced_bloch_vector(amplitudes, qubit, num_qubits))
        lines.append(
            f"    qubit {qubit}: P(1)={probability_one:.4f}   "
            f"bloch=({x:+.3f}, {y:+.3f}, {z:+.3f})"
        )
    return "\n".join(lines)


def _clean(value: float, epsilon: float = 1e-9) -> float:
    """Snap near-zero floats to exactly 0.0 so formatting never prints ``-0.000``."""
    return 0.0 if abs(value) < epsilon else value


def _qubit_probability_one(amplitudes: list[complex], qubit: int) -> float:
    """Return the marginal probability of measuring ``qubit`` as 1."""
    return sum(abs(amp) ** 2 for basis, amp in enumerate(amplitudes) if (basis >> qubit) & 1)


def _reduced_bloch_vector(
    amplitudes: list[complex], qubit: int, num_qubits: int
) -> tuple[float, float, float]:
    """Return the single-qubit Bloch vector from the partial trace over the rest."""
    psi = np.asarray(amplitudes, dtype=complex).reshape([2] * num_qubits)
    axis = num_qubits - 1 - qubit
    psi = np.moveaxis(psi, axis, 0).reshape(2, -1)
    rho = psi @ psi.conj().T
    x = 2 * rho[0, 1].real
    y = -2 * rho[0, 1].imag
    z = (rho[0, 0] - rho[1, 1]).real
    return float(x), float(y), float(z)


def _render_clbit_watch(event: ExecutionFinishedEvent, clbits: tuple[int, ...]) -> str:
    """Render the measured P(1) for each watched clbit from execution counts."""
    counts = event.counts or {}
    total = sum(counts.values())
    if not total:
        return "(no counts yet)"
    lines = [f"counts {event.describe()}:"]
    for clbit in clbits:
        ones = sum(n for bitstring, n in counts.items() if _clbit_value(bitstring, clbit) == "1")
        lines.append(f"    clbit {clbit}: P(1)={ones / total:.4f}  ({ones}/{total} shots)")
    return "\n".join(lines)


def _clbit_value(bitstring: str, clbit: int) -> str:
    """Extract one classical bit's value from a Qiskit counts key.

    Counts keys may contain register-separating spaces (e.g. ``"01 10"`` for
    two classical registers); strip them so indices line up with the flat
    clbit numbering used elsewhere, then index from the right since bit 0 is
    the rightmost character.
    """
    flat = bitstring.replace(" ", "")
    if clbit >= len(flat):
        return "?"
    return flat[len(flat) - 1 - clbit]
