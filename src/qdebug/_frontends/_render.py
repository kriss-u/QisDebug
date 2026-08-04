"""Frontend-agnostic data prep shared by every :class:`TimelineFrontend`.

Nothing here depends on ipywidgets, tkinter, or the terminal: it only turns
:class:`~qdebug._events.Timeline` data into plain text/PNG bytes that each
frontend then presents in its own way.
"""

from __future__ import annotations

import base64
import io
from typing import Any, Iterable

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
    :class:`~qdebug._events.TelemetryEvent`), so sorting by it produces the
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

    Requires matplotlib (``pip install "qdebug[viz]"``); degrades to None
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

    Requires matplotlib (``pip install "qdebug[viz]"``); degrades to None
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


def escape(text: str) -> str:
    """Minimal HTML escaping for widget content."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
