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
    PropertySetChanges,
    StateSnapshotEvent,
    TelemetryEvent,
    Timeline,
    TranspilePassEvent,
)
from .._analysis import (
    DiffPoint,
    TimelineDiff,
    circuit_object_tracks,
    diff_timeline_points,
    track_keys_upto,
)

_MAX_PROPERTY_VALUE_LENGTH = 120

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

    A "track" is a distinct circuit lineage -- the circuit as constructed,
    and one key per transpile *run* (``"transpiled"``, then
    ``"transpiled#2"`` for a second ``transpile()`` call on the same
    circuit, and so on for ``"backend"`` runs) -- computed once by
    :func:`~qisdebug._analysis.circuit_object_tracks` and shared with
    :func:`~qisdebug._analysis.diff_timelines`'s ``track_a``/``track_b`` so
    the circuit panel and the diff picker always agree on what a track key
    means. Once a track has a value it is carried forward to every later
    event until a newer snapshot replaces it, so the circuit panel always
    shows the most recent known state instead of only appearing on the
    event that happened to produce it.

    Also returns, per event, which track was most recently touched (carried
    forward the same way), so callers can highlight/scroll to the circuit
    that is actually relevant to the currently selected event.
    """
    object_series, active_series = circuit_object_tracks(timeline)
    text_cache: dict[int, str] = {}
    text_series: list[dict[str, str]] = []
    for snapshot in object_series:
        rendered: dict[str, str] = {}
        for key, circuit in snapshot.items():
            cache_key = id(circuit)
            if cache_key not in text_cache:
                try:
                    text_cache[cache_key] = str(circuit.draw(output="text", fold=-1))
                except Exception:
                    continue
            rendered[key] = text_cache[cache_key]
        text_series.append(rendered)
    return text_series, active_series


def merge_timelines(
    timelines: list[Timeline],
) -> tuple[
    list[tuple[Timeline, TelemetryEvent]], list[dict[str, str]], list[str | None]
]:
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
        active_labels.append(
            track_label(f"{prefix}:{active_key}") if active_key else None
        )
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
    """Turn a ``circuit_name:track`` or ``circuit_name:track#run`` key into a label."""
    prefix, _, key = combined_key.rpartition(":")
    base, sep, run = key.partition("#")
    label = TRACK_LABELS.get(base, base)
    if sep:
        label = f"{label} #{run}"
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
    lines = [
        f"pass:     {event.pass_name}{suffix}",
        f"duration: {event.duration:.6f}s",
        f"depth:    {event.depth}   size: {event.size}   changed: {event.changed}",
        f"ops:      {event.ops}",
    ]
    if event.property_changes:
        lines.append("")
        lines.append(_render_property_changes(event.property_changes))
    return "\n".join(lines)


def _truncate_property_value(value: str) -> str:
    """Cap a property_set repr for display -- Layout/DAG reprs can be very long."""
    if len(value) <= _MAX_PROPERTY_VALUE_LENGTH:
        return value
    return value[:_MAX_PROPERTY_VALUE_LENGTH] + "..."


def _render_property_changes(changes: PropertySetChanges) -> str:
    """Render a property_set diff as +/-/~ lines, one change per line."""
    lines = ["property changes:"]
    for key, value in changes.added.items():
        lines.append(f"  + {key} = {_truncate_property_value(value)}")
    for key, value in changes.removed.items():
        lines.append(f"  - {key} = {_truncate_property_value(value)}")
    for key, (old, new) in changes.changed.items():
        lines.append(
            f"  ~ {key}: {_truncate_property_value(old)} -> {_truncate_property_value(new)}"
        )
    return "\n".join(lines)


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

        return _figure_to_data_uri(
            plot_bloch_multivector(np.asarray(event.statevector))
        )
    except Exception:
        return None


def render_bloch_bytes(event: StateSnapshotEvent) -> bytes | None:
    """Like :func:`render_bloch` but returns raw PNG bytes for GUI use."""
    if event.statevector is None:
        return None
    try:
        from qiskit.visualization import plot_bloch_multivector

        return _figure_to_png_bytes(
            plot_bloch_multivector(np.asarray(event.statevector))
        )
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
        name = (
            timeline.name or timeline.circuit_id
            if timeline is not None
            else self.circuit_id
        )
        return f"{name} — {self.kind} {self.ref}"


@dataclass(frozen=True)
class StageMark:
    """One pinned point in time across every circuit tracked so far.

    Pinned interactively (mark a/mark b) while stepping through a merged,
    possibly multi-circuit timeline in any frontend, mirroring
    :class:`WatchTarget`; not persisted on any :class:`~qisdebug._events.Timeline`
    itself. A *stage* is deliberately just "as of this global seq", not a
    single circuit -- construction and transpilation can both be mid-flight
    at the same point in time, so a stage can hold more than one circuit.
    Which circuits actually qualify is derived on demand from whichever
    timelines are loaded (:meth:`circuits_at`) rather than captured at mark
    time, so it stays correct even if more circuits appear afterwards. Two
    stages are diffed against each other with :func:`render_stage_diff`
    (one chosen circuit per side) or :func:`render_stage_overall_diff`
    (every circuit common to both).
    """

    seq: int

    def circuits_at(self, timelines_by_id: dict[str, Timeline]) -> list[Timeline]:
        """Every tracked circuit with at least one event at or before this stage."""
        return [
            timeline
            for timeline in timelines_by_id.values()
            if any(event.seq <= self.seq for event in timeline.events)
        ]

    def elements_at(self, timelines_by_id: dict[str, Timeline]) -> list[str]:
        """Every ``"<circuit>:<track>"`` element available at this stage.

        A circuit contributes one element per track it has by this point
        (see :func:`~qisdebug._analysis.circuit_object_tracks`): its
        construction track, plus one per transpile run -- a circuit
        transpiled twice offers both ``"bell:transpiled"`` and
        ``"bell:transpiled#2"`` as distinct, separately diffable elements,
        not just one ``"bell"``. This is exactly what :func:`render_stage_diff`
        and :func:`render_stage_overall_diff` diff against, so listing it is
        how a caller shows the user what's actually available to pick.
        """
        elements = []
        for timeline in self.circuits_at(timelines_by_id):
            prefix = timeline.name or timeline.circuit_id
            for key in track_keys_upto(timeline, self.seq):
                elements.append(f"{prefix}:{key}")
        return elements

    def label(self, timelines_by_id: dict[str, Timeline]) -> str:
        """Return e.g. ``"stage @ seq 12 (bell, bell_t)"``."""
        names = ", ".join(
            timeline.name or timeline.circuit_id
            for timeline in self.circuits_at(timelines_by_id)
        )
        suffix = f" ({names})" if names else ""
        return f"stage @ seq {self.seq}{suffix}"


def render_state_diff(diff: TimelineDiff) -> str:
    """Render a :class:`~qisdebug._analysis.TimelineDiff` as readable +/-/= text.

    The gate alignment prints as unified-diff-style lines over
    :meth:`~qisdebug._events.GateEvent.describe`; a ``"replace"`` entry
    prints as a ``-`` line followed by a ``+`` line. Followed by the
    summary line, then the statevector/counts diffs when present.
    """
    lines = [
        f"diff: {diff.side_label(diff.name_a, diff.circuit_id_a, diff.seq_a, diff.track_a)}"
        f"  vs  {diff.side_label(diff.name_b, diff.circuit_id_b, diff.seq_b, diff.track_b)}",
        "",
        "gates:",
    ]
    for entry in diff.gate_diff:
        if entry.op == "equal" and entry.a is not None:
            lines.append(f"    {entry.a.describe()}")
        else:
            if entry.a is not None:
                lines.append(f"  - {entry.a.describe()}")
            if entry.b is not None:
                lines.append(f"  + {entry.b.describe()}")
    lines.append("")
    lines.append(diff.summary_line())
    if diff.statevector_diff is not None:
        lines.append("")
        lines.append(diff.statevector_diff.describe())
    if diff.counts_diff is not None:
        lines.append("")
        lines.append(diff.counts_diff.describe())
    return "\n".join(lines)


def _find_circuit(name_or_id: str, circuits: list[Timeline]) -> Timeline | None:
    """Look up one of a stage's circuits by its name or circuit id."""
    for timeline in circuits:
        if name_or_id in (timeline.circuit_id, timeline.name):
            return timeline
    return None


def resolve_stage_element(
    stage: StageMark, timelines_by_id: dict[str, Timeline], choice: str | None
) -> tuple[Timeline, str] | None:
    """Pick one (circuit, track) element within a stage.

    ``choice`` may be an exact ``"circuit:track"`` element (see
    :meth:`StageMark.elements_at`), or a bare circuit name/id or bare track
    name that unambiguously narrows the stage down to one element. Left
    unset, the stage's only element resolves automatically. Returns
    ``None`` when the pick is ambiguous or matches nothing, so the caller
    can list :meth:`StageMark.elements_at` instead of silently guessing
    which circuit -- or which track of that circuit -- was meant.
    """
    elements = stage.elements_at(timelines_by_id)
    if choice is None:
        candidates = elements
    elif choice in elements:
        candidates = [choice]
    else:
        candidates = [
            element
            for element in elements
            if choice in element.split(":", 1)
        ]
    if len(candidates) != 1:
        return None
    name, _, track = candidates[0].partition(":")
    timeline = _find_circuit(name, stage.circuits_at(timelines_by_id))
    if timeline is None:
        return None
    return timeline, track


def render_stage_diff(
    stage_a: StageMark,
    stage_b: StageMark,
    timelines_by_id: dict[str, Timeline],
    element_a: str | None = None,
    element_b: str | None = None,
) -> str:
    """Render the full per-gate diff between one element from each stage.

    ``element_a``/``element_b`` pick which circuit *and which of its
    tracks* -- construction, or a specific transpile run -- to diff on each
    side (see :meth:`StageMark.elements_at`); left unset, a stage with
    exactly one element resolves automatically. Ambiguous or unresolved
    picks return a message listing that stage's options instead of
    guessing, since guessing wrong here would mean silently diffing the
    pre-transpile construction gates when a transpiled circuit was meant.
    The rendered header always names the track used (e.g. ``[transpiled]``)
    so the choice actually made is never ambiguous after the fact. For a
    birds-eye view across every element a stage holds, use
    :func:`render_stage_overall_diff` instead.
    """
    resolved_a = resolve_stage_element(stage_a, timelines_by_id, element_a)
    resolved_b = resolve_stage_element(stage_b, timelines_by_id, element_b)
    if resolved_a is None:
        options = ", ".join(stage_a.elements_at(timelines_by_id))
        return (
            f"stage A has multiple elements ({options}) -- "
            "specify which one, e.g. diff <a> <b>"
        )
    if resolved_b is None:
        options = ", ".join(stage_b.elements_at(timelines_by_id))
        return (
            f"stage B has multiple elements ({options}) -- "
            "specify which one, e.g. diff <a> <b>"
        )
    timeline_a, track_a = resolved_a
    timeline_b, track_b = resolved_b
    diff = diff_timeline_points(
        timeline_a,
        timeline_b,
        at_a=DiffPoint(stage_a.seq, track_a),
        at_b=DiffPoint(stage_b.seq, track_b),
    )
    return render_state_diff(diff)


def render_stage_overall_diff(
    stage_a: StageMark, stage_b: StageMark, timelines_by_id: dict[str, Timeline]
) -> str:
    """Render a one-line summary diff for every element common to both stages.

    Unlike :func:`render_stage_diff`, which drills into one chosen pair,
    this covers every element (matched by exact ``"circuit:track"`` key, so
    a circuit's construction and transpiled tracks are compared
    separately) present in both stages, each reduced to
    :meth:`~qisdebug._analysis.TimelineDiff.summary_line`. Each line is
    prefixed with the element key it came from, so which track a given
    summary refers to is always explicit.
    """
    common = sorted(
        set(stage_a.elements_at(timelines_by_id))
        & set(stage_b.elements_at(timelines_by_id))
    )
    if not common:
        return "(no element is present in both stages)"
    lines = []
    for element in common:
        name, _, track = element.partition(":")
        timeline_a = _find_circuit(name, stage_a.circuits_at(timelines_by_id))
        timeline_b = _find_circuit(name, stage_b.circuits_at(timelines_by_id))
        if timeline_a is None or timeline_b is None:
            continue
        diff = diff_timeline_points(
            timeline_a,
            timeline_b,
            at_a=DiffPoint(stage_a.seq, track),
            at_b=DiffPoint(stage_b.seq, track),
        )
        lines.append(f"{element}: {diff.summary_line()}")
    return "\n".join(lines)


_WatchBitKind = Literal["qubit", "clbit"]


def _watch_bits(
    target: WatchTarget, timeline: Timeline
) -> tuple[_WatchBitKind, tuple[int, ...]]:
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
    return timeline.snapshot_upto(upto_seq)


def _latest_execution_finished(
    timeline: Timeline, upto_seq: int
) -> ExecutionFinishedEvent | None:
    """Return the most recent execution-finished event at or before ``upto_seq``, if any."""
    return timeline.execution_finished_upto(upto_seq)


def _render_qubit_watch(snapshot: StateSnapshotEvent, qubits: tuple[int, ...]) -> str:
    """Render P(1) and the reduced Bloch vector for each watched qubit."""
    amplitudes = snapshot.amplitudes()
    if not amplitudes:
        return "(no state captured yet)"
    num_qubits = snapshot.num_qubits or int(np.log2(len(amplitudes)))
    lines = [f"state {snapshot.describe()}:"]
    for qubit in qubits:
        probability_one = _qubit_probability_one(amplitudes, qubit)
        x, y, z = (
            _clean(v) for v in _reduced_bloch_vector(amplitudes, qubit, num_qubits)
        )
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
    return sum(
        abs(amp) ** 2 for basis, amp in enumerate(amplitudes) if (basis >> qubit) & 1
    )


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
        ones = sum(
            n
            for bitstring, n in counts.items()
            if _clbit_value(bitstring, clbit) == "1"
        )
        lines.append(
            f"    clbit {clbit}: P(1)={ones / total:.4f}  ({ones}/{total} shots)"
        )
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
