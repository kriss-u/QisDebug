"""Interactive Jupyter widget for exploring QDebug timelines.

Deliberately decoupled from the instrumentation layers: it only reads the
public fields of :class:`~qdebug._events.Timeline`, so either side can be
replaced independently. Requires ipywidgets (``pip install "qdebug[viz]"``).
"""

from __future__ import annotations

import json
import tempfile
import webbrowser
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ._events import (
    GateEvent,
    StateSnapshotEvent,
    TelemetryEvent,
    Timeline,
    TranspilePassEvent,
)


TRACK_LABELS = {
    "circuit": "Circuit (construction)",
    "transpiled": "Circuit (transpiled)",
}


def _as_timelines(timelines: Timeline | Iterable[Timeline]) -> list[Timeline]:
    """Normalize a single timeline or an iterable of them into a list."""
    if isinstance(timelines, Timeline):
        return [timelines]
    return list(timelines)


def show(timelines: Timeline | Iterable[Timeline]) -> Any:
    """Build and display an interactive timeline explorer widget.

    Accepts a single :class:`Timeline` or several; events from multiple
    timelines are merged into one chronological sequence, with the circuit
    panel showing each circuit's most recently known state side by side.
    Returns the top-level widget so notebooks render it as the cell output.
    """
    try:
        import ipywidgets as widgets
    except ImportError as exc:
        raise ImportError(
            "qdebug.visualizer requires ipywidgets; install it with pip install 'qdebug[viz]'"
        ) from exc

    timelines = _as_timelines(timelines)
    merged, tracks = _merge_timelines(timelines)
    if not merged:
        ids = ", ".join(t.circuit_id for t in timelines)
        return widgets.HTML(value=f"<i>Timeline(s) {ids} have no events.</i>")

    slider = widgets.IntSlider(
        value=1,
        min=1,
        max=len(merged),
        description="event",
        continuous_update=False,
        layout=widgets.Layout(width="70%"),
    )
    header = widgets.HTML()
    body = widgets.HTML()
    circuits = widgets.HTML()

    def render(change: Any = None) -> None:
        source, event = merged[slider.value - 1]
        header.value = (
            f"<b>{slider.value}/{len(merged)}</b> &mdash; "
            f"<code>{_escape(str(source.name))}</code> &mdash; "
            f"<code>{event.type.value}</code> at {event.timestamp:.3f} &mdash; "
            f"{_escape(event.describe())}"
        )
        body.value = f"<pre style='line-height:1.2'>{_escape(_render_event(source, event))}</pre>"
        circuits.value = _render_tracks_html(tracks[slider.value - 1])

    slider.observe(render, names="value")
    render()
    names = ", ".join(_escape(str(t.name)) for t in timelines)
    title = widgets.HTML(
        value=f"<h4>QDebug timeline &mdash; {names} ({len(merged)} events)</h4>"
    )
    return widgets.VBox([title, slider, header, circuits, body])


def _circuit_tracks(timeline: Timeline) -> list[dict[str, str]]:
    """Return, per event, the latest known circuit drawing for each track.

    A "track" is a distinct circuit lineage (the circuit as constructed, the
    circuit as last seen mid-transpile, ...). Once a track has a value it is
    carried forward to every later event until a newer snapshot replaces it,
    so the circuit panel always shows the most recent known state instead of
    only appearing on the event that happened to produce it.
    """
    current: dict[str, str] = {}
    series: list[dict[str, str]] = []
    gate_count = 0
    for event in timeline.events:
        if isinstance(event, GateEvent):
            gate_count += 1
            circuit = timeline.circuit_at(gate_count)
            if circuit is not None:
                try:
                    current["circuit"] = str(circuit.draw(output="text"))
                except Exception:
                    pass
        elif isinstance(event, TranspilePassEvent) and event.circuit is not None:
            try:
                current["transpiled"] = str(event.circuit.draw(output="text"))
            except Exception:
                pass
        series.append(dict(current))
    return series


def _merge_timelines(
    timelines: list[Timeline],
) -> tuple[list[tuple[Timeline, TelemetryEvent]], list[dict[str, str]]]:
    """Interleave events from several timelines in global sequence order.

    ``seq`` is a process-wide monotonic counter (see
    :class:`~qdebug._events.TelemetryEvent`), so sorting by it produces the
    order things actually happened across circuits, not just within one.
    Circuit tracks are computed independently per timeline (each carries
    forward its own state) and merged under a per-circuit-prefixed key so
    the panel can show every circuit's latest known state side by side.
    """
    per_timeline_tracks = {t.circuit_id: _circuit_tracks(t) for t in timelines}
    merged = sorted(
        ((t, event) for t in timelines for event in t.events),
        key=lambda pair: pair[1].seq,
    )
    cursor = {t.circuit_id: 0 for t in timelines}
    combined: dict[str, str] = {}
    series: list[dict[str, str]] = []
    for source, _event in merged:
        local = per_timeline_tracks[source.circuit_id][cursor[source.circuit_id]]
        cursor[source.circuit_id] += 1
        prefix = source.name or source.circuit_id
        for key, text in local.items():
            combined[f"{prefix}:{key}"] = text
        series.append(dict(combined))
    return merged, series


def _render_tracks_html(tracks: dict[str, str]) -> str:
    """Render the circuit panel HTML for one point in the timeline."""
    if not tracks:
        return "<i>no circuit recorded yet</i>"
    sections = []
    for key, text in tracks.items():
        sections.append(
            f"<b>{_escape(_track_label(key))}</b>"
            f"<pre style='line-height:1.2'>{_escape(text)}</pre>"
        )
    return "".join(sections)


def _track_label(combined_key: str) -> str:
    """Turn a ``circuit_name:track`` key into a display label."""
    prefix, _, key = combined_key.rpartition(":")
    label = TRACK_LABELS.get(key, key)
    return f"{prefix} — {label}" if prefix else label


def show_browser(
    timelines: Timeline | Iterable[Timeline], path: str | None = None, *, open_browser: bool = True
) -> Path:
    """Render one or more timelines as a standalone HTML page and open it in a browser.

    Accepts a single :class:`Timeline` or several; when several are given,
    their events are merged into one chronological view (see
    :func:`_merge_timelines`) so you can see how circuits interleaved.
    Unlike :func:`show`, this has no ipywidgets/Jupyter dependency; it works
    from a plain Python script. Returns the path to the written HTML file.
    """
    timelines = _as_timelines(timelines)
    merged, tracks = _merge_timelines(timelines)
    frames = [
        {
            "header": (
                f"<code>{_escape(str(source.name))}</code> &mdash; "
                f"{event.type.value} &mdash; t={event.timestamp:.3f} &mdash; "
                f"{_escape(event.describe())}"
            ),
            "body": _escape(_render_event(source, event)),
            "circuits": {
                _track_label(key): _escape(text) for key, text in tracks[i].items()
            },
        }
        for i, (source, event) in enumerate(merged)
    ]
    names = ", ".join(_escape(str(t.name)) for t in timelines)
    html = _BROWSER_TEMPLATE.format(
        title=_escape(f"QDebug timeline — {names} ({len(merged)} events)"),
        frames=json.dumps(frames),
    )

    if path is None:
        handle = tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", prefix="qdebug_", delete=False, encoding="utf-8"
        )
        handle.write(html)
        handle.close()
        out_path = Path(handle.name)
    else:
        out_path = Path(path)
        out_path.write_text(html, encoding="utf-8")

    if open_browser:
        webbrowser.open(out_path.resolve().as_uri())
    return out_path


_BROWSER_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, sans-serif; margin: 2rem; color: #1a1a1a; }}
  h4 {{ margin-bottom: 0.5rem; }}
  input[type=range] {{ width: 70%; }}
  #header {{ margin: 0.75rem 0; font-size: 0.95rem; }}
  #circuits {{ display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem; }}
  #circuits > div {{ flex: 1 1 auto; min-width: 0; }}
  #circuits h5 {{ margin: 0 0 0.25rem; }}
  pre {{ background: #f5f5f5; padding: 1rem; border-radius: 6px; overflow-x: auto; line-height: 1.2; }}
</style>
</head>
<body>
  <h4>{title}</h4>
  <input id="slider" type="range" min="1" value="1">
  <div id="header"></div>
  <div id="circuits"></div>
  <pre id="body"></pre>
<script>
  const frames = {frames};
  const slider = document.getElementById("slider");
  const header = document.getElementById("header");
  const body = document.getElementById("body");
  const circuits = document.getElementById("circuits");
  slider.max = frames.length;
  function render() {{
    const f = frames[slider.value - 1];
    header.innerHTML = `<b>${{slider.value}}/${{frames.length}}</b> &mdash; ${{f.header}}`;
    body.innerHTML = f.body;
    const labels = Object.keys(f.circuits);
    circuits.innerHTML = labels.length
      ? labels.map(label => `<div><h5>${{label}}</h5><pre>${{f.circuits[label]}}</pre></div>`).join("")
      : "<i>no circuit recorded yet</i>";
  }}
  slider.addEventListener("input", render);
  render();
</script>
</body>
</html>
"""


def _render_event(timeline: Timeline, event: TelemetryEvent) -> str:
    """Return the detail panel text for one event.

    The circuit itself is rendered separately by the always-on circuit
    panel (see :func:`_circuit_tracks`), so this only covers per-event
    metadata that panel doesn't show.
    """
    if isinstance(event, GateEvent):
        return event.describe()
    if isinstance(event, StateSnapshotEvent):
        return format_statevector(event)
    if isinstance(event, TranspilePassEvent):
        return _render_pass(event)
    lines = []
    for key, value in event.model_dump(mode="json").items():
        if key not in ("type", "seq", "timestamp"):
            lines.append(f"{key}: {value}")
    return "\n".join(lines) or event.describe()


def _render_pass(event: TranspilePassEvent) -> str:
    """Show pass statistics for one transpiler pass."""
    return (
        f"pass:     {event.pass_name}\n"
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


def _escape(text: str) -> str:
    """Minimal HTML escaping for widget content."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
