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
from typing import Any

import numpy as np

from ._events import (
    GateEvent,
    StateSnapshotEvent,
    TelemetryEvent,
    Timeline,
    TranspilePassEvent,
)


def show(timeline: Timeline) -> Any:
    """Build and display an interactive timeline explorer widget.

    Returns the top-level widget so notebooks render it as the cell output.
    """
    try:
        import ipywidgets as widgets
    except ImportError as exc:
        raise ImportError(
            "qdebug.visualizer requires ipywidgets; install it with pip install 'qdebug[viz]'"
        ) from exc

    if not timeline.events:
        return widgets.HTML(value=f"<i>Timeline {timeline.circuit_id} has no events.</i>")

    slider = widgets.IntSlider(
        value=1,
        min=1,
        max=len(timeline.events),
        description="event",
        continuous_update=False,
        layout=widgets.Layout(width="70%"),
    )
    header = widgets.HTML()
    body = widgets.HTML()

    def render(change: Any = None) -> None:
        event = timeline.events[slider.value - 1]
        header.value = (
            f"<b>{slider.value}/{len(timeline.events)}</b> &mdash; "
            f"<code>{event.type.value}</code> at {event.timestamp:.3f} &mdash; "
            f"{_escape(event.describe())}"
        )
        body.value = f"<pre style='line-height:1.2'>{_escape(_render_event(timeline, event))}</pre>"

    slider.observe(render, names="value")
    render()
    title = widgets.HTML(
        value=f"<h4>QDebug timeline &mdash; {_escape(str(timeline.name))} "
        f"({timeline.num_qubits} qubits, {len(timeline.events)} events)</h4>"
    )
    return widgets.VBox([title, slider, header, body])


def show_browser(timeline: Timeline, path: str | None = None, *, open_browser: bool = True) -> Path:
    """Render the timeline as a standalone HTML page and open it in a browser.

    Unlike :func:`show`, this has no ipywidgets/Jupyter dependency; it works
    from a plain Python script. Returns the path to the written HTML file.
    """
    frames = [
        {
            "header": (
                f"{event.type.value} &mdash; t={event.timestamp:.3f} &mdash; "
                f"{_escape(event.describe())}"
            ),
            "body": _escape(_render_event(timeline, event)),
        }
        for event in timeline.events
    ]
    html = _BROWSER_TEMPLATE.format(
        title=_escape(f"QDebug timeline — {timeline.name} "
                      f"({timeline.num_qubits} qubits, {len(timeline.events)} events)"),
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
  pre {{ background: #f5f5f5; padding: 1rem; border-radius: 6px; overflow-x: auto; line-height: 1.2; }}
</style>
</head>
<body>
  <h4>{title}</h4>
  <input id="slider" type="range" min="1" value="1">
  <div id="header"></div>
  <pre id="body"></pre>
<script>
  const frames = {frames};
  const slider = document.getElementById("slider");
  const header = document.getElementById("header");
  const body = document.getElementById("body");
  slider.max = frames.length;
  function render() {{
    const f = frames[slider.value - 1];
    header.innerHTML = `<b>${{slider.value}}/${{frames.length}}</b> &mdash; ${{f.header}}`;
    body.innerHTML = f.body;
  }}
  slider.addEventListener("input", render);
  render();
</script>
</body>
</html>
"""


def _render_event(timeline: Timeline, event: TelemetryEvent) -> str:
    """Return the detail panel text for one event."""
    if isinstance(event, GateEvent):
        return _render_gate(timeline, event)
    if isinstance(event, StateSnapshotEvent):
        return format_statevector(event)
    if isinstance(event, TranspilePassEvent):
        return _render_pass(event)
    lines = []
    for key, value in event.model_dump(mode="json").items():
        if key not in ("type", "seq", "timestamp"):
            lines.append(f"{key}: {value}")
    return "\n".join(lines) or event.describe()


def _render_gate(timeline: Timeline, event: GateEvent) -> str:
    """Draw the circuit as it looked once this gate had been appended."""
    gates = timeline.gate_events
    position = next((i for i, e in enumerate(gates) if e.seq == event.seq), len(gates) - 1)
    circuit = timeline.circuit_at(position + 1)
    if circuit is None:
        return event.describe()
    try:
        return str(circuit.draw(output="text"))
    except Exception:
        return event.describe()


def _render_pass(event: TranspilePassEvent) -> str:
    """Show pass statistics plus the circuit snapshot when one was kept."""
    lines = [
        f"pass:     {event.pass_name}",
        f"duration: {event.duration:.6f}s",
        f"depth:    {event.depth}   size: {event.size}   changed: {event.changed}",
        f"ops:      {event.ops}",
    ]
    if event.circuit is not None:
        try:
            lines += ["", str(event.circuit.draw(output="text"))]
        except Exception:
            pass
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


def _escape(text: str) -> str:
    """Minimal HTML escaping for widget content."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
