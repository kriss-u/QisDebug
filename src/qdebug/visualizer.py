"""Interactive Jupyter widget for exploring QDebug timelines.

Deliberately decoupled from the instrumentation layers: it only reads the
public fields of :class:`~qdebug._events.Timeline`, so either side can be
replaced independently. Requires ipywidgets (``pip install "qdebug[viz]"``).
"""

from __future__ import annotations

import base64
import io
import json
import tempfile
import webbrowser
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ._events import (
    ExecutionFinishedEvent,
    GateEvent,
    StateSnapshotEvent,
    TelemetryEvent,
    Timeline,
    TranspileFinishedEvent,
    TranspilePassEvent,
)

DEFAULT_AUTOPLAY_INTERVAL_MS = 800


TRACK_LABELS = {
    "circuit": "Circuit (construction)",
    "transpiled": "Circuit (transpiled)",
    "backend": "Circuit (backend compiled)",
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
    merged, tracks, _active_labels = _merge_timelines(timelines)
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
    extra = widgets.HTML()

    def render(change: Any = None) -> None:
        source, event = merged[slider.value - 1]
        header.value = (
            f"<b>{slider.value}/{len(merged)}</b> &mdash; "
            f"<code>{_escape(str(source.name))}</code> &mdash; "
            f"<code>{event.type.value}</code> at {event.timestamp:.3f} &mdash; "
            f"{_escape(event.describe())}"
        )
        body.value = (
            "<pre style='line-height:1.2;white-space:pre;overflow-x:auto'>"
            f"{_escape(_render_event(source, event))}</pre>"
        )
        circuits.value = _render_tracks_html(tracks[slider.value - 1])
        label, image = _render_extra_visual(event)
        extra.value = (
            f"<b>{_escape(label)}</b><br><img src='{image}' style='max-width:100%'>"
            if image
            else ""
        )

    slider.observe(render, names="value")
    render()
    names = ", ".join(_escape(str(t.name)) for t in timelines)
    title = widgets.HTML(
        value=f"<h4>QDebug timeline &mdash; {names} ({len(merged)} events)</h4>"
    )

    play = widgets.Play(
        value=1,
        min=1,
        max=len(merged),
        step=1,
        interval=DEFAULT_AUTOPLAY_INTERVAL_MS,
        description="Autoplay",
    )
    interval_input = widgets.BoundedIntText(
        value=DEFAULT_AUTOPLAY_INTERVAL_MS,
        min=50,
        max=10_000,
        step=50,
        description="ms/step",
        layout=widgets.Layout(width="160px"),
    )
    stop_button = widgets.Button(description="Stop", icon="stop", layout=widgets.Layout(width="80px"))

    def on_stop(_button: Any) -> None:
        """Pause autoplay and jump back to the first event."""
        play.playing = False
        slider.value = slider.min

    stop_button.on_click(on_stop)
    widgets.jslink((play, "value"), (slider, "value"))
    widgets.jslink((play, "interval"), (interval_input, "value"))
    controls = widgets.HBox([play, interval_input, stop_button])

    return widgets.VBox([title, controls, slider, header, circuits, body, extra])


def _circuit_tracks(timeline: Timeline) -> tuple[list[dict[str, str]], list[str | None]]:
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


def _merge_timelines(
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
    per_timeline_tracks = {t.circuit_id: _circuit_tracks(t) for t in timelines}
    merged = sorted(
        ((t, event) for t in timelines for event in t.events),
        key=lambda pair: pair[1].seq,
    )
    cursor = {t.circuit_id: 0 for t in timelines}
    combined: dict[str, str] = {}
    series: list[dict[str, str]] = []
    active_labels: list[str | None] = []
    for source, _event in merged:
        series_for_source, active_for_source = per_timeline_tracks[source.circuit_id]
        idx = cursor[source.circuit_id]
        local = series_for_source[idx]
        active_key = active_for_source[idx]
        cursor[source.circuit_id] += 1
        prefix = source.name or source.circuit_id
        for key, text in local.items():
            combined[f"{prefix}:{key}"] = text
        series.append(dict(combined))
        active_labels.append(_track_label(f"{prefix}:{active_key}") if active_key else None)
    return merged, series, active_labels


def _render_tracks_html(tracks: dict[str, str]) -> str:
    """Render the circuit panel HTML for one point in the timeline."""
    if not tracks:
        return "<i>no circuit recorded yet</i>"
    sections = []
    for key, text in tracks.items():
        sections.append(
            f"<b>{_escape(_track_label(key))}</b>"
            "<pre style='line-height:1.2;white-space:pre;overflow-x:auto'>"
            f"{_escape(text)}</pre>"
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
    merged, tracks, active_labels = _merge_timelines(timelines)
    frames = []
    for i, (source, event) in enumerate(merged):
        extra_label, extra_image = _render_extra_visual(event)
        frames.append(
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
                "active": active_labels[i],
                "extraLabel": extra_label,
                "extraImage": extra_image,
            }
        )
    names = ", ".join(_escape(str(t.name)) for t in timelines)
    html = _BROWSER_TEMPLATE.format(
        title=_escape(f"QDebug timeline — {names} ({len(merged)} events)"),
        frames=json.dumps(frames).replace("</", "<\\/"),
        default_interval=DEFAULT_AUTOPLAY_INTERVAL_MS,
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
<script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script>
<style>
  :root {{
    --bg: #f6f7f9;
    --surface: #ffffff;
    --border: #d8dce2;
    --text: #1c2128;
    --text-muted: #57606a;
    --accent: #2f6feb;
    --accent-soft: #eaf1ff;
    --mono: "SFMono-Regular", "SF Mono", Consolas, "Liberation Mono", Menlo, monospace;
    --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: var(--sans);
    margin: 0;
    padding: 2rem 2.5rem 3rem;
    background: var(--bg);
    color: var(--text);
  }}
  h4 {{
    margin: 0 0 1rem;
    font-size: 1.05rem;
    font-weight: 600;
    letter-spacing: -0.01em;
  }}
  #nav {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.6rem 0.75rem;
    margin-bottom: 1rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
  }}
  #nav button {{
    font-family: var(--sans);
    font-size: 0.85rem;
    font-weight: 500;
    padding: 0.4rem 0.85rem;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: var(--surface);
    color: var(--text);
    cursor: pointer;
    transition: background 0.1s ease, border-color 0.1s ease;
  }}
  #nav button:hover:not(:disabled) {{ background: var(--accent-soft); border-color: var(--accent); }}
  #nav button:disabled {{ cursor: default; opacity: 0.4; }}
  #nav button.playing {{ background: var(--accent); border-color: var(--accent); color: #fff; }}
  #nav button.playing:hover:not(:disabled) {{ background: var(--accent); border-color: var(--accent); }}
  #nav .event-count {{
    font-family: var(--mono);
    font-size: 0.85rem;
    color: var(--text-muted);
    margin: 0 0.25rem;
  }}
  #nav input[type=number] {{
    font-family: var(--mono);
    width: 3.75rem;
    text-align: center;
    padding: 0.35rem 0.25rem;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: var(--surface);
    color: var(--text);
  }}
  #nav input[type=number]:focus {{ outline: none; border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft); }}
  #nav .sep {{ width: 1px; align-self: stretch; background: var(--border); margin: 0 0.25rem; }}
  #nav input[type=number].interval {{ width: 4.5rem; }}
  #extra-viz {{
    margin-bottom: 1rem;
    padding: 0.65rem 0.9rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
  }}
  #extra-viz h5 {{
    margin: 0 0 0.5rem;
    font-family: var(--sans);
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    color: var(--text-muted);
  }}
  #extra-viz img {{ max-width: 100%; display: block; }}
  #header {{
    margin-bottom: 1rem;
    padding: 0.65rem 0.9rem;
    font-family: var(--mono);
    font-size: 0.85rem;
    line-height: 1.5;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text-muted);
  }}
  #header b {{ color: var(--text); font-family: var(--sans); font-weight: 600; }}
  #header code {{ background: var(--accent-soft); color: var(--accent); padding: 0.1rem 0.35rem; border-radius: 4px; font-size: 0.82rem; }}
  #circuits {{
    display: flex;
    gap: 1rem;
    overflow-x: auto;
    scroll-behavior: smooth;
    padding: 0.25rem 0.25rem 1rem;
    margin-bottom: 1rem;
  }}
  #circuits > i {{ color: var(--text-muted); font-family: var(--sans); font-size: 0.9rem; padding: 0.5rem; }}
  .circuit-card {{
    flex: 0 0 auto;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
    background: var(--surface);
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
  }}
  .circuit-card.active {{ border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft); }}
  .circuit-card h5 {{
    margin: 0 0 0.4rem;
    font-family: var(--sans);
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    color: var(--text-muted);
  }}
  .circuit-card.active h5 {{ color: var(--accent); }}
  .circuit-card pre {{ margin: 0; background: transparent; border: none; padding: 0; }}
  pre {{
    font-family: var(--mono);
    font-size: 0.82rem;
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 1rem;
    border-radius: 8px;
    overflow-x: auto;
    white-space: pre;
    line-height: 1.45;
  }}
</style>
</head>
<body>
  <h4>{title}</h4>
  <script type="application/json" id="qdebug-frames">{frames}</script>
  <script>
    function qdebugViz() {{
      return {{
        frames: JSON.parse(document.getElementById("qdebug-frames").textContent),
        index: 0,
        playing: false,
        paused: false,
        timer: null,
        intervalMs: {default_interval},
        first() {{ this.stop(); this.index = 0; }},
        prev() {{ this.stop(); if (this.index > 0) this.index--; }},
        next() {{ this.stop(); if (this.index < this.frames.length - 1) this.index++; }},
        last() {{ this.stop(); this.index = this.frames.length - 1; }},
        jumpTo(value) {{
          this.stop();
          const n = Math.min(Math.max(1, value), this.frames.length);
          this.index = n - 1;
        }},
        tick() {{
          if (this.index < this.frames.length - 1) {{
            this.index++;
          }} else {{
            this.stop();
          }}
        }},
        play() {{
          if (this.playing || this.index >= this.frames.length - 1) return;
          this.playing = true;
          this.paused = false;
          this.timer = setInterval(() => this.tick(), this.intervalMs);
        }},
        pause() {{
          if (!this.playing) return;
          clearInterval(this.timer);
          this.timer = null;
          this.playing = false;
          this.paused = true;
        }},
        resume() {{ if (this.paused) this.play(); }},
        togglePlay() {{
          if (this.playing) {{
            this.pause();
          }} else {{
            this.play();
          }}
        }},
        stop() {{
          clearInterval(this.timer);
          this.timer = null;
          this.playing = false;
          this.paused = false;
        }},
        restartTimer() {{
          if (this.playing) {{
            clearInterval(this.timer);
            this.timer = setInterval(() => this.tick(), this.intervalMs);
          }}
        }},
        scrollToActive() {{
          this.$nextTick(() => {{
            const active = this.$refs.circuitsBox.querySelector('.circuit-card.active');
            if (active) active.scrollIntoView({{ behavior: 'smooth', inline: 'center', block: 'nearest' }});
          }});
        }},
      }};
    }}
  </script>
  <div x-data="qdebugViz()">
    <div id="nav">
      <button @click="first()" :disabled="index === 0 || playing">&#9198; First</button>
      <button @click="prev()" :disabled="index === 0 || playing">&#9664; Prev</button>
      <span class="event-count">event <input type="number" min="1" :max="frames.length" :value="index + 1" :disabled="playing" @change="jumpTo($event.target.valueAsNumber)"> / <span x-text="frames.length"></span></span>
      <button @click="next()" :disabled="index === frames.length - 1 || playing">Next &#9654;</button>
      <button @click="last()" :disabled="index === frames.length - 1 || playing">Last &#9197;</button>
      <span class="sep"></span>
      <button @click="togglePlay()" :class="{{ playing: playing }}" :disabled="!playing && index >= frames.length - 1">
        <span x-show="!playing">&#9654; <span x-text="paused ? 'Resume' : 'Play'"></span></span>
        <span x-show="playing">&#10074;&#10074; Pause</span>
      </button>
      <button @click="stop()" :disabled="!playing && !paused">&#9632; Stop</button>
      <span class="event-count">ms/step <input type="number" class="interval" min="50" max="10000" step="50" x-model.number="intervalMs" @change="restartTimer()"></span>
    </div>
    <div id="header">
      <b x-text="(index + 1) + '/' + frames.length"></b>&nbsp;&mdash;&nbsp;<span x-html="frames[index].header"></span>
    </div>
    <div id="circuits" x-ref="circuitsBox" x-effect="index; scrollToActive()">
      <template x-if="Object.keys(frames[index].circuits).length === 0">
        <i>no circuit recorded yet</i>
      </template>
      <template x-for="label in Object.keys(frames[index].circuits)" :key="label">
        <div class="circuit-card" :class="{{ active: label === frames[index].active }}">
          <h5 x-text="label"></h5>
          <pre x-html="frames[index].circuits[label]"></pre>
        </div>
      </template>
    </div>
    <pre x-html="frames[index].body"></pre>
    <template x-if="frames[index].extraImage">
      <div id="extra-viz">
        <h5 x-text="frames[index].extraLabel"></h5>
        <img :src="frames[index].extraImage" alt="event visualization">
      </div>
    </template>
  </div>
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


def _escape(text: str) -> str:
    """Minimal HTML escaping for widget content."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _figure_to_data_uri(fig: Any) -> str:
    """Encode a matplotlib figure as a base64 PNG data URI, then close it."""
    import matplotlib.pyplot as plt

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_histogram(event: ExecutionFinishedEvent) -> str | None:
    """Render an execution's counts as a histogram, or None if unavailable.

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


def _render_extra_visual(event: TelemetryEvent) -> tuple[str, str | None]:
    """Return (label, data_uri) for an event's supplementary visualization, if any."""
    if isinstance(event, ExecutionFinishedEvent) and event.counts:
        image = render_histogram(event)
        return ("Counts histogram", image) if image else ("", None)
    if isinstance(event, StateSnapshotEvent):
        image = render_bloch(event)
        return ("Bloch sphere", image) if image else ("", None)
    return "", None
