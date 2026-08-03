"""Jupyter widget frontend: an interactive timeline explorer for notebooks."""

from __future__ import annotations

import importlib.util
from typing import Any, ClassVar

from .._events import Timeline
from . import _render
from .base import TimelineFrontend

DEFAULT_AUTOPLAY_INTERVAL_MS = 800


class JupyterFrontend(TimelineFrontend):
    """Renders an ipywidgets widget tree for the notebook to display."""

    name: ClassVar[str] = "jupyter"

    def is_available(self) -> bool:
        return importlib.util.find_spec("ipywidgets") is not None

    def unavailable_reason(self) -> str:
        return "ipywidgets is not installed; install it with pip install 'qdebug[viz]'"

    def show(self, timelines: list[Timeline]) -> Any:
        try:
            import ipywidgets as widgets
        except ImportError as exc:
            raise ImportError(self.unavailable_reason()) from exc

        merged, tracks, _active_labels = _render.merge_timelines(timelines)
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
                f"<code>{_render.escape(str(source.name))}</code> &mdash; "
                f"<code>{event.type.value}</code> at {event.timestamp:.3f} &mdash; "
                f"{_render.escape(event.describe())}"
            )
            body.value = (
                "<pre style='line-height:1.2;white-space:pre;overflow-x:auto'>"
                f"{_render.escape(_render.render_event(source, event))}</pre>"
            )
            circuits.value = self._render_tracks_html(tracks[slider.value - 1])
            label, image = _render.render_extra_visual(event)
            extra.value = (
                f"<b>{_render.escape(label)}</b><br><img src='{image}' style='max-width:100%'>"
                if image
                else ""
            )

        slider.observe(render, names="value")
        render()
        names = ", ".join(_render.escape(str(t.name)) for t in timelines)
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
        stop_button = widgets.Button(
            description="Stop", icon="stop", layout=widgets.Layout(width="80px")
        )

        def on_stop(_button: Any) -> None:
            """Pause autoplay and jump back to the first event."""
            play.playing = False
            slider.value = slider.min

        stop_button.on_click(on_stop)
        widgets.jslink((play, "value"), (slider, "value"))
        widgets.jslink((play, "interval"), (interval_input, "value"))
        controls = widgets.HBox([play, interval_input, stop_button])

        return widgets.VBox([title, controls, slider, header, circuits, body, extra])

    def _render_tracks_html(self, tracks: dict[str, str]) -> str:
        """Render the circuit panel HTML for one point in the timeline."""
        if not tracks:
            return "<i>no circuit recorded yet</i>"
        sections = []
        for key, text in tracks.items():
            sections.append(
                f"<b>{_render.escape(_render.track_label(key))}</b>"
                "<pre style='line-height:1.2;white-space:pre;overflow-x:auto'>"
                f"{_render.escape(text)}</pre>"
            )
        return "".join(sections)
