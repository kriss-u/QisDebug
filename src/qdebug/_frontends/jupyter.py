"""Jupyter widget frontend: an interactive timeline explorer for notebooks."""

from __future__ import annotations

import importlib.util
from typing import Any, Callable, ClassVar

from .._events import TelemetryEvent, Timeline
from . import _render
from .base import TimelineFrontend

DEFAULT_AUTOPLAY_INTERVAL_MS = 800


class _WatchPanel:
    """The watch kind/ref pickers, add button, and per-watch list as one unit.

    Bundling this widget cluster and its mutable ``targets`` list into an
    object -- rather than a handful of separate locals and closures inside
    ``JupyterFrontend.show`` -- keeps that method focused on wiring the main
    playback widgets together.
    """

    def __init__(self, timelines_by_id: dict[str, Timeline]) -> None:
        import ipywidgets as widgets

        self._timelines_by_id = timelines_by_id
        self.targets: list[_render.WatchTarget] = []
        self.kind = widgets.Dropdown(
            options=["qubit", "clbit", "register"],
            description="watch:",
            layout=widgets.Layout(width="200px"),
        )
        self.ref = widgets.Dropdown(options=[], layout=widgets.Layout(width="140px"))
        self.add_button = widgets.Button(
            description="Add", icon="plus", layout=widgets.Layout(width="80px")
        )
        self.list_box = widgets.VBox([])
        self.html = widgets.HTML()

    @property
    def controls(self) -> Any:
        """The kind/ref/add-button row, built on demand rather than stored."""
        import ipywidgets as widgets

        return widgets.HBox([self.kind, self.ref, self.add_button])

    def refresh_ref_choices(self, source: Timeline) -> None:
        """Repopulate the ref dropdown for ``source`` given the current kind."""
        if self.kind.value == "qubit":
            self.ref.options = [str(i) for i in range(source.num_qubits)]
        elif self.kind.value == "clbit":
            self.ref.options = [str(i) for i in range(source.num_clbits)]
        else:
            self.ref.options = [*source.qregs, *source.cregs]

    def add(self, source: Timeline, on_change: Callable[[], None]) -> None:
        """Add the currently selected ref as a watch target, then re-render."""
        if not self.ref.options:
            return
        kind = self.kind.value
        assert kind in ("qubit", "clbit", "register")
        target = _render.WatchTarget(
            circuit_id=source.circuit_id, kind=kind, ref=str(self.ref.value)
        )
        if target not in self.targets:
            self.targets.append(target)
            self._refresh_list(on_change)
            on_change()

    def _refresh_list(self, on_change: Callable[[], None]) -> None:
        import ipywidgets as widgets

        rows = []
        for target in list(self.targets):
            timeline = self._timelines_by_id.get(target.circuit_id)

            def on_remove(_button: Any, target: _render.WatchTarget = target) -> None:
                self.targets.remove(target)
                self._refresh_list(on_change)
                on_change()

            remove_button = widgets.Button(description="x", layout=widgets.Layout(width="28px"))
            remove_button.on_click(on_remove)
            rows.append(widgets.HBox([widgets.Label(target.label(timeline)), remove_button]))
        self.list_box.children = tuple(rows)

    def render(self, event: TelemetryEvent) -> None:
        """Refresh the watch text panel for the currently selected event."""
        text = _render.render_watches(self.targets, self._timelines_by_id, event.seq)
        self.html.value = (
            "<pre style='line-height:1.2;white-space:pre;overflow-x:auto'>"
            f"{_render.escape(text)}</pre>"
            if text
            else ""
        )


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

        timelines_by_id = {t.circuit_id: t for t in timelines}
        watch_panel = _WatchPanel(timelines_by_id)

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
            watch_panel.render(event)
            watch_panel.refresh_ref_choices(source)

        slider.observe(render, names="value")
        watch_panel.kind.observe(lambda _change: render(), names="value")
        watch_panel.add_button.on_click(
            lambda _button: watch_panel.add(merged[slider.value - 1][0], render)
        )
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

        return widgets.VBox(
            [
                title,
                controls,
                slider,
                header,
                circuits,
                body,
                extra,
                watch_panel.controls,
                watch_panel.list_box,
                watch_panel.html,
            ]
        )

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
