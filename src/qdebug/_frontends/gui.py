"""Native GUI frontend: a tkinter window, used when a display is available.

tkinter ships with the Python standard library, so this needs no extra
dependency beyond what a normal desktop Python install already provides.
"""

from __future__ import annotations

import base64
import importlib.util
import os
import sys
import tkinter
from tkinter import ttk
from typing import Callable, ClassVar

from .._events import TelemetryEvent, Timeline
from . import _render
from ._playback import PlaybackController
from .base import TimelineFrontend

DEFAULT_AUTOPLAY_INTERVAL_MS = 800


def gui_available() -> bool:
    """Return whether a native GUI window could actually be opened here."""
    if importlib.util.find_spec("tkinter") is None:
        return False
    if sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        return False
    try:
        root = tkinter.Tk()
        root.withdraw()
        root.destroy()
    except tkinter.TclError:
        return False
    return True


class TkFrontend(TimelineFrontend):
    """Opens a Tk window with a slider, play controls, and a text/image panel."""

    name: ClassVar[str] = "gui"

    def is_available(self) -> bool:
        return gui_available()

    def unavailable_reason(self) -> str:
        return (
            "no display is available for a native GUI window "
            "(tkinter missing, or DISPLAY/WAYLAND_DISPLAY unset)"
        )

    def show(self, timelines: list[Timeline]) -> None:
        if not self.is_available():
            raise RuntimeError(self.unavailable_reason())
        merged, tracks, _active_labels = _render.merge_timelines(timelines)
        names = ", ".join(str(t.name) for t in timelines)
        window = _TimelineWindow(names, merged, tracks)
        window.run()


class _TimelineWindow:  # pylint: disable=too-many-instance-attributes
    """A single Tk window presenting one merged, navigable timeline.

    This class only does two things: lay out Tk widgets and translate Tk
    events into calls on a :class:`~._playback.PlaybackController`. It also
    implements the controller's ``Scheduler`` protocol via ``root.after``.
    All play/pause/step/seek *logic* lives in the controller so a future
    toolkit (e.g. PyQt6) can reuse it by writing a new thin window class
    around the same controller instead of re-deriving that state machine.

    Holds one attribute per widget by design: it's a stateful window
    object, not a data model, and splitting the widgets into a
    sub-container would only add indirection without reducing complexity.
    """

    def __init__(
        self,
        title: str,
        merged: list[tuple[Timeline, TelemetryEvent]],
        tracks: list[dict[str, str]],
    ) -> None:
        self.tracks = tracks
        self.photo: tkinter.PhotoImage | None = None
        self._syncing_slider = False

        self.root = tkinter.Tk()
        self.root.title(f"QDebug timeline — {title}")
        self.root.geometry("900x650")

        if not merged:
            tkinter.Label(self.root, text="Timeline(s) have no events.").pack(padx=20, pady=20)
            self.controller: PlaybackController | None = None
            return

        self.header_var = tkinter.StringVar()
        self.text = tkinter.Text(self.root, wrap="none", font=("Courier", 11))
        self.image_label = tkinter.Label(self.root)
        self.play_button = tkinter.Button(self.root, text="Play", command=self._toggle_play)
        self.interval_input = tkinter.Spinbox(
            self.root, from_=50, to=10_000, increment=50, width=6
        )
        self.slider = ttk.Scale(
            self.root, from_=1, to=len(merged), orient="horizontal", command=self._on_slider
        )
        self.controller = PlaybackController(
            merged=merged,
            scheduler=self,
            get_interval_ms=self._interval_ms,
            on_render=self._render,
            on_play_state_changed=self._on_play_state_changed,
        )
        self._layout_widgets()
        self._render()

    def run(self) -> None:
        """Enter the Tk event loop. Blocks until the window is closed."""
        self.root.mainloop()

    # -- Scheduler protocol, used by PlaybackController for autoplay -----

    def schedule(self, delay_ms: int, callback: Callable[[], None]) -> object:
        return self.root.after(delay_ms, callback)

    def cancel(self, handle: object) -> None:
        self.root.after_cancel(handle)  # type: ignore[arg-type]

    # -- Layout ------------------------------------------------------------

    def _layout_widgets(self) -> None:
        assert self.controller is not None
        controller = self.controller

        tkinter.Label(self.root, textvariable=self.header_var, anchor="w", justify="left").pack(
            fill="x", padx=10, pady=(10, 4)
        )
        self.text.pack(fill="both", expand=True, padx=10, pady=4)
        self.image_label.pack(padx=10, pady=4)

        controls = tkinter.Frame(self.root)
        controls.pack(fill="x", padx=10, pady=(4, 10))
        tkinter.Button(controls, text="⏮ First", command=lambda: self._goto(0)).pack(side="left")
        tkinter.Button(
            controls, text="◀ Prev", command=lambda: self._goto(controller.index - 1)
        ).pack(side="left")
        self.play_button.pack(side="left")
        tkinter.Button(
            controls, text="Next ▶", command=lambda: self._goto(controller.index + 1)
        ).pack(side="left")
        tkinter.Button(
            controls, text="Last ⏭", command=lambda: self._goto(controller.last_index)
        ).pack(side="left")
        tkinter.Label(controls, text="ms/step").pack(side="left", padx=(12, 2))
        self.interval_input.delete(0, "end")
        self.interval_input.insert(0, str(DEFAULT_AUTOPLAY_INTERVAL_MS))
        self.interval_input.pack(side="left")

        self.slider.pack(fill="x", padx=10, pady=(0, 4))

    # -- Widget <-> PlaybackController glue --------------------------------

    def _render(self) -> None:
        """Refresh every widget to reflect the controller's current event."""
        assert self.controller is not None
        controller = self.controller

        source, event = controller.current
        index = controller.index
        self.header_var.set(
            f"{index + 1}/{len(controller.merged)} — {source.name} — {event.type.value} "
            f"at {event.timestamp:.3f} — {event.describe()}"
        )
        self.text.delete("1.0", "end")
        self.text.insert(
            "end",
            _render.render_tracks_text(self.tracks[index])
            + "\n\n"
            + _render.render_event(source, event),
        )
        self._syncing_slider = True
        try:
            self.slider.set(index + 1)
        finally:
            self._syncing_slider = False

        _label, png_bytes = _render.render_extra_visual_bytes(event)
        if png_bytes:
            self.photo = tkinter.PhotoImage(data=base64.b64encode(png_bytes))
            self.image_label.configure(image=self.photo)
        else:
            self.image_label.configure(image="")
            self.photo = None

    def _on_play_state_changed(self, playing: bool) -> None:
        self.play_button.configure(text="Pause" if playing else "Play")

    def _goto(self, index: int) -> None:
        assert self.controller is not None
        self.controller.goto(index)

    def _toggle_play(self) -> None:
        assert self.controller is not None
        self.controller.toggle_play()

    def _interval_ms(self) -> int:
        try:
            return int(self.interval_input.get())
        except ValueError:
            return DEFAULT_AUTOPLAY_INTERVAL_MS

    def _on_slider(self, value: str) -> None:
        if self._syncing_slider:
            return
        self._goto(int(float(value)) - 1)
