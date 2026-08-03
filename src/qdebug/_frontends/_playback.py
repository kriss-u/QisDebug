"""Toolkit-agnostic playback state machine shared by scrubbable frontends.

Any frontend that presents a timeline as a play/pause/step/seek sequence
(the Tk window today, a future PyQt6 window tomorrow) drives this
controller instead of reimplementing the state machine against its own
event loop. The controller knows nothing about widgets; it talks to the
toolkit only through the :class:`Scheduler` protocol and the callbacks
passed to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

from .._events import TelemetryEvent, Timeline


class Scheduler(Protocol):
    """Toolkit-specific hook for delayed, cancellable callbacks.

    Tk implements this with ``root.after``/``root.after_cancel``; a PyQt6
    frontend would implement it with ``QTimer.singleShot``/``timer.stop``.
    """

    def schedule(self, delay_ms: int, callback: Callable[[], None]) -> object:
        """Schedule ``callback`` after ``delay_ms`` and return a handle."""

    def cancel(self, handle: object) -> None:
        """Cancel a callback previously returned by :meth:`schedule`."""


@dataclass
class PlaybackController:
    """Owns the current index and autoplay state for one merged timeline."""

    merged: list[tuple[Timeline, TelemetryEvent]]
    scheduler: Scheduler
    get_interval_ms: Callable[[], int]
    on_render: Callable[[], None]
    on_play_state_changed: Callable[[bool], None]
    index: int = field(default=0, init=False)
    playing: bool = field(default=False, init=False)
    _handle: object | None = field(default=None, init=False)

    @property
    def current(self) -> tuple[Timeline, TelemetryEvent]:
        return self.merged[self.index]

    @property
    def last_index(self) -> int:
        return len(self.merged) - 1

    def goto(self, index: int) -> None:
        """Jump to ``index``, stopping autoplay first."""
        self.stop()
        self.index = min(max(index, 0), self.last_index)
        self.on_render()

    def stop(self) -> None:
        """Pause autoplay, if running."""
        if self._handle is not None:
            self.scheduler.cancel(self._handle)
            self._handle = None
        if self.playing:
            self.playing = False
            self.on_play_state_changed(False)

    def toggle_play(self) -> None:
        """Start or pause autoplay."""
        if self.playing:
            self.stop()
            return
        if self.index >= self.last_index:
            return
        self.playing = True
        self.on_play_state_changed(True)
        self._handle = self.scheduler.schedule(self.get_interval_ms(), self._tick)

    def _tick(self) -> None:
        if self.index >= self.last_index:
            self.stop()
            return
        self.index += 1
        self.on_render()
        self._handle = self.scheduler.schedule(self.get_interval_ms(), self._tick)
