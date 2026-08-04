"""Toolkit-agnostic playback state machine shared by scrubbable frontends.

Any frontend that presents a timeline as a play/pause/step/seek sequence
(the Tk window today, a future PyQt6 window tomorrow) drives this
controller instead of reimplementing the state machine against its own
event loop. The controller knows nothing about widgets; it talks to the
toolkit only through the :class:`Scheduler` protocol and the callbacks
passed to it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Generic, TypeVar

from .._events import TelemetryEvent, Timeline

Handle = TypeVar("Handle")


class Scheduler(ABC, Generic[Handle]):
    """Toolkit-specific hook for delayed, cancellable callbacks.

    Generic in the handle type returned by :meth:`schedule` and consumed by
    :meth:`cancel`, since that's a different concrete type per toolkit (a
    PySide6 frontend uses ``QTimer``; a Tk one would use the id ``after``
    returns). Concrete frontends must subclass this explicitly (e.g.
    ``class _TimelineWindow(Scheduler[QTimer])``) rather than merely
    matching its shape, mirroring how :class:`~.base.TimelineFrontend` is
    subclassed elsewhere in this package -- Python then refuses to
    instantiate a subclass that's missing either method, instead of only a
    type checker catching it.
    """

    @abstractmethod
    def schedule(self, delay_ms: int, callback: Callable[[], None]) -> Handle:
        """Schedule ``callback`` after ``delay_ms`` and return a handle."""

    @abstractmethod
    def cancel(self, handle: Handle) -> None:
        """Cancel a callback previously returned by :meth:`schedule`."""


@dataclass(frozen=True)
class PlaybackHooks:
    """UI callbacks a :class:`PlaybackController` reports through, bundled
    into one field so the controller isn't individually plumbed through
    three same-shaped constructor parameters.
    """

    get_interval_ms: Callable[[], int]
    on_render: Callable[[], None]
    on_play_state_changed: Callable[[bool], None]


@dataclass
class PlaybackController(Generic[Handle]):
    """Owns the current index and autoplay state for one merged timeline."""

    merged: list[tuple[Timeline, TelemetryEvent]]
    scheduler: Scheduler[Handle]
    hooks: PlaybackHooks
    index: int = field(default=0, init=False)
    playing: bool = field(default=False, init=False)
    _handle: Handle | None = field(default=None, init=False)

    @property
    def current(self) -> tuple[Timeline, TelemetryEvent]:
        """The (source timeline, event) pair at the current index."""
        return self.merged[self.index]

    @property
    def last_index(self) -> int:
        """The highest valid index into :attr:`merged`."""
        return len(self.merged) - 1

    def goto(self, index: int) -> None:
        """Jump to ``index``, stopping autoplay first."""
        self.stop()
        self.index = min(max(index, 0), self.last_index)
        self.hooks.on_render()

    def stop(self) -> None:
        """Pause autoplay, if running."""
        if self._handle is not None:
            self.scheduler.cancel(self._handle)
            self._handle = None
        if self.playing:
            self.playing = False
            self.hooks.on_play_state_changed(False)

    def toggle_play(self) -> None:
        """Start or pause autoplay."""
        if self.playing:
            self.stop()
            return
        if self.index >= self.last_index:
            return
        self.playing = True
        self.hooks.on_play_state_changed(True)
        self._handle = self.scheduler.schedule(self.hooks.get_interval_ms(), self._tick)

    def _tick(self) -> None:
        if self.index >= self.last_index:
            self.stop()
            return
        self.index += 1
        self.hooks.on_render()
        self._handle = self.scheduler.schedule(self.hooks.get_interval_ms(), self._tick)
