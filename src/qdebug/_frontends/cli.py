"""Text-based interactive frontend: the universal fallback (e.g. SSH sessions).

Built on the standard library's :mod:`cmd` module rather than a hand-rolled
``input()`` loop, so command parsing, aliasing, and help text come from a
purpose-built interpreter instead of ad hoc string matching. ``cmd`` ships
with Python, so this frontend keeps no extra dependency and stays available
in any environment.
"""

from __future__ import annotations

import cmd
from typing import ClassVar

from .._events import ExecutionFinishedEvent, TelemetryEvent, Timeline
from . import _render
from .base import TimelineFrontend


class _TimelineShell(cmd.Cmd):
    """Line-oriented REPL for stepping through a merged timeline."""

    prompt = "qdebug> "

    def __init__(
        self,
        merged: list[tuple[Timeline, TelemetryEvent]],
        tracks: list[dict[str, str]],
    ) -> None:
        super().__init__()
        self.merged = merged
        self.tracks = tracks
        self.index = 0

    def print_frame(self) -> None:
        """Print the currently selected event and its circuit/state panels."""
        source, event = self.merged[self.index]
        print()
        print(
            f"[{self.index + 1}/{len(self.merged)}] {source.name} — {event.type.value} "
            f"at {event.timestamp:.3f} — {event.describe()}"
        )
        print(_render.render_tracks_text(self.tracks[self.index]))
        print(_render.render_event(source, event))
        if isinstance(event, ExecutionFinishedEvent) and event.counts:
            print()
            print("Counts histogram:")
            print(_render.render_histogram_ascii(event.counts))

    def emptyline(self) -> bool:
        """A bare Enter repeats "next" rather than cmd's default (repeat last)."""
        return self.do_next("")

    def do_next(self, _arg: str) -> bool:
        """n, next: advance to the next event."""
        self.index = min(self.index + 1, len(self.merged) - 1)
        self.print_frame()
        return False

    do_n = do_next

    def do_prev(self, _arg: str) -> bool:
        """p, prev: go back to the previous event."""
        self.index = max(self.index - 1, 0)
        self.print_frame()
        return False

    do_p = do_prev

    def do_first(self, _arg: str) -> bool:
        """f, first: jump to the first event."""
        self.index = 0
        self.print_frame()
        return False

    do_f = do_first

    def do_last(self, _arg: str) -> bool:
        """l, last: jump to the last event."""
        self.index = len(self.merged) - 1
        self.print_frame()
        return False

    do_l = do_last

    def do_goto(self, arg: str) -> bool:
        """g N, goto N: jump to event N."""
        target = arg.strip()
        if not target.isdigit():
            print(f"not a valid event number: {target!r}")
            return False
        self.index = min(max(int(target) - 1, 0), len(self.merged) - 1)
        self.print_frame()
        return False

    do_g = do_goto

    def do_quit(self, _arg: str) -> bool:
        """q, quit: exit."""
        return True

    do_q = do_quit
    do_h = cmd.Cmd.do_help

    def do_EOF(self, _arg: str) -> bool:  # pylint: disable=invalid-name
        """Exit on end-of-input (e.g. Ctrl-D or a closed pipe).

        Named ``do_EOF`` rather than snake_case because :mod:`cmd` dispatches
        the pseudo-command it synthesizes on EOF by that exact name.
        """
        print()
        return True


class CLIFrontend(TimelineFrontend):
    """Prints one event at a time and steps through them via stdin commands."""

    name: ClassVar[str] = "cli"

    def is_available(self) -> bool:
        return True

    def unavailable_reason(self) -> str:
        return ""

    def show(self, timelines: list[Timeline]) -> None:
        merged, tracks, _active_labels = _render.merge_timelines(timelines)
        if not merged:
            ids = ", ".join(t.circuit_id for t in timelines)
            print(f"Timeline(s) {ids} have no events.")
            return

        names = ", ".join(str(t.name) for t in timelines)
        print(f"QDebug timeline — {names} ({len(merged)} events)")
        print("commands: n(ext)  p(rev)  f(irst)  l(ast)  g(oto) N  h(elp)  q(uit)")

        shell = _TimelineShell(merged, tracks)
        shell.print_frame()
        try:
            shell.cmdloop()
        except KeyboardInterrupt:
            print()
