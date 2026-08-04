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
        timelines_by_id: dict[str, Timeline],
    ) -> None:
        super().__init__()
        self.merged = merged
        self.tracks = tracks
        self.timelines_by_id = timelines_by_id
        self.index = 0
        self.watches: list[_render.WatchTarget] = []

    def print_frame(self) -> None:
        """Print the currently selected event and its circuit/state/watch panels."""
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
        watch_text = _render.render_watches(self.watches, self.timelines_by_id, event.seq)
        if watch_text:
            print()
            print("Watches:")
            print(watch_text)

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

    def do_watch(self, arg: str) -> bool:
        """w, watch qubit N | watch clbit N | watch register NAME: track it from
        the current circuit at every later step, until unwatched."""
        target = self._parse_watch_arg(arg, "watch")
        if target is None:
            return False
        if target not in self.watches:
            self.watches.append(target)
        source, _event = self.merged[self.index]
        print(f"watching {target.label(source)}")
        self.print_frame()
        return False

    do_w = do_watch

    def do_unwatch(self, arg: str) -> bool:
        """unwatch qubit N | unwatch clbit N | unwatch register NAME | unwatch all"""
        if arg.strip() == "all":
            self.watches.clear()
            print("cleared all watches")
            self.print_frame()
            return False
        target = self._parse_watch_arg(arg, "unwatch")
        if target is None:
            return False
        if target in self.watches:
            self.watches.remove(target)
            print(f"unwatched {target.ref}")
        else:
            print("not watching that")
        self.print_frame()
        return False

    def do_watches(self, _arg: str) -> bool:
        """List currently active watches."""
        if not self.watches:
            print("no active watches")
            return False
        for target in self.watches:
            print(f"  {target.label(self.timelines_by_id.get(target.circuit_id))}")
        return False

    def _parse_watch_arg(self, arg: str, command: str) -> _render.WatchTarget | None:
        """Parse ``"<kind> <ref>"`` into a watch target on the current circuit."""
        parts = arg.split(maxsplit=1)
        usage = f"usage: {command} qubit N | {command} clbit N | {command} register NAME"
        if len(parts) != 2 or parts[0] not in ("qubit", "clbit", "register"):
            print(usage)
            return None
        kind, ref = parts[0], parts[1].strip()
        if kind in ("qubit", "clbit") and not ref.isdigit():
            print(f"{kind} must be a bit index, got {ref!r}")
            return None
        assert kind in ("qubit", "clbit", "register")
        source, _event = self.merged[self.index]
        return _render.WatchTarget(circuit_id=source.circuit_id, kind=kind, ref=ref)

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
        print(
            "commands: n(ext)  p(rev)  f(irst)  l(ast)  g(oto) N  "
            "w(atch)/unwatch qubit|clbit|register ...  watches  h(elp)  q(uit)"
        )

        timelines_by_id = {t.circuit_id: t for t in timelines}
        shell = _TimelineShell(merged, tracks, timelines_by_id)
        shell.print_frame()
        try:
            shell.cmdloop()
        except KeyboardInterrupt:
            print()
