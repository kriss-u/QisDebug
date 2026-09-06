"""Text-based frontend: the universal fallback (e.g. SSH sessions).

On a real terminal (both stdin and stdout attached to a tty) this launches
:class:`~._tui.TimelineApp`, a full-screen `textual` UI with persistent
panels for the timeline, circuit/event detail, watches, breakpoints, and
diff -- the GUI's always-visible-panel experience, not a scrolling
transcript, entirely keyboard-operable since a remote terminal may not
deliver mouse events at all.

Anywhere else -- piped/redirected stdin or stdout, a plain Jupyter notebook
with no real pty, or a test harness monkeypatching ``builtins.input`` (which
reports ``sys.stdin.isatty() is False``) -- there's no terminal for a
full-screen app to control, so :func:`_run_fallback_loop` drives the same
:class:`~._cli_session.CliSession` through a plain print/input loop instead.
Both paths accept the identical command language; only the richness of the
live display differs.
"""

from __future__ import annotations

import sys
from typing import ClassVar

from rich.console import Console

from .._events import Timeline
from ._cli_session import CliSession
from .base import TimelineFrontend


def _print_frame(console: Console, session: CliSession) -> None:
    """Print the currently selected event and its circuit/state/watch/diff
    panels -- the fallback loop's equivalent of the TUI's
    :meth:`~._tui.TimelineApp.refresh_panels`.
    """
    # The real terminal width, so the counts histogram's bars scale to
    # actually fit it rather than assuming a fixed width always does.
    info_width = max(20, console.size.width - 2)
    frame = session.current_frame(info_width=info_width)
    console.print()
    console.print(
        f"[bold]\\[{frame.index + 1}/{frame.total}][/bold] {frame.circuit_name} — "
        f"{frame.event_type} at {frame.timestamp:.3f} — {frame.description}"
    )
    console.print(frame.tracks_text)
    if frame.other_circuits:
        names_not_shown = ", ".join(frame.other_circuits)
        console.print(f"(+{len(frame.other_circuits)} more not shown: {names_not_shown})")
    console.print(frame.event_text)
    if frame.histogram_text:
        console.print()
        console.print("Counts histogram:")
        console.print(frame.histogram_text)
    if frame.bloch_text:
        console.print()
        console.print(frame.bloch_text)
    watch_text = session.render_watches()
    if watch_text:
        console.print()
        console.print("Watches:")
        console.print(watch_text)
    diff_status = session.diff_status_line()
    if diff_status is not None:
        console.print()
        console.print(diff_status + "  (see: diff full)")


def _run_fallback_loop(console: Console, session: CliSession, names: str) -> None:
    """Drive ``session`` through a plain print/input loop, for any context
    with no real terminal to run the full-screen TUI in.
    """
    console.print(f"QisDebug timeline — {names} ({len(session.merged)} events)")
    console.print("type h(elp) for the full command list, q(uit) to exit")
    _print_frame(console, session)
    while not session.done:
        try:
            line = input("qisdebug> ")
        except EOFError:
            console.print()
            break
        except KeyboardInterrupt:
            console.print()
            break
        outcome = session.execute(line)
        for message in outcome.messages:
            console.print(message)
        if outcome.refresh:
            _print_frame(console, session)


class CLIFrontend(TimelineFrontend):
    """The universal fallback: a full-screen TUI on a real terminal, a plain
    print/input loop everywhere else."""

    name: ClassVar[str] = "cli"

    def is_available(self) -> bool:
        return True

    def unavailable_reason(self) -> str:
        return ""

    def show(self, timelines: list[Timeline]) -> None:
        console = Console()
        if not any(t.events for t in timelines):
            ids = ", ".join(t.circuit_id for t in timelines)
            console.print(f"Timeline(s) {ids} have no events.")
            return

        session = CliSession(timelines)
        names = ", ".join(str(t.name) for t in timelines)

        if sys.stdin.isatty() and sys.stdout.isatty():
            from ._tui import TimelineApp  # pylint: disable=import-outside-toplevel

            TimelineApp(session, f"{names} ({len(session.merged)} events)").run()
            return
        _run_fallback_loop(console, session, names)


__all__ = ["CLIFrontend"]
