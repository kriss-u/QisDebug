"""Text-based interactive frontend: the universal fallback (e.g. SSH sessions).

Built on the standard library's :mod:`cmd` module rather than a hand-rolled
``input()`` loop, so command parsing, aliasing, and help text come from a
purpose-built interpreter instead of ad hoc string matching. ``cmd`` ships
with Python, so this frontend keeps no extra dependency and stays available
in any environment.
"""

from __future__ import annotations

import cmd
import sys
from dataclasses import dataclass
from typing import ClassVar

from .._breakpoints import BreakpointCondition
from .._events import ExecutionFinishedEvent, TelemetryEvent, Timeline
from . import _breakpoints as _bp
from . import _circuit_diff, _render
from .base import TimelineFrontend

_MARK_SLOTS = ("a", "b")
_DIFF_REMOVED_ON = "\x1b[41;97m"  # red background, bright white text
_DIFF_ADDED_ON = "\x1b[42;30m"  # green background, black text
_DIFF_HIGHLIGHT_OFF = "\x1b[0m"
_DIFF_REMOVED_TEXT = "\x1b[31m"  # red text
_DIFF_ADDED_TEXT = "\x1b[32m"  # green text


def _highlight_row(row: str, column_ranges: list[tuple[int, int]], color: str) -> str:
    """Wrap the diffing column ranges of one drawing row in a terminal
    background color, when stdout is a tty (plain text otherwise, since
    ANSI codes would just show up as garbage in a redirected/piped output).
    """
    if not column_ranges or not sys.stdout.isatty():
        return row
    pieces = []
    cursor = 0
    for start, end in column_ranges:
        if start >= len(row):
            break
        pieces.append(row[cursor:start])
        pieces.append(color + row[start : min(end, len(row))] + _DIFF_HIGHLIGHT_OFF)
        cursor = end
    pieces.append(row[cursor:])
    return "".join(pieces)


def _highlight_gate_diff_lines(text: str) -> str:
    """Color each :func:`~._render.render_state_diff` unified-diff line
    red/green text (not the background fill used for the circuit drawing
    panes above it, see :func:`_highlight_row` -- these lines are already
    just one gate's description, so a background block reads as more
    alarming than the same red/green shown as plain colored text).
    """
    if not sys.stdout.isatty():
        return text
    lines = []
    for line in text.split("\n"):
        if line.startswith("  - "):
            lines.append(_DIFF_REMOVED_TEXT + line + _DIFF_HIGHLIGHT_OFF)
        elif line.startswith("  + "):
            lines.append(_DIFF_ADDED_TEXT + line + _DIFF_HIGHLIGHT_OFF)
        else:
            lines.append(line)
    return "\n".join(lines)


def _render_diff_side_text(side: _circuit_diff.CircuitDiffSide, color: str) -> str:
    """Render one side of a :class:`~._circuit_diff.CircuitSplitDiff` as its
    own full-width block: a label line followed by the drawing, with the
    columns unique to this side highlighted.
    """
    rows = [
        _highlight_row(row, side.row_ranges, color) for row in side.text.split("\n")
    ]
    return "\n".join([side.label, *rows])


def _render_circuit_split_diff_text(split_diff: _circuit_diff.CircuitSplitDiff) -> str:
    """Render a :class:`~._circuit_diff.CircuitSplitDiff` as two stacked,
    full-width panes -- circuit A on top (red for content only it has),
    circuit B below (green for content only it has). Stacked rather than
    side by side because a typical terminal is nowhere near twice a
    circuit drawing's width, and a narrow terminal wrapping two
    interleaved panes is far harder to read than scrolling past one full
    circuit at a time.
    """
    return "\n\n".join(
        [
            _render_diff_side_text(split_diff.a, _DIFF_REMOVED_ON),
            _render_diff_side_text(split_diff.b, _DIFF_ADDED_ON),
        ]
    )


@dataclass
class _DiffSelection:
    """Which element (``"<circuit>:<track>"``, e.g. ``"bell:transpiled"``) to
    diff on each side, and whether to show the full per-pair diff or the
    "overall" one-line-per-element summary."""

    overall: bool = False
    element_a: str | None = None
    element_b: str | None = None


class _TimelineShell(cmd.Cmd):  # pylint: disable=too-many-instance-attributes
    """Line-oriented REPL for stepping through a merged timeline.

    One attribute per independent interactive feature (watches, marks/diff
    selection, breakpoints) plus the underlying merged-timeline state --
    the same shape as the GUI's ``_TimelineWindow``, which carries the same
    disable for the same reason.
    """

    prompt = "qisdebug> "

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
        self.marks: dict[str, _render.StageMark | None] = {"a": None, "b": None}
        self.diff_selection = _DiffSelection()
        self.breakpoints = _bp.FrontendBreakpoints()

    def print_frame(self) -> None:
        """Print the currently selected event and its circuit/state/watch/diff panels."""
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
        watch_text = _render.render_watches(
            self.watches, self.timelines_by_id, event.seq
        )
        if watch_text:
            print()
            print("Watches:")
            print(watch_text)
        diff_text = self._render_diff()
        if diff_text:
            print()
            print("-- Diff A vs B --")
            print(diff_text)

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
        """w, watch qubit N | watch clbit N | watch register NAME |
        watch track TRACK: track it from the current circuit at every later
        step, until unwatched. A qubit/clbit/register watch reports
        gates-touched on every circuit lineage (construction, transpiled,
        backend-compiled) automatically; ``watch track TRACK`` instead
        watches one lineage's transpile pass stats directly (see
        ``elements`` for the available track names)."""
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
        """unwatch qubit N | unwatch clbit N | unwatch register NAME |
        unwatch track TRACK | unwatch all"""
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

    def do_mark(self, arg: str) -> bool:
        """mark a | mark b: pin the current event's point in time as that
        side of a diff. A stage can hold more than one circuit, and each
        circuit more than one track -- its construction, plus one per
        transpile run (see "elements a"/"elements b" to list what a pinned
        stage actually has). Use "diff <element_a> <element_b>" to pick
        which one on each side if a stage is ambiguous. Once both are
        pinned, the diff between them appears automatically."""
        slot = arg.strip()
        if slot not in _MARK_SLOTS:
            print("usage: mark a | mark b")
            return False
        _source, event = self.merged[self.index]
        mark = _render.StageMark(seq=event.seq)
        self.marks[slot] = mark
        print(f"marked {slot}: {mark.label(self.timelines_by_id)}")
        elements = mark.elements_at(self.timelines_by_id)
        if len(elements) > 1:
            print(f"  elements available: {', '.join(elements)}")
        self.print_frame()
        return False

    def do_elements(self, arg: str) -> bool:
        """elements a | elements b: list the "<circuit>:<track>" elements
        available at that mark's stage -- each circuit's construction track
        plus one per transpile run -- for use with "diff <a> <b>"."""
        slot = arg.strip()
        if slot not in _MARK_SLOTS:
            print("usage: elements a | elements b")
            return False
        mark = self.marks[slot]
        if mark is None:
            print(f"{slot} is not marked (see: mark {slot})")
            return False
        elements = mark.elements_at(self.timelines_by_id)
        if not elements:
            print("(no elements)")
        for element in elements:
            print(f"  {element}")
        return False

    def do_unmark(self, arg: str) -> bool:
        """unmark a | unmark b | unmark all"""
        target = arg.strip()
        if target == "all":
            self.marks = {"a": None, "b": None}
            self.diff_selection = _DiffSelection()
            print("cleared all marks")
            self.print_frame()
            return False
        if target not in _MARK_SLOTS:
            print("usage: unmark a | unmark b | unmark all")
            return False
        self.marks[target] = None
        if target == "a":
            self.diff_selection.element_a = None
        else:
            self.diff_selection.element_b = None
        print(f"unmarked {target}")
        self.print_frame()
        return False

    def do_diff(self, arg: str) -> bool:
        """d, diff: print the diff between marks A and B on demand, without
        stepping. Also shown automatically after every frame once both are
        marked. The header always names the track each side came from
        (e.g. "[transpiled]"), so it's never ambiguous whether a diff used
        the raw construction gates or a transpiled circuit.
        diff overall: one summary line per element common to both stages,
        instead of the full detail for a single chosen pair.
        diff <element_a> <element_b>: pick which element to compare on each
        side (see "elements a"/"elements b" for the options) when a stage
        holds more than one; remembered for later frames until changed or
        unmarked."""
        parts = arg.split()
        if parts == ["overall"]:
            self.diff_selection.overall = True
        elif len(parts) == 2:
            self.diff_selection.element_a, self.diff_selection.element_b = parts
            self.diff_selection.overall = False
        elif parts:
            print("usage: diff | diff overall | diff <element_a> <element_b>")
            return False
        else:
            self.diff_selection.overall = False
        diff_text = self._render_diff()
        if diff_text:
            print("-- Diff A vs B --")
            print(diff_text)
        else:
            print("mark both a and b to compare (see: mark a, mark b)")
        return False

    do_d = do_diff

    def _render_diff(self) -> str:
        """Render the current A/B diff per ``self.diff_selection``: any
        active watches' values at A vs B first (so a change in watched
        state is the first thing seen, not buried below the full circuit
        diagram), then the split-view circuit diagram diff (what the two
        circuits actually look like, side by side), then the gate/state-level
        summary.
        """
        mark_a, mark_b = self.marks["a"], self.marks["b"]
        if mark_a is None or mark_b is None:
            return ""
        sections = []
        watch_diff_text = _render.render_watch_diff(
            self.watches, self.timelines_by_id, mark_a.seq, mark_b.seq
        )
        if watch_diff_text:
            sections.append(
                "-- Watch diff --\n" + _highlight_gate_diff_lines(watch_diff_text)
            )
        if self.diff_selection.overall:
            sections.append(
                _render.render_stage_overall_diff(mark_a, mark_b, self.timelines_by_id)
            )
            return "\n\n".join(sections)
        element_a = self.diff_selection.element_a
        element_b = self.diff_selection.element_b
        # Identical to the GUI's call below -- both frontends resolving the
        # same (mark_a, mark_b, element_a, element_b) pick through the one
        # shared function is the intended fix for what used to be duplicated
        # *logic*, not a second instance of the duplication itself.
        # pylint: disable=duplicate-code
        resolved = _circuit_diff.resolve_split_and_stage_diff(
            mark_a,
            mark_b,
            self.timelines_by_id,
            element_a=element_a,
            element_b=element_b,
        )
        if isinstance(resolved, str):
            sections.append(resolved)
            return "\n\n".join(sections)
        # pylint: enable=duplicate-code
        split_diff, stage_diff_text = resolved
        sections.append(_render_circuit_split_diff_text(split_diff))
        sections.append(_highlight_gate_diff_lines(stage_diff_text))
        return "\n\n".join(sections)

    def do_break(self, arg: str) -> bool:
        """b, break gate NAME [Q1 Q2 ...] | break seq N | break watch qubit|clbit|register
        REF gt|lt|ge|le|eq|ne THRESHOLD | break anomaly [info|warning|error]
        Add a conditional breakpoint, scoped to the circuit currently on
        screen. "continue" jumps to the first later event matching any
        breakpoint."""
        condition = self._parse_break_arg(arg)
        if condition is None:
            return False
        handle = self.breakpoints.add(condition)
        print(f"breakpoint {handle}: {_bp.describe_condition(condition)}")
        return False

    do_b = do_break

    def do_breakpoints(self, _arg: str) -> bool:
        """bp, breakpoints: list active breakpoints."""
        if not self.breakpoints:
            print("no active breakpoints")
            return False
        for handle, condition in self.breakpoints.items():
            print(f"  [{handle}] {_bp.describe_condition(condition)}")
        return False

    do_bp = do_breakpoints

    def do_unbreak(self, arg: str) -> bool:
        """unbreak HANDLE | unbreak all"""
        target = arg.strip()
        if target == "all":
            self.breakpoints.clear()
            print("cleared all breakpoints")
            return False
        if target not in self.breakpoints:
            print(f"no breakpoint {target!r} (see: breakpoints)")
            return False
        self.breakpoints.remove(target)
        print(f"removed breakpoint {target}")
        return False

    def do_continue(self, _arg: str) -> bool:
        """c, continue: jump to the next event matching any breakpoint."""
        if not self.breakpoints:
            print("no active breakpoints (see: break)")
            return False
        hit = _bp.find_next_hit(self.merged, self.breakpoints.as_dict(), self.index)
        if hit is None:
            print("no later event matches any breakpoint")
            return False
        index, handle = hit
        self.index = index
        condition = self.breakpoints.get(handle)
        assert condition is not None
        print(f"hit breakpoint {handle}: {_bp.describe_condition(condition)}")
        self.print_frame()
        return False

    do_c = do_continue

    def _parse_break_arg(self, arg: str) -> BreakpointCondition | None:
        """Parse one ``break`` command's arguments into a condition, scoped
        to the circuit currently on screen. Delegates the actual tokenizing
        to :func:`~._breakpoints.parse_condition_args`, shared with the GUI's
        condition-builder row so both accept exactly the same syntax.
        """
        parts = arg.split()
        if not parts:
            print(f"usage: break {_bp.CONDITION_USAGE}")
            return None
        source, _event = self.merged[self.index]
        try:
            return _bp.parse_condition_args(parts[0], parts[1:], source.circuit_id)
        except ValueError as exc:
            print(str(exc))
            return None

    def _parse_watch_arg(self, arg: str, command: str) -> _render.WatchTarget | None:
        """Parse ``"<kind> <ref>"`` into a watch target on the current circuit."""
        parts = arg.split(maxsplit=1)
        usage = (
            f"usage: {command} qubit N | {command} clbit N | "
            f"{command} register NAME | {command} track TRACK"
        )
        if len(parts) != 2 or parts[0] not in ("qubit", "clbit", "register", "track"):
            print(usage)
            return None
        kind, ref = parts[0], parts[1].strip()
        if kind in ("qubit", "clbit") and not ref.isdigit():
            print(f"{kind} must be a bit index, got {ref!r}")
            return None
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
        print(f"QisDebug timeline — {names} ({len(merged)} events)")
        print(
            "commands: n(ext)  p(rev)  f(irst)  l(ast)  g(oto) N  "
            "w(atch)/unwatch qubit|clbit|register|track ...  watches  "
            "mark a|b  unmark a|b|all  elements a|b  "
            "d(iff) [overall | <element_a> <element_b>]  "
            "b(reak) gate|seq|watch|anomaly ...  bp  unbreak HANDLE|all  c(ontinue)  "
            "h(elp)  q(uit)"
        )

        timelines_by_id = {t.circuit_id: t for t in timelines}
        shell = _TimelineShell(merged, tracks, timelines_by_id)
        shell.print_frame()
        try:
            shell.cmdloop()
        except KeyboardInterrupt:
            print()
