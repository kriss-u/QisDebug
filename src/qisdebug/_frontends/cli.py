"""Text-based interactive frontend: the universal fallback (e.g. SSH sessions).

Built on :mod:`rich` for rendering and, on a real terminal, `prompt_toolkit`
for the input loop (history, tab-completion) -- a purpose-built line editor
and renderer instead of ad hoc ``print``/``input`` calls and hand-rolled ANSI
codes. Neither library is optional here: unlike the GUI frontend, this one is
the guaranteed fallback (see :mod:`._detect`), so both are core dependencies.

Non-interactive input (piped stdin, or a test harness monkeypatching
``builtins.input``, which reports ``sys.stdin.isatty() is False``) never
touches `prompt_toolkit`'s ``PromptSession`` -- there is no terminal for it
to control, and ``input()`` is what such callers already intercept. Only a
real tty gets history/completion; a piped session gets the same plain
``input(prompt)`` loop either way, since there is no line editing to offer it.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable, ClassVar, Iterable

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.table import Table
from rich.text import Text

from .._breakpoints import BreakpointCondition
from .._events import ExecutionFinishedEvent, TelemetryEvent, Timeline
from . import _breakpoints as _bp
from . import _circuit_diff, _render
from .base import TimelineFrontend

_MARK_SLOTS = ("a", "b")
_REMOVED_STYLE = "white on red"
_ADDED_STYLE = "black on green"
_REMOVED_TEXT_STYLE = "red"
_ADDED_TEXT_STYLE = "green"

_COMMAND_NAMES = (
    "next", "n", "prev", "p", "first", "f", "last", "l", "goto", "g",
    "watch", "w", "unwatch", "watches", "mark", "elements", "unmark",
    "diff", "d", "break", "b", "breakpoints", "bp", "unbreak",
    "continue", "c", "hitfirst", "hf", "hitprev", "hp", "hitlast", "hl",
    "help", "h", "quit", "q",
)  # fmt: skip

_HELP_TEXT = """\
n(ext)                         advance to the next event
p(rev)                         go back to the previous event
f(irst)                        jump to the first event
l(ast)                         jump to the last event
g(oto) N                       jump to event N
w(atch) qubit|clbit|register|track REF
                                track REF from the current circuit
unwatch qubit|clbit|register|track REF | unwatch all
watches                        list active watches
mark a | mark b                pin the current event as that diff slot
elements a | elements b        list "<circuit>:<track>" elements at a mark
unmark a | unmark b | unmark all
d(iff) [overall | <element_a> <element_b>]
                                show/select the diff between marks A and B
b(reak) gate|seq|watch|expect|prob|tvd|anomaly ...
                                add a breakpoint (Tab-completes on a tty)
bp, breakpoints                 list active breakpoints
unbreak HANDLE | unbreak all
c(ontinue)                     jump to the next event matching any breakpoint
hf(irst) | hp(rev) | hl(ast)   jump to the first/previous/last matching event
h(elp)                         this text
q(uit)                         exit
"""


def _highlight_row(row: str, column_ranges: list[tuple[int, int]], style: str) -> Text:
    """Wrap ``column_ranges`` of one drawing row in a rich style, leaving the
    rest of the row plain. Rich strips styling on its own when the
    destination isn't a terminal, so unlike the ANSI codes this replaces,
    no manual ``isatty`` gating is needed here.
    """
    text = Text(row)
    for start, end in column_ranges:
        if start >= len(row):
            break
        text.stylize(style, start, min(end, len(row)))
    return text


def _highlight_gate_diff_lines(text: str) -> Text:
    """Color each :func:`~._render.render_state_diff` unified-diff line
    red/green text (not a background fill the way :func:`_highlight_row`
    marks a circuit drawing -- these lines are already just one gate's
    description, so a background block reads as more alarming than the same
    red/green shown as plain colored text).
    """
    result = Text()
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("  - "):
            result.append(line, style=_REMOVED_TEXT_STYLE)
        elif line.startswith("  + "):
            result.append(line, style=_ADDED_TEXT_STYLE)
        else:
            result.append(line)
        if i != len(lines) - 1:
            result.append("\n")
    return result


def _render_diff_side(side: _circuit_diff.CircuitDiffSide, style: str) -> Text:
    """Render one side of a :class:`~._circuit_diff.CircuitSplitDiff` as its
    own full-width block: a label line followed by the drawing, with the
    columns unique to this side highlighted.
    """
    result = Text(side.label + "\n")
    rows = side.text.split("\n")
    for i, row in enumerate(rows):
        result.append_text(_highlight_row(row, side.row_ranges, style))
        if i != len(rows) - 1:
            result.append("\n")
    return result


def _render_circuit_split_diff(split_diff: _circuit_diff.CircuitSplitDiff) -> Text:
    """Render a :class:`~._circuit_diff.CircuitSplitDiff` as two stacked,
    full-width panes -- circuit A on top (red for content only it has),
    circuit B below (green for content only it has). Stacked rather than
    side by side because a typical terminal is nowhere near twice a
    circuit drawing's width, and a narrow terminal wrapping two interleaved
    panes is far harder to read than scrolling past one full circuit at a
    time.
    """
    result = _render_diff_side(split_diff.a, _REMOVED_STYLE)
    result.append("\n\n")
    result.append_text(_render_diff_side(split_diff.b, _ADDED_STYLE))
    return result


@dataclass
class _DiffSelection:
    """Which element (``"<circuit>:<track>"``, e.g. ``"bell:transpiled"``) to
    diff on each side, and whether to show the full per-pair diff or the
    "overall" one-line-per-element summary."""

    overall: bool = False
    element_a: str | None = None
    element_b: str | None = None


class _TimelineShell:
    # pylint: disable=too-many-instance-attributes,too-many-public-methods
    """Line-oriented REPL for stepping through a merged timeline.

    One attribute per independent interactive feature (watches, marks/diff
    selection, breakpoints) plus the underlying merged-timeline state -- the
    same shape as the GUI's ``_TimelineWindow``, which carries the same
    disable for the same reason. Commands dispatch through ``self._commands``,
    a plain ``{name: bound method}`` table, rather than a framework's own
    dispatch convention -- this class owns nothing framework-specific, so it
    stays exercisable directly from tests without going through a REPL loop.
    Public-method count is inflated by one method per command (each with its
    aliases sharing one method) rather than by too many independent
    responsibilities.
    """

    def __init__(
        self,
        console: Console,
        merged: list[tuple[Timeline, TelemetryEvent]],
        tracks: list[dict[str, str]],
        timelines_by_id: dict[str, Timeline],
    ) -> None:
        self.console = console
        self.merged = merged
        self.tracks = tracks
        self.timelines_by_id = timelines_by_id
        self.index = 0
        self.watches: list[_render.WatchTarget] = []
        self.marks: dict[str, _render.StageMark | None] = {"a": None, "b": None}
        self.diff_selection = _DiffSelection()
        self.breakpoints = _bp.FrontendBreakpoints()
        self.done = False
        self._commands: dict[str, Callable[[str], None]] = {
            "next": self.do_next, "n": self.do_next,
            "prev": self.do_prev, "p": self.do_prev,
            "first": self.do_first, "f": self.do_first,
            "last": self.do_last, "l": self.do_last,
            "goto": self.do_goto, "g": self.do_goto,
            "watch": self.do_watch, "w": self.do_watch,
            "unwatch": self.do_unwatch,
            "watches": self.do_watches,
            "mark": self.do_mark,
            "elements": self.do_elements,
            "unmark": self.do_unmark,
            "diff": self.do_diff, "d": self.do_diff,
            "break": self.do_break, "b": self.do_break,
            "breakpoints": self.do_breakpoints, "bp": self.do_breakpoints,
            "unbreak": self.do_unbreak,
            "continue": self.do_continue, "c": self.do_continue,
            "hitfirst": self.do_hitfirst, "hf": self.do_hitfirst,
            "hitprev": self.do_hitprev, "hp": self.do_hitprev,
            "hitlast": self.do_hitlast, "hl": self.do_hitlast,
            "help": self.do_help, "h": self.do_help,
            "quit": self.do_quit, "q": self.do_quit,
        }  # fmt: skip

    def dispatch(self, line: str) -> None:
        """Parse and run one command line, or repeat "next" on a bare Enter."""
        stripped = line.strip()
        if not stripped:
            self.do_next("")
            return
        name, _, arg = stripped.partition(" ")
        handler = self._commands.get(name)
        if handler is None:
            self.console.print(f"unknown command: {name!r} (see: help)")
            return
        handler(arg.strip())

    def print_frame(self) -> None:
        """Print the currently selected event and its circuit/state/watch/diff panels."""
        source, event = self.merged[self.index]
        self.console.print()
        self.console.print(
            f"[bold]\\[{self.index + 1}/{len(self.merged)}][/bold] {source.name} — "
            f"{event.type.value} at {event.timestamp:.3f} — {event.describe()}"
        )
        self.console.print(_render.render_tracks_text(self.tracks[self.index]))
        self.console.print(_render.render_event(source, event))
        if isinstance(event, ExecutionFinishedEvent) and event.counts:
            self.console.print()
            self.console.print("Counts histogram:")
            self.console.print(_render.render_histogram_ascii(event.counts))
        watch_text = _render.render_watches(
            self.watches, self.timelines_by_id, event.seq
        )
        if watch_text:
            self.console.print()
            self.console.print("Watches:")
            self.console.print(watch_text)
        diff_content = self._render_diff()
        if diff_content is not None:
            self.console.print()
            self.console.print("-- Diff A vs B --")
            self.console.print(diff_content)

    def do_next(self, _arg: str) -> None:
        """n, next: advance to the next event."""
        self.index = min(self.index + 1, len(self.merged) - 1)
        self.print_frame()

    def do_prev(self, _arg: str) -> None:
        """p, prev: go back to the previous event."""
        self.index = max(self.index - 1, 0)
        self.print_frame()

    def do_first(self, _arg: str) -> None:
        """f, first: jump to the first event."""
        self.index = 0
        self.print_frame()

    def do_last(self, _arg: str) -> None:
        """l, last: jump to the last event."""
        self.index = len(self.merged) - 1
        self.print_frame()

    def do_goto(self, arg: str) -> None:
        """g N, goto N: jump to event N."""
        target = arg.strip()
        if not target.isdigit():
            self.console.print(f"not a valid event number: {target!r}")
            return
        self.index = min(max(int(target) - 1, 0), len(self.merged) - 1)
        self.print_frame()

    def do_quit(self, _arg: str) -> None:
        """q, quit: exit."""
        self.done = True

    def do_watch(self, arg: str) -> None:
        """w N, watch qubit N | watch clbit N | watch register NAME |
        watch track TRACK: track it from the current circuit at every later
        step, until unwatched."""
        target = self._parse_watch_arg(arg, "watch")
        if target is None:
            return
        if target not in self.watches:
            self.watches.append(target)
        source, _event = self.merged[self.index]
        self.console.print(f"watching {target.label(source)}")
        self.print_frame()

    def do_unwatch(self, arg: str) -> None:
        """unwatch qubit N | unwatch clbit N | unwatch register NAME |
        unwatch track TRACK | unwatch all"""
        if arg.strip() == "all":
            self.watches.clear()
            self.console.print("cleared all watches")
            self.print_frame()
            return
        target = self._parse_watch_arg(arg, "unwatch")
        if target is None:
            return
        if target in self.watches:
            self.watches.remove(target)
            self.console.print(f"unwatched {target.ref}")
        else:
            self.console.print("not watching that")
        self.print_frame()

    def do_watches(self, _arg: str) -> None:
        """List currently active watches."""
        if not self.watches:
            self.console.print("no active watches")
            return
        for target in self.watches:
            self.console.print(f"  {target.label(self.timelines_by_id.get(target.circuit_id))}")

    def do_mark(self, arg: str) -> None:
        """mark a | mark b: pin the current event's point in time as that
        side of a diff. A stage can hold more than one circuit, and each
        circuit more than one track -- its construction, plus one per
        transpile run (see "elements a"/"elements b" to list what a pinned
        stage actually has). Use "diff <element_a> <element_b>" to pick
        which one on each side if a stage is ambiguous. Once both are
        pinned, the diff between them appears automatically."""
        slot = arg.strip()
        if slot not in _MARK_SLOTS:
            self.console.print("usage: mark a | mark b")
            return
        _source, event = self.merged[self.index]
        mark = _render.StageMark(seq=event.seq)
        self.marks[slot] = mark
        self.console.print(f"marked {slot}: {mark.label(self.timelines_by_id)}")
        elements = mark.elements_at(self.timelines_by_id)
        if len(elements) > 1:
            self.console.print(f"  elements available: {', '.join(elements)}")
        self.print_frame()

    def do_elements(self, arg: str) -> None:
        """elements a | elements b: list the "<circuit>:<track>" elements
        available at that mark's stage -- each circuit's construction track
        plus one per transpile run -- for use with "diff <a> <b>"."""
        slot = arg.strip()
        if slot not in _MARK_SLOTS:
            self.console.print("usage: elements a | elements b")
            return
        mark = self.marks[slot]
        if mark is None:
            self.console.print(f"{slot} is not marked (see: mark {slot})")
            return
        elements = mark.elements_at(self.timelines_by_id)
        if not elements:
            self.console.print("(no elements)")
        for element in elements:
            self.console.print(f"  {element}")

    def do_unmark(self, arg: str) -> None:
        """unmark a | unmark b | unmark all"""
        target = arg.strip()
        if target == "all":
            self.marks = {"a": None, "b": None}
            self.diff_selection = _DiffSelection()
            self.console.print("cleared all marks")
            self.print_frame()
            return
        if target not in _MARK_SLOTS:
            self.console.print("usage: unmark a | unmark b | unmark all")
            return
        self.marks[target] = None
        if target == "a":
            self.diff_selection.element_a = None
        else:
            self.diff_selection.element_b = None
        self.console.print(f"unmarked {target}")
        self.print_frame()

    def do_diff(self, arg: str) -> None:
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
            self.console.print("usage: diff | diff overall | diff <element_a> <element_b>")
            return
        else:
            self.diff_selection.overall = False
        diff_content = self._render_diff()
        if diff_content is not None:
            self.console.print("-- Diff A vs B --")
            self.console.print(diff_content)
        else:
            self.console.print("mark both a and b to compare (see: mark a, mark b)")

    def _render_diff(self) -> Text | str | None:
        """Render the current A/B diff per ``self.diff_selection``: any
        active watches' values at A vs B first (so a change in watched state
        is the first thing seen, not buried below the full circuit diagram),
        then the split-view circuit diagram diff (what the two circuits
        actually look like, side by side), then the gate/state-level
        summary. Returns None when neither mark is set yet.
        """
        mark_a, mark_b = self.marks["a"], self.marks["b"]
        if mark_a is None or mark_b is None:
            return None
        sections: list[Text | str] = []
        watch_diff_text = _render.render_watch_diff(
            self.watches, self.timelines_by_id, mark_a.seq, mark_b.seq
        )
        if watch_diff_text:
            header = Text("-- Watch diff --\n")
            header.append_text(_highlight_gate_diff_lines(watch_diff_text))
            sections.append(header)
        if self.diff_selection.overall:
            sections.append(
                _render.render_stage_overall_diff(mark_a, mark_b, self.timelines_by_id)
            )
            return _join_sections(sections)
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
            return _join_sections(sections)
        # pylint: enable=duplicate-code
        split_diff, stage_diff_text = resolved
        sections.append(_render_circuit_split_diff(split_diff))
        sections.append(_highlight_gate_diff_lines(stage_diff_text))
        return _join_sections(sections)

    def do_break(self, arg: str) -> None:
        """b, break gate NAME [Q1 Q2 ...] | break seq CMP N |
        break watch qubit|clbit|register REF CMP THRESHOLD |
        break expect X|Y|Z QUBIT CMP THRESHOLD | break prob OUTCOME CMP THRESHOLD |
        break tvd OUTCOME=PROB [OUTCOME=PROB ...] CMP THRESHOLD |
        break anomaly [info|warning|error]
        CMP is one of >= <= == != > <. Every kind may end with 'on CIRCUIT'
        (or 'on *' for gate/seq/anomaly) to scope elsewhere than the circuit
        currently on screen, the default. Tab-completion is available on a
        real terminal after 'break '. hf/hp/c(ontinue)/hl jump to the
        first/previous/next/last event matching any breakpoint."""
        condition = self._parse_break_arg(arg)
        if condition is None:
            return
        handle = self.breakpoints.add(condition)
        self.console.print(f"breakpoint {handle}: {_bp.describe_condition(condition)}")

    def do_breakpoints(self, _arg: str) -> None:
        """bp, breakpoints: list active breakpoints."""
        if not self.breakpoints:
            self.console.print("no active breakpoints")
            return
        table = Table(show_header=True, header_style="bold")
        table.add_column("handle")
        table.add_column("condition")
        for handle, condition in self.breakpoints.items():
            table.add_row(handle, _bp.describe_condition(condition))
        self.console.print(table)

    def do_unbreak(self, arg: str) -> None:
        """unbreak HANDLE | unbreak all"""
        target = arg.strip()
        if target == "all":
            self.breakpoints.clear()
            self.console.print("cleared all breakpoints")
            return
        if target not in self.breakpoints:
            self.console.print(f"no breakpoint {target!r} (see: breakpoints)")
            return
        self.breakpoints.remove(target)
        self.console.print(f"removed breakpoint {target}")

    def _jump_to_hit(self, hit: tuple[int, str] | None, *, none_message: str) -> None:
        """Jump to a breakpoint hit found by one of the ``find_*_hit``
        functions, or report why there wasn't one. Shared by
        continue/hitfirst/hitprev/hitlast so the four only differ in which
        finder they call.
        """
        if hit is None:
            self.console.print(none_message)
            return
        index, handle = hit
        self.index = index
        condition = self.breakpoints.get(handle)
        assert condition is not None
        self.console.print(f"hit breakpoint {handle}: {_bp.describe_condition(condition)}")
        self.print_frame()

    def do_continue(self, _arg: str) -> None:
        """c, continue: jump to the next event matching any breakpoint."""
        if not self.breakpoints:
            self.console.print("no active breakpoints (see: break)")
            return
        hit = _bp.find_next_hit(self.merged, self.breakpoints.as_dict(), self.index)
        self._jump_to_hit(hit, none_message="no later event matches any breakpoint")

    def do_hitfirst(self, _arg: str) -> None:
        """hf, hitfirst: jump to the earliest event matching any breakpoint."""
        if not self.breakpoints:
            self.console.print("no active breakpoints (see: break)")
            return
        hit = _bp.find_first_hit(self.merged, self.breakpoints.as_dict())
        self._jump_to_hit(hit, none_message="no event matches any breakpoint")

    def do_hitprev(self, _arg: str) -> None:
        """hp, hitprev: jump to the previous event matching any breakpoint."""
        if not self.breakpoints:
            self.console.print("no active breakpoints (see: break)")
            return
        hit = _bp.find_prev_hit(self.merged, self.breakpoints.as_dict(), self.index)
        self._jump_to_hit(hit, none_message="no earlier event matches any breakpoint")

    def do_hitlast(self, _arg: str) -> None:
        """hl, hitlast: jump to the latest event matching any breakpoint."""
        if not self.breakpoints:
            self.console.print("no active breakpoints (see: break)")
            return
        hit = _bp.find_last_hit(self.merged, self.breakpoints.as_dict())
        self._jump_to_hit(hit, none_message="no event matches any breakpoint")

    def do_help(self, _arg: str) -> None:
        """h, help: this text."""
        self.console.print(_HELP_TEXT)

    def _parse_break_arg(self, arg: str) -> BreakpointCondition | None:
        """Parse one ``break`` command's arguments into a condition, scoped
        to the circuit currently on screen. Delegates the actual tokenizing
        to :func:`~._breakpoints.parse_condition_args`, shared with the GUI's
        condition-builder row so both accept exactly the same syntax.
        """
        parts = arg.split()
        if not parts:
            self.console.print(f"usage: break {_bp.CONDITION_USAGE}")
            return None
        source, _event = self.merged[self.index]
        try:
            return _bp.parse_condition_args(parts[0], parts[1:], source.circuit_id)
        except ValueError as exc:
            self.console.print(str(exc))
            return None

    def _parse_watch_arg(self, arg: str, command: str) -> _render.WatchTarget | None:
        """Parse ``"<kind> <ref>"`` into a watch target on the current circuit."""
        parts = arg.split(maxsplit=1)
        usage = (
            f"usage: {command} qubit N | {command} clbit N | "
            f"{command} register NAME | {command} track TRACK"
        )
        if len(parts) != 2 or parts[0] not in ("qubit", "clbit", "register", "track"):
            self.console.print(usage)
            return None
        kind, ref = parts[0], parts[1].strip()
        if kind in ("qubit", "clbit") and not ref.isdigit():
            self.console.print(f"{kind} must be a bit index, got {ref!r}")
            return None
        source, _event = self.merged[self.index]
        return _render.WatchTarget(circuit_id=source.circuit_id, kind=kind, ref=ref)

    def break_completions(self, tokens: list[str]) -> list[str]:
        """Suggestions for the token after ``break``/``b``, given everything
        already typed on the line after the command name. Shared by the
        `prompt_toolkit` completer below with what ``_breakpoints`` already
        offers the GUI, so tab-completion never drifts from what the parser
        would go on to accept.
        """
        source, _event = self.merged[self.index]
        timeline = self.timelines_by_id.get(source.circuit_id)
        circuits = _bp.circuit_choices(self.timelines_by_id)
        if not tokens or tokens[0] not in _bp.KINDS:
            return list(_bp.KINDS)
        return _bp.suggest_tokens(tokens[0], tokens[1:], timeline, circuits)


def _join_sections(sections: list[Text | str]) -> Text | str:
    """Join rendered diff sections with a blank line, preserving rich
    ``Text`` styling when any section carries it (plain ``str`` otherwise).
    """
    if not any(isinstance(s, Text) for s in sections):
        return "\n\n".join(str(s) for s in sections)
    joined = Text()
    for i, section in enumerate(sections):
        joined.append_text(section if isinstance(section, Text) else Text(section))
        if i != len(sections) - 1:
            joined.append("\n\n")
    return joined


class _ShellCompleter(Completer):
    """Tab-completion for the interactive prompt: command names at the start
    of a line, and ``break``/``b`` condition tokens (via
    :meth:`_TimelineShell.break_completions`, which delegates to
    :func:`~._breakpoints.suggest_tokens`) after that -- the same source the
    old `cmd.Cmd`-based ``complete_break`` used, so tab-completion behavior
    is unchanged.
    """

    def __init__(self, shell: _TimelineShell) -> None:
        self._shell = shell

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        words = document.text_before_cursor.split(" ")
        if len(words) == 1:
            word = words[0]
            for name in _COMMAND_NAMES:
                if name.startswith(word):
                    yield Completion(name, start_position=-len(word))
            return
        if words[0] not in ("break", "b"):
            return
        word = words[-1]
        for candidate in self._shell.break_completions(words[1:-1]):
            if candidate.startswith(word):
                yield Completion(candidate, start_position=-len(word))


def _make_session(shell: _TimelineShell) -> PromptSession[str]:
    """Build the `prompt_toolkit` session used on a real terminal: in-memory
    command history plus :class:`_ShellCompleter`.
    """
    return PromptSession(history=InMemoryHistory(), completer=_ShellCompleter(shell))


class CLIFrontend(TimelineFrontend):
    """Prints one event at a time and steps through them via stdin commands."""

    name: ClassVar[str] = "cli"

    def is_available(self) -> bool:
        return True

    def unavailable_reason(self) -> str:
        return ""

    def show(self, timelines: list[Timeline]) -> None:
        console = Console()
        merged, tracks, _active_labels = _render.merge_timelines(timelines)
        if not merged:
            ids = ", ".join(t.circuit_id for t in timelines)
            console.print(f"Timeline(s) {ids} have no events.")
            return

        names = ", ".join(str(t.name) for t in timelines)
        console.print(f"QisDebug timeline — {names} ({len(merged)} events)")
        console.print("type h(elp) for the full command list, q(uit) to exit")

        timelines_by_id = {t.circuit_id: t for t in timelines}
        shell = _TimelineShell(console, merged, tracks, timelines_by_id)
        shell.print_frame()

        session = _make_session(shell) if sys.stdin.isatty() else None
        try:
            while not shell.done:
                try:
                    line = session.prompt("qisdebug> ") if session else input("qisdebug> ")
                except EOFError:
                    console.print()
                    break
                except KeyboardInterrupt:
                    console.print()
                    break
                shell.dispatch(line)
        except KeyboardInterrupt:
            console.print()
