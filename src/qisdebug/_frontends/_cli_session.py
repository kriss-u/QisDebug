"""Presentation-free command/session core shared by the interactive TUI
(:mod:`._tui`) and the non-interactive fallback loop (:mod:`.cli`).

Both presentation layers drive the exact same navigation, watch, mark/diff,
and breakpoint state through :class:`CliSession`, so the command language
(``n``, ``mark a``, ``break gate cx``, ...) and its behavior never diverge
between a real terminal and a piped/notebook session -- only how richly the
result gets displayed differs. This mirrors the project's existing rule that
frontend-agnostic computation lives once and gets presented twice (see
``_render.py``, which this module is itself built on).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, cast

from rich.text import Text

from .._breakpoints import BreakpointCondition
from .._events import (
    ExecutionFinishedEvent,
    StateSnapshotEvent,
    TelemetryEvent,
    Timeline,
)
from . import _ascii_charts
from . import _breakpoints as _bp
from . import _circuit_diff, _render

WatchKind = Literal["qubit", "clbit", "register", "track"]

_MARK_SLOTS = ("a", "b")
_WATCH_KINDS = ("qubit", "clbit", "register", "track")
_FOLD_LEVELS: tuple[int, ...] = (40, 60, 80, 120, 200, -1)
# Unlimited fold by default: the circuit draws at its natural width and the
# TUI's circuit panel scrolls horizontally to it (see "#circuit-pane" in
# _tui.py, which sets overflow-x: auto for exactly this), rather than
# wrapping up front. "Zoom out" (zoom_out) then means stepping down this
# ladder to a narrower fold -- compacting the drawing to fit without
# scrolling -- and "zoom in" clamps back at this unlimited end.
_DEFAULT_FOLD_INDEX = len(_FOLD_LEVELS) - 1
_DEFAULT_HISTOGRAM_WIDTH = 44
_TIMELINE_NAME_WIDTH = 10
_TIMELINE_TYPE_WIDTH = 20
_REMOVED_STYLE = "white on red"
_ADDED_STYLE = "black on green"
_REMOVED_TEXT_STYLE = "red"
_ADDED_TEXT_STYLE = "green"

COMMAND_NAMES = (
    "next", "n", "prev", "p", "first", "f", "last", "l", "goto", "g",
    "watch", "w", "unwatch", "watches", "mark", "elements", "unmark",
    "diff", "d", "break", "b", "breakpoints", "bp", "unbreak",
    "continue", "c", "hitfirst", "hf", "hitprev", "hp", "hitlast", "hl",
    "help", "h", "quit", "q",
)  # fmt: skip

_REFRESH_COMMANDS = frozenset(
    {
        "next", "n", "prev", "p", "first", "f", "last", "l", "goto", "g",
        "watch", "w", "unwatch", "mark", "unmark",
        "continue", "c", "hitfirst", "hf", "hitprev", "hp", "hitlast", "hl",
    }
)  # fmt: skip

HELP_TEXT = """\
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
d(iff) [full | hide | overall | <element_a> <element_b>]
                                show/hide/select the diff between marks A and B
b(reak) gate|seq|watch|expect|prob|tvd|anomaly ...
                                add a breakpoint
bp, breakpoints                 list active breakpoints
unbreak HANDLE | unbreak all
c(ontinue)                     jump to the next event matching any breakpoint
hf(irst) | hp(rev) | hl(ast)   jump to the first/previous/last matching event
h(elp)                         this text
q(uit)                         exit

Only the circuit relevant to the current event is drawn by default, and the
diff (once both marks are set) shows as a one-line summary; a presentation
layer decides how "show everything"/"zoom"/"mark" map to keys (the TUI:
x, D, +/-/0, m/u then a/b) -- this text lists the command language, not the
key bindings.
"""


class CommandError(Exception):
    """A user-facing usage/validation error from a session command.

    Caught by :meth:`CliSession.execute` and turned into a message rather
    than propagated -- these always mean "bad input", never a bug in the
    session itself.
    """


@dataclass
class Frame:  # pylint: disable=too-many-instance-attributes
    """Everything about the event currently selected, computed once so both
    presentation layers show identical content -- a plain data bundle, not
    accumulated state, so the usual too-many-attributes smell doesn't apply
    here any more than it would to a wider database row."""

    index: int
    total: int
    circuit_name: str | None
    event_type: str
    timestamp: float
    description: str
    tracks_text: str
    other_circuits: list[str]
    event_text: str
    histogram_text: str | None
    bloch_text: str | None


@dataclass
class CommandOutcome:
    """What running one command line produced: user-facing messages (plain
    strings or, for the ``diff`` highlight, a styled :class:`rich.text.Text`)
    plus whether the full frame should be redrawn afterward -- some commands
    (``break``, ``watches``, ``help``, ...) only ever report a status line
    and never move the playhead or its watched state, so redrawing the whole
    frame for them would just be noise the old REPL never printed either.
    """

    messages: list[str | Text] = field(default_factory=list)
    refresh: bool = False


@dataclass
class _DiffSelection:
    """Which element (``"<circuit>:<track>"``, e.g. ``"bell:transpiled"``) to
    diff on each side, and whether to show the full per-pair diff or the
    "overall" one-line-per-element summary."""

    overall: bool = False
    element_a: str | None = None
    element_b: str | None = None


def _highlight_row(row: str, column_ranges: list[tuple[int, int]], style: str) -> Text:
    """Wrap ``column_ranges`` of one drawing row in a rich style, leaving the
    rest of the row plain."""
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
    side by side because a typical terminal is nowhere near twice a circuit
    drawing's width.
    """
    result = _render_diff_side(split_diff.a, _REMOVED_STYLE)
    result.append("\n\n")
    result.append_text(_render_diff_side(split_diff.b, _ADDED_STYLE))
    return result


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


class CliSession:
    # pylint: disable=too-many-instance-attributes,too-many-public-methods
    """Navigation index, watches, marks/diff selection, and breakpoints for
    one merged timeline -- the same state the GUI's ``_TimelineWindow`` keeps
    (hence the same pylint disable, for the same reason), minus anything
    about how it gets drawn. Every command is exposed twice: a small typed
    method for a presentation layer that wants to drive one action directly
    (e.g. the TUI's key bindings calling :meth:`go_next`), and via
    :meth:`execute`, which parses the same free-text command language the
    REPL always accepted (e.g. a typed ``n``) and returns messages instead of
    printing them.
    """

    def __init__(self, timelines: list[Timeline]) -> None:
        self._timelines = list(timelines)
        self.timelines_by_id = {t.circuit_id: t for t in timelines}
        self.index = 0
        self.watches: list[_render.WatchTarget] = []
        self.marks: dict[str, _render.StageMark | None] = {"a": None, "b": None}
        self.diff_selection = _DiffSelection()
        self.breakpoints = _bp.FrontendBreakpoints()
        self.done = False
        self.show_all_circuits = False
        self.diff_expanded = False
        self._fold_index = _DEFAULT_FOLD_INDEX
        self.merged: list[tuple[Timeline, TelemetryEvent]] = []
        self.tracks: list[dict[str, str]] = []
        self.active_labels: list[str | None] = []
        self._recompute_tracks()
        self._commands: dict[str, Callable[[str], list[str | Text]]] = {
            "next": self._cmd_next, "n": self._cmd_next,
            "prev": self._cmd_prev, "p": self._cmd_prev,
            "first": self._cmd_first, "f": self._cmd_first,
            "last": self._cmd_last, "l": self._cmd_last,
            "goto": self._cmd_goto, "g": self._cmd_goto,
            "watch": self._cmd_watch, "w": self._cmd_watch,
            "unwatch": self._cmd_unwatch,
            "watches": self._cmd_watches,
            "mark": self._cmd_mark,
            "elements": self._cmd_elements,
            "unmark": self._cmd_unmark,
            "diff": self._cmd_diff, "d": self._cmd_diff,
            "break": self._cmd_break, "b": self._cmd_break,
            "breakpoints": self._cmd_breakpoints, "bp": self._cmd_breakpoints,
            "unbreak": self._cmd_unbreak,
            "continue": self._cmd_continue, "c": self._cmd_continue,
            "hitfirst": self._cmd_hitfirst, "hf": self._cmd_hitfirst,
            "hitprev": self._cmd_hitprev, "hp": self._cmd_hitprev,
            "hitlast": self._cmd_hitlast, "hl": self._cmd_hitlast,
            "help": self._cmd_help, "h": self._cmd_help,
            "quit": self._cmd_quit, "q": self._cmd_quit,
        }  # fmt: skip

    # -- current event -------------------------------------------------

    def current_source_event(self) -> tuple[Timeline, TelemetryEvent]:
        """Return the ``(Timeline, TelemetryEvent)`` pair at the playhead."""
        return self.merged[self.index]

    def current_frame(self, *, info_width: int | None = None) -> Frame:
        """Compute everything about the event currently selected.

        ``tracks_text`` shows only the circuit the current event actually
        belongs to by default (see :func:`~._render.split_active_track`);
        :meth:`toggle_show_all_circuits` switches it to the full,
        every-tracked-circuit expansion instead. ``other_circuits`` names
        whatever else is tracked but not drawn -- kept separate from
        ``tracks_text`` rather than appended into it, so a presentation
        layer can show it somewhere that doesn't scroll/zoom with the
        circuit drawing itself (it's metadata about the session, not part
        of the diagram). ``info_width`` is the actual available width of
        whatever panel/terminal is about to display ``histogram_text`` -- a
        presentation layer that knows its real render width (the TUI's
        info-pane widget, the fallback loop's real terminal columns) should
        pass it, so the bar chart scales to fit instead of assuming a fixed
        width always does; omitted, a conservative default is used.
        """
        source, event = self.merged[self.index]
        histogram_text = None
        if isinstance(event, ExecutionFinishedEvent) and event.counts:
            histogram_text = _ascii_charts.render_histogram_ascii(
                event.counts, max_width=info_width or _DEFAULT_HISTOGRAM_WIDTH
            )
        bloch_text = None
        if isinstance(event, StateSnapshotEvent):
            bloch_text = _ascii_charts.render_bloch_ascii(event)
        tracks = self.tracks[self.index]
        if self.show_all_circuits:
            tracks_text = _render.render_tracks_text(tracks)
            other_circuits: list[str] = []
        else:
            tracks_text, other_circuits = _render.split_active_track(
                tracks, self.active_labels[self.index]
            )
        return Frame(
            index=self.index,
            total=len(self.merged),
            circuit_name=source.name,
            event_type=event.type.value,
            timestamp=event.timestamp,
            description=event.describe(),
            tracks_text=tracks_text,
            other_circuits=other_circuits,
            event_text=_render.render_event(source, event),
            histogram_text=histogram_text,
            bloch_text=bloch_text,
        )

    # -- circuit panel: collapse/expand, zoom -----------------------------

    def _recompute_tracks(self) -> None:
        """(Re)derive ``merged``/``tracks``/``active_labels`` at the current
        zoom level. Event order and count never change with the fold width,
        so the playhead index stays valid and only needs clamping the very
        first time, when there may be no events at all yet.
        """
        self.merged, self.tracks, self.active_labels = _render.merge_timelines(
            self._timelines, fold=_FOLD_LEVELS[self._fold_index]
        )
        if self.merged:
            self.index = min(self.index, len(self.merged) - 1)

    def toggle_show_all_circuits(self) -> bool:
        """Flip between showing only the active circuit and every tracked
        circuit's every track at once; returns the new state."""
        self.show_all_circuits = not self.show_all_circuits
        return self.show_all_circuits

    @property
    def fold_width(self) -> int:
        """The circuit drawing's current line-wrap width (``-1`` = unlimited)."""
        return _FOLD_LEVELS[self._fold_index]

    def zoom_in(self) -> int:
        """Widen the circuit drawing (less wrapping); returns the new fold width."""
        self._fold_index = min(self._fold_index + 1, len(_FOLD_LEVELS) - 1)
        self._recompute_tracks()
        return _FOLD_LEVELS[self._fold_index]

    def zoom_out(self) -> int:
        """Narrow the circuit drawing (more wrapping, more compact); returns
        the new fold width."""
        self._fold_index = max(self._fold_index - 1, 0)
        self._recompute_tracks()
        return _FOLD_LEVELS[self._fold_index]

    def zoom_reset(self) -> int:
        """Return the circuit drawing to its default width."""
        self._fold_index = _DEFAULT_FOLD_INDEX
        self._recompute_tracks()
        return _FOLD_LEVELS[self._fold_index]

    def timeline_rows(self, max_width: int | None = None) -> list[str]:
        """One line per merged event, for a persistent timeline panel.

        The circuit-name column is kept narrow (circuit names are typically
        short) while the event-type column stays at its own fixed width --
        the two are sized independently, not both trimmed together. A row
        that still doesn't fit within ``max_width`` (a caller's actual panel
        width; unbounded if omitted) is truncated with a trailing ``…`` --
        visibly marked as cut off, not silently clipped by the widget with
        no sign anything is missing.
        """
        rows = []
        for i, (source, event) in enumerate(self.merged):
            name = source.name or source.circuit_id
            row = (
                f"{i + 1:>4}  {name:<{_TIMELINE_NAME_WIDTH}.{_TIMELINE_NAME_WIDTH}}  "
                f"{event.type.value:<{_TIMELINE_TYPE_WIDTH}.{_TIMELINE_TYPE_WIDTH}}  "
                f"{event.describe()}"
            )
            if max_width is not None and len(row) > max_width:
                row = row[: max(0, max_width - 1)] + "…"
            rows.append(row)
        return rows

    # -- navigation ------------------------------------------------------

    def go_next(self) -> None:
        """Advance to the next event, clamped to the last one."""
        self.index = min(self.index + 1, len(self.merged) - 1)

    def go_prev(self) -> None:
        """Go back to the previous event, clamped to the first one."""
        self.index = max(self.index - 1, 0)

    def go_first(self) -> None:
        """Jump to the first event."""
        self.index = 0

    def go_last(self) -> None:
        """Jump to the last event."""
        self.index = len(self.merged) - 1

    def go_to(self, one_based: int) -> None:
        """Jump to event ``one_based`` (1-indexed, like ``goto N``), clamped
        to the timeline's bounds."""
        self.index = min(max(one_based - 1, 0), len(self.merged) - 1)

    # -- watches -----------------------------------------------------------

    def parse_watch_arg(self, arg: str) -> tuple[WatchKind, str]:
        """Parse ``"<kind> <ref>"`` into a ``(kind, ref)`` pair, or raise
        :class:`CommandError`."""
        parts = arg.split(maxsplit=1)
        usage = "usage: watch|unwatch qubit N | clbit N | register NAME | track TRACK"
        if len(parts) != 2 or parts[0] not in _WATCH_KINDS:
            raise CommandError(usage)
        kind, ref = cast(WatchKind, parts[0]), parts[1].strip()
        if kind in ("qubit", "clbit") and not ref.isdigit():
            raise CommandError(f"{kind} must be a bit index, got {ref!r}")
        return kind, ref

    def watch_completions(self, tokens: list[str]) -> list[str]:
        """Suggestions for the token after ``watch``/``unwatch``, given
        everything already typed on the line after the command name.
        Mirrors :meth:`break_completions`'s contract exactly (tokens in,
        valid next tokens out), so the same completion-driven input popup
        drives both commands.
        """
        if not tokens:
            return list(_WATCH_KINDS)
        if len(tokens) > 1:
            return []
        timeline, event = self.current_source_event()
        refs_by_kind: dict[str, Callable[[Timeline], list[str]]] = {
            "qubit": _bp.qubit_range,
            "clbit": _bp.clbit_range,
            "register": _bp.register_names,
            "track": lambda tl: _render.track_keys_upto(tl, event.seq),
        }
        return refs_by_kind.get(tokens[0], lambda _t: [])(timeline)

    def add_watch(self, kind: WatchKind, ref: str) -> _render.WatchTarget:
        """Start tracking ``kind``/``ref`` from the circuit currently on screen."""
        source, _event = self.current_source_event()
        target = _render.WatchTarget(circuit_id=source.circuit_id, kind=kind, ref=ref)
        if target not in self.watches:
            self.watches.append(target)
        return target

    def remove_watch(self, kind: WatchKind, ref: str) -> _render.WatchTarget | None:
        """Stop tracking ``kind``/``ref`` on the current circuit; ``None`` if
        it wasn't being watched."""
        source, _event = self.current_source_event()
        target = _render.WatchTarget(circuit_id=source.circuit_id, kind=kind, ref=ref)
        if target in self.watches:
            self.watches.remove(target)
            return target
        return None

    def clear_watches(self) -> None:
        """Stop tracking every watch."""
        self.watches.clear()

    def unwatch_completions(self, tokens: list[str]) -> list[str]:
        """Suggestions for the token after ``unwatch``: unlike
        :meth:`watch_completions`, this offers only the kind/ref of an
        *actually active* watch (there's nothing useful about completing
        towards a watch that isn't set), plus ``"all"`` as the lone
        first-token alternative.
        """
        if not tokens:
            return [*dict.fromkeys(target.kind for target in self.watches), "all"]
        if tokens[0] == "all" or len(tokens) > 1:
            return []
        return [target.ref for target in self.watches if target.kind == tokens[0]]

    def watch_labels(self) -> list[str]:
        """One label per active watch, for a persistent watch panel."""
        return [
            target.label(self.timelines_by_id.get(target.circuit_id))
            for target in self.watches
        ]

    def render_watches(self) -> str:
        """Render every active watch's current value, up to the playhead."""
        _source, event = self.current_source_event()
        return _render.render_watches(self.watches, self.timelines_by_id, event.seq)

    # -- marks / diff --------------------------------------------------

    def mark(self, slot: str) -> _render.StageMark:
        """Pin the current event's point in time as diff slot ``slot`` ("a"/"b")."""
        if slot not in _MARK_SLOTS:
            raise CommandError("usage: mark a | mark b")
        _source, event = self.current_source_event()
        stage_mark = _render.StageMark(seq=event.seq)
        self.marks[slot] = stage_mark
        return stage_mark

    def elements(self, slot: str) -> list[str]:
        """List the "<circuit>:<track>" elements available at ``slot``'s stage."""
        if slot not in _MARK_SLOTS:
            raise CommandError("usage: elements a | elements b")
        stage_mark = self.marks[slot]
        if stage_mark is None:
            raise CommandError(f"{slot} is not marked (see: mark {slot})")
        return stage_mark.elements_at(self.timelines_by_id)

    def unmark(self, slot: str) -> None:
        """Clear one mark slot, or ``"all"`` for both plus the diff selection."""
        if slot == "all":
            self.marks = {"a": None, "b": None}
            self.diff_selection = _DiffSelection()
            return
        if slot not in _MARK_SLOTS:
            raise CommandError("usage: unmark a | unmark b | unmark all")
        self.marks[slot] = None
        if slot == "a":
            self.diff_selection.element_a = None
        else:
            self.diff_selection.element_b = None

    def set_diff_overall(self) -> None:
        """Switch the diff panel to the one-line-per-element summary."""
        self.diff_selection.overall = True

    def set_diff_pair(self, element_a: str, element_b: str) -> None:
        """Pick which element to compare on each side of a multi-element stage."""
        self.diff_selection.element_a = element_a
        self.diff_selection.element_b = element_b
        self.diff_selection.overall = False

    def set_diff_element(self, slot: str, element: str) -> None:
        """Pick which element ``slot`` ("a"/"b") diffs, one side at a time --
        for a presentation layer that asks which element to use right when a
        stage is marked (see :meth:`elements`), rather than only after both
        slots are already set the way typed ``diff <a> <b>``/:meth:`set_diff_pair`
        require.
        """
        if slot not in _MARK_SLOTS:
            raise CommandError("usage: mark a | mark b")
        if slot == "a":
            self.diff_selection.element_a = element
        else:
            self.diff_selection.element_b = element
        self.diff_selection.overall = False

    def reset_diff_mode(self) -> None:
        """Switch the diff panel back to the default single-pair detail view,
        without forgetting which elements were picked."""
        self.diff_selection.overall = False

    def diff_ready(self) -> bool:
        """Whether both mark slots are set, i.e. there is something to diff."""
        return self.marks["a"] is not None and self.marks["b"] is not None

    def diff_status_line(self) -> str | None:
        """One-line "diff ready" summary for the collapsed default view,
        naming both marked stages without rendering the (potentially long)
        diff content itself. ``None`` when either mark is unset -- callers
        use that to hide the diff panel entirely rather than show a status
        line for a diff that doesn't exist yet. See :attr:`diff_expanded`
        for the flag a presentation layer toggles to show :meth:`render_diff`'s
        full content instead.
        """
        mark_a, mark_b = self.marks["a"], self.marks["b"]
        if mark_a is None or mark_b is None:
            return None
        return (
            f"diff ready -- A: {mark_a.label(self.timelines_by_id)}   "
            f"B: {mark_b.label(self.timelines_by_id)}"
        )

    def toggle_diff_expanded(self) -> bool:
        """Flip between the collapsed status line and the full diff content;
        returns the new state."""
        self.diff_expanded = not self.diff_expanded
        return self.diff_expanded

    def render_diff(self) -> Text | str | None:
        """Render the current A/B diff, or ``None`` if either mark is unset.

        Any active watches' values at A vs B first (so a change in watched
        state is the first thing seen, not buried below the full circuit
        diagram), then the split-view circuit diagram diff, then the
        gate/state-level summary.
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
        # Identical to the GUI's call -- both frontends resolving the same
        # (mark_a, mark_b, element_a, element_b) pick through the one shared
        # function is the intended fix for what used to be duplicated
        # *logic*, not a second instance of the duplication itself.
        # pylint: disable=duplicate-code
        resolved = _circuit_diff.resolve_split_and_stage_diff(
            mark_a,
            mark_b,
            self.timelines_by_id,
            element_a=self.diff_selection.element_a,
            element_b=self.diff_selection.element_b,
        )
        if isinstance(resolved, str):
            sections.append(resolved)
            return _join_sections(sections)
        # pylint: enable=duplicate-code
        split_diff, stage_diff_text = resolved
        sections.append(_render_circuit_split_diff(split_diff))
        sections.append(_highlight_gate_diff_lines(stage_diff_text))
        return _join_sections(sections)

    # -- breakpoints ------------------------------------------------------

    def parse_break_arg(self, arg: str) -> BreakpointCondition:
        """Parse one ``break`` command's arguments into a condition, scoped
        to the circuit currently on screen."""
        parts = arg.split()
        if not parts:
            raise CommandError(f"usage: break {_bp.CONDITION_USAGE}")
        source, _event = self.current_source_event()
        try:
            return _bp.parse_condition_args(parts[0], parts[1:], source.circuit_id)
        except ValueError as exc:
            raise CommandError(str(exc)) from exc

    def add_breakpoint(self, condition: BreakpointCondition) -> str:
        """Register a breakpoint condition, returning its handle."""
        return self.breakpoints.add(condition)

    def remove_breakpoint(self, handle: str) -> bool:
        """Unregister a breakpoint by handle; ``False`` if it wasn't active."""
        if handle not in self.breakpoints:
            return False
        self.breakpoints.remove(handle)
        return True

    def clear_breakpoints(self) -> None:
        """Remove every breakpoint."""
        self.breakpoints.clear()

    def unbreak_completions(self, tokens: list[str]) -> list[str]:
        """Suggestions for the token after ``unbreak``: an active
        breakpoint's handle, or ``"all"``."""
        if tokens:
            return []
        return [*(handle for handle, _ in self.breakpoint_rows()), "all"]

    def breakpoint_rows(self) -> list[tuple[str, str]]:
        """``(handle, description)`` per active breakpoint, for a persistent panel."""
        return [
            (handle, _bp.describe_condition(condition))
            for handle, condition in self.breakpoints.items()
        ]

    def break_completions(self, tokens: list[str]) -> list[str]:
        """Suggestions for the token after ``break``/``b``, given everything
        already typed on the line after the command name. Delegates to
        :func:`~._breakpoints.suggest_tokens`, shared with the GUI's own
        condition builder, so completion never drifts from what the parser
        would go on to accept.
        """
        source, _event = self.current_source_event()
        timeline = self.timelines_by_id.get(source.circuit_id)
        circuits = _bp.circuit_choices(self.timelines_by_id)
        if not tokens or tokens[0] not in _bp.KINDS:
            return list(_bp.KINDS)
        return _bp.suggest_tokens(tokens[0], tokens[1:], timeline, circuits)

    def _jump(self, hit: tuple[int, str] | None, *, none_message: str) -> tuple[int, str]:
        if hit is None:
            raise CommandError(none_message)
        self.index = hit[0]
        return hit

    def jump_next_hit(self) -> tuple[int, str]:
        """Jump to the next event matching any breakpoint."""
        if not self.breakpoints:
            raise CommandError("no active breakpoints (see: break)")
        hit = _bp.find_next_hit(self.merged, self.breakpoints.as_dict(), self.index)
        return self._jump(hit, none_message="no later event matches any breakpoint")

    def jump_first_hit(self) -> tuple[int, str]:
        """Jump to the earliest event matching any breakpoint."""
        if not self.breakpoints:
            raise CommandError("no active breakpoints (see: break)")
        hit = _bp.find_first_hit(self.merged, self.breakpoints.as_dict())
        return self._jump(hit, none_message="no event matches any breakpoint")

    def jump_prev_hit(self) -> tuple[int, str]:
        """Jump to the previous event matching any breakpoint."""
        if not self.breakpoints:
            raise CommandError("no active breakpoints (see: break)")
        hit = _bp.find_prev_hit(self.merged, self.breakpoints.as_dict(), self.index)
        return self._jump(hit, none_message="no earlier event matches any breakpoint")

    def jump_last_hit(self) -> tuple[int, str]:
        """Jump to the latest event matching any breakpoint."""
        if not self.breakpoints:
            raise CommandError("no active breakpoints (see: break)")
        hit = _bp.find_last_hit(self.merged, self.breakpoints.as_dict())
        return self._jump(hit, none_message="no event matches any breakpoint")

    # -- free-text command execution ---------------------------------------

    def execute(self, line: str) -> CommandOutcome:
        """Parse and run one command line, returning messages instead of
        printing them, plus whether a presentation layer should redraw the
        whole frame afterward. A bare/whitespace-only line repeats "next",
        matching the historical REPL's blank-Enter behavior.
        """
        stripped = line.strip()
        if not stripped:
            self.go_next()
            return CommandOutcome(refresh=True)
        name, _, arg = stripped.partition(" ")
        handler = self._commands.get(name)
        if handler is None:
            return CommandOutcome([f"unknown command: {name!r} (see: help)"])
        messages = handler(arg.strip())
        return CommandOutcome(messages, refresh=name in _REFRESH_COMMANDS)

    # -- per-command handlers, each returning message lines ----------------

    def _cmd_next(self, _arg: str) -> list[str | Text]:
        self.go_next()
        return []

    def _cmd_prev(self, _arg: str) -> list[str | Text]:
        self.go_prev()
        return []

    def _cmd_first(self, _arg: str) -> list[str | Text]:
        self.go_first()
        return []

    def _cmd_last(self, _arg: str) -> list[str | Text]:
        self.go_last()
        return []

    def _cmd_goto(self, arg: str) -> list[str | Text]:
        if not arg.isdigit():
            return [f"not a valid event number: {arg!r}"]
        self.go_to(int(arg))
        return []

    def _cmd_quit(self, _arg: str) -> list[str | Text]:
        self.done = True
        return []

    def _cmd_watch(self, arg: str) -> list[str | Text]:
        try:
            kind, ref = self.parse_watch_arg(arg)
        except CommandError as exc:
            return [str(exc)]
        target = self.add_watch(kind, ref)
        source, _event = self.current_source_event()
        return [f"watching {target.label(source)}"]

    def _cmd_unwatch(self, arg: str) -> list[str | Text]:
        if arg == "all":
            self.clear_watches()
            return ["cleared all watches"]
        try:
            kind, ref = self.parse_watch_arg(arg)
        except CommandError as exc:
            return [str(exc)]
        removed = self.remove_watch(kind, ref)
        if removed is None:
            return ["not watching that"]
        return [f"unwatched {removed.ref}"]

    def _cmd_watches(self, _arg: str) -> list[str | Text]:
        if not self.watches:
            return ["no active watches"]
        return [f"  {label}" for label in self.watch_labels()]

    def _cmd_mark(self, arg: str) -> list[str | Text]:
        try:
            stage_mark = self.mark(arg)
        except CommandError as exc:
            return [str(exc)]
        messages: list[str | Text] = [f"marked {arg}: {stage_mark.label(self.timelines_by_id)}"]
        elements = stage_mark.elements_at(self.timelines_by_id)
        if len(elements) > 1:
            messages.append(f"  elements available: {', '.join(elements)}")
        return messages

    def _cmd_elements(self, arg: str) -> list[str | Text]:
        try:
            elements = self.elements(arg)
        except CommandError as exc:
            return [str(exc)]
        if not elements:
            return ["(no elements)"]
        return [f"  {element}" for element in elements]

    def _cmd_unmark(self, arg: str) -> list[str | Text]:
        try:
            self.unmark(arg)
        except CommandError as exc:
            return [str(exc)]
        return ["cleared all marks"] if arg == "all" else [f"unmarked {arg}"]

    def _cmd_diff(self, arg: str) -> list[str | Text]:
        """Every form (bare, ``full``, ``overall``, ``<a> <b>``) shows the
        diff immediately -- typing the command is the request to see it now,
        which is also how it expands a persistent panel that otherwise
        defaults to :meth:`diff_status_line`'s collapsed summary (see
        :attr:`diff_expanded`). ``diff hide`` is the one form that instead
        collapses that panel back down.
        """
        parts = arg.split()
        if parts == ["hide"]:
            self.diff_expanded = False
            return ["diff collapsed"] if self.diff_ready() else []
        if parts == ["overall"]:
            self.set_diff_overall()
        elif len(parts) == 2:
            self.set_diff_pair(parts[0], parts[1])
        elif parts and parts != ["full"]:
            return [
                "usage: diff | diff full | diff hide | diff overall | "
                "diff <element_a> <element_b>"
            ]
        else:
            self.reset_diff_mode()
        if not self.diff_ready():
            return ["mark both a and b to compare (see: mark a, mark b)"]
        self.diff_expanded = True
        rendered = self.render_diff()
        return [rendered] if rendered is not None else []

    def _cmd_break(self, arg: str) -> list[str | Text]:
        try:
            condition = self.parse_break_arg(arg)
        except CommandError as exc:
            return [str(exc)]
        handle = self.add_breakpoint(condition)
        return [f"breakpoint {handle}: {_bp.describe_condition(condition)}"]

    def _cmd_breakpoints(self, _arg: str) -> list[str | Text]:
        rows = self.breakpoint_rows()
        if not rows:
            return ["no active breakpoints"]
        return [f"  [{handle}] {description}" for handle, description in rows]

    def _cmd_unbreak(self, arg: str) -> list[str | Text]:
        if arg == "all":
            self.clear_breakpoints()
            return ["cleared all breakpoints"]
        if not self.remove_breakpoint(arg):
            return [f"no breakpoint {arg!r} (see: breakpoints)"]
        return [f"removed breakpoint {arg}"]

    def _describe_hit(self, handle: str) -> str:
        condition = self.breakpoints.get(handle)
        assert condition is not None
        return f"hit breakpoint {handle}: {_bp.describe_condition(condition)}"

    def _cmd_continue(self, _arg: str) -> list[str | Text]:
        try:
            _index, handle = self.jump_next_hit()
        except CommandError as exc:
            return [str(exc)]
        return [self._describe_hit(handle)]

    def _cmd_hitfirst(self, _arg: str) -> list[str | Text]:
        try:
            _index, handle = self.jump_first_hit()
        except CommandError as exc:
            return [str(exc)]
        return [self._describe_hit(handle)]

    def _cmd_hitprev(self, _arg: str) -> list[str | Text]:
        try:
            _index, handle = self.jump_prev_hit()
        except CommandError as exc:
            return [str(exc)]
        return [self._describe_hit(handle)]

    def _cmd_hitlast(self, _arg: str) -> list[str | Text]:
        try:
            _index, handle = self.jump_last_hit()
        except CommandError as exc:
            return [str(exc)]
        return [self._describe_hit(handle)]

    def _cmd_help(self, _arg: str) -> list[str | Text]:
        return [HELP_TEXT]
