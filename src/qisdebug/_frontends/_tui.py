"""Full-screen terminal UI: the primary CLI experience on a real terminal.

Built on `textual` -- persistent panels for the timeline, the circuit
drawing, the event's other info (description/state/histogram), the active
watches, and the active breakpoints, all updating live as the playhead
moves. This is deliberately the GUI's experience (``gui.py``'s
always-visible panels, not a scrolling transcript) rendered for an SSH/plain
terminal session: every panel is reachable and operable by keyboard alone
(``Tab``/arrow keys/``Enter``), since a remote terminal may not even deliver
mouse events.

The circuit drawing and the event's other info (statevector, histogram,
property changes, ...) are two separate panels -- circuit above, info below
-- rather than one blob of text, so both stay visible together instead of
one running into the other. The diff between marks A and B is its own
full-screen :class:`DiffScreen`, not a section squeezed alongside those two
panels -- there's no room to see a circuit diagram and a diff at once
otherwise, and a diff can be long (watch diff, split circuit diagram, gate
diff).

There is no permanently docked command box -- the frequent actions
(navigation, mark/unmark A/B, continue-to-breakpoint, collapse/expand, zoom,
help, quit) are single-key (or two-key, for mark/unmark) :attr:`TimelineApp.BINDINGS`,
and everything else (the full free-text command language this frontend has
always accepted, e.g. ``watch qubit 0`` or ``break tvd 00=0.5 01=0.5 >=
0.1``) goes through a transient, vim-style ``:`` command line
(:class:`CommandScreen`) that only exists for the one line being typed, the
same way ``:`` opens Vim's command line rather than a panel permanently
eating a row of the screen.

:class:`TimelineApp` only *presents* -- it drives one
:class:`~._cli_session.CliSession` per run and never duplicates its command
parsing or diff/watch computation; :mod:`.cli`'s non-interactive fallback
loop drives the same session class when no real terminal is attached.
"""

from __future__ import annotations

import re
import sys

from rich.text import Text
from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import Footer, Input, Label, ListItem, ListView, Static

from . import _guided
from . import _render
from ._cli_session import HELP_TEXT, CliSession, CommandError

# Kept as a constant, not read back off the widget at mount time, since the
# widget's laid-out size isn't reliably available that early -- and this is
# a fixed CSS width (see "#timeline-list" below, which must match) rather
# than something that changes at runtime anyway.
_TIMELINE_LIST_WIDTH = 32

_CHORD_HINTS = {
    "m": "mark: press a or b (Esc to cancel)",
    "u": "unmark: press a, b, or u for both (Esc to cancel)",
}


_BAR_SPAN = re.compile(r"[▓░]+")


def _underline_row_dividers(text: str) -> Text:
    """Underline just the shaded bar in every row except the last, as a
    divider between histogram outcomes that costs no extra vertical space
    -- a border under the bar itself, not under the whole row (the label
    and count/percentage stay plain), and not a blank line, which would
    make an already-tight column taller for no reason.
    """
    lines = text.split("\n")
    result = Text()
    for i, line in enumerate(lines):
        match = _BAR_SPAN.search(line) if i < len(lines) - 1 else None
        if match is None:
            result.append(line)
        else:
            result.append(line[: match.start()])
            result.append(line[match.start() : match.end()], style="underline")
            result.append(line[match.end() :])
        if i != len(lines) - 1:
            result.append("\n")
    return result


class HelpScreen(ModalScreen[None]):
    """A dismissable overlay listing the command language and key bindings --
    the keyboard-discoverable equivalent of the always-visible ``Footer``,
    for the fuller command reference that doesn't fit in a footer line.
    """

    DEFAULT_CSS = """
    HelpScreen {
        align: center middle;
    }
    #help-text {
        width: auto;
        height: auto;
        border: round $accent;
        padding: 1 2;
        background: $panel;
    }
    """

    # No "question_mark" binding here: the App's own "?" binding is
    # priority=True (see TimelineApp.BINDINGS), which always intercepts the
    # key before it would reach this screen's bindings -- action_show_help
    # already toggles the screen closed when it's the one on top, so a
    # second binding for the same key here would just be unreachable.
    BINDINGS = [Binding("escape", "dismiss_help", "Close")]

    def compose(self) -> ComposeResult:
        yield Static(HELP_TEXT, id="help-text", markup=False)

    def action_dismiss_help(self) -> None:
        """Close the help overlay."""
        self.dismiss()


class CommandScreen(ModalScreen[str | None]):
    """A one-line, vim-``:``-style command entry that exists only while it's
    being used: opened by the ``:`` binding, dismissed with the typed line on
    Enter or with ``None`` on Escape -- never a permanent fixture of the
    layout the way a docked command box would be.
    """

    DEFAULT_CSS = """
    CommandScreen {
        align: center bottom;
        background: transparent;
    }
    #command-bar {
        dock: bottom;
        height: 1;
        background: $panel;
    }
    #command-prompt {
        width: 1;
        color: $accent;
    }
    #command-input {
        border: none;
        background: transparent;
        padding: 0;
    }
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def compose(self) -> ComposeResult:
        with Horizontal(id="command-bar"):
            yield Label(":", id="command-prompt")
            yield Input(
                placeholder="command (Enter to run, Esc to cancel)", id="command-input"
            )

    def on_mount(self) -> None:
        """Focus the input the instant this transient screen appears."""
        self.query_one(Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Dismiss with the typed line, for :meth:`TimelineApp.action_open_command`."""
        self.dismiss(event.value)

    def action_cancel(self) -> None:
        """Close without running anything."""
        self.dismiss(None)


class ElementPickerScreen(Screen[str | None]):
    """Full-screen picker for which ``"<circuit>:<track>"`` element a mark
    should diff, shown only when a stage actually has more than one
    candidate -- a single-element stage is used automatically, no prompt
    needed (see :meth:`TimelineApp._mark_and_pick_element`). Mirrors
    :meth:`~._cli_session.CliSession.elements`, the same enumeration the
    typed ``elements a``/``elements b`` commands already listed; this just
    makes picking one a UI gesture instead of a second typed command.

    A full screen (matching :class:`DiffScreen`), not a small centered
    modal -- an earlier version tried floating a compact box over the main
    screen and live-updating *that* screen's own ``#circuit`` widget behind
    it, which needed careful size-capping to avoid the floating box
    growing to cover the very panel it was meant to sit over (Textual's
    "auto"-sized containers around unconstrained children can do that).
    This sidesteps the whole problem: the preview lives in its own pane,
    ``#picker-preview``, right here on this screen, updated as the
    highlighted row changes -- nothing reaches back into the screen
    underneath.
    """

    DEFAULT_CSS = """
    ElementPickerScreen {
        layout: vertical;
    }
    #picker-title {
        height: 1;
        background: $boost;
        padding: 0 1;
    }
    #picker-body {
        height: 1fr;
    }
    #picker-list {
        width: 40;
        border-right: solid $accent;
    }
    #picker-preview {
        width: 1fr;
        padding: 0 1;
        overflow-x: auto;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("q", "cancel", "Cancel", show=False),
    ]

    def __init__(self, title: str, elements: list[str], tracks: dict[str, str]) -> None:
        super().__init__()
        self._title = title
        self._elements = elements
        self._tracks = tracks

    def compose(self) -> ComposeResult:
        yield Static(self._title, id="picker-title")
        with Horizontal(id="picker-body"):
            yield ListView(
                *(
                    ListItem(Label(_render.track_label(element), markup=False))
                    for element in self._elements
                ),
                id="picker-list",
            )
            yield Static(id="picker-preview", markup=False)
        yield Footer()

    def on_mount(self) -> None:
        """Focus the option list the instant this screen appears, and
        preview whichever element starts out highlighted (index 0)."""
        list_view = self.query_one(ListView)
        list_view.focus()
        self._preview(0)

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        """Live-preview whichever element is currently highlighted in this
        screen's own preview pane, not just the one eventually chosen --
        so arrowing through candidates shows what each one actually looks
        like before committing to it.
        """
        if event.list_view.index is not None:
            self._preview(event.list_view.index)

    def _preview(self, index: int) -> None:
        element = self._elements[index]
        self.query_one("#picker-preview", Static).update(Text(self._tracks.get(element, "")))

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Dismiss with the chosen element. Stopped here rather than left to
        bubble: an unstopped ``ListView.Selected`` would otherwise keep
        going up to :meth:`TimelineApp.on_list_view_selected`, which would
        wrongly treat this picker's selected row as a timeline jump.
        """
        event.stop()
        self.dismiss(self._elements[event.index])

    def action_cancel(self) -> None:
        """Close without picking anything."""
        self.dismiss(None)


class DiffScreen(Screen[None]):
    """Full-screen view of the A/B diff: watch diff, split circuit diagram,
    and gate/stage diff all get the whole terminal instead of a section
    squeezed alongside the circuit/info panels, which left no room to see a
    circuit diagram and a diff at the same time.
    """

    # Screen.app is typed generically as App[object] since a Screen doesn't
    # know which App subclass hosts it; this only ever runs inside
    # TimelineApp, so declare that concretely for `self.app` in action_close.
    app: TimelineApp

    DEFAULT_CSS = """
    DiffScreen {
        layout: vertical;
    }
    #diff-status {
        height: 1;
        background: $boost;
        padding: 0 1;
    }
    #diff-body {
        height: 1fr;
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("escape", "close", "Back"),
        Binding("q", "close", "Back", show=False),
    ]

    def __init__(self, status: str, content: Text | str) -> None:
        super().__init__()
        self._status = status
        self._content = content if isinstance(content, Text) else Text(content)

    def compose(self) -> ComposeResult:
        yield Static(self._status, id="diff-status", markup=False)
        with VerticalScroll(id="diff-body"):
            yield Static(self._content, id="diff-content", markup=False)
        yield Footer()

    def action_close(self) -> None:
        """Return to the main timeline screen."""
        self.app.pop_screen()


class TimelineApp(App[None]):  # pylint: disable=too-many-public-methods
    """The persistent-panel timeline viewer.

    Every panel is rebuilt from :attr:`session` after any action that could
    have changed what it shows -- there is no incremental patching, since a
    single merged timeline's panels are cheap to re-render in full and
    keeping them trivially always-correct is worth more here than the
    (unmeasurable) cost of redrawing a few lines of text on every keypress.
    Public-method count is inflated by Textual's own dispatch convention --
    every ``BINDINGS`` entry and ``on_<message>`` handler is its own public
    method by design, not a sign this class has too many responsibilities
    (see ``gui.py``'s ``_TimelineWindow``, which carries the same disable
    for the same reason).
    """

    TITLE = "qisdebug"

    CSS = """
    Screen {
        layout: vertical;
    }
    * {
        /* scrollbar-size-* isn't inherited in Textual, so a rule on
           Screen alone (tried first) never reached the scrollable panes
           below it -- a universal selector is what actually gets there. */
        scrollbar-size-vertical: 1;
        scrollbar-size-horizontal: 0;
    }
    #status {
        height: 1;
        background: $boost;
        padding: 0 1;
    }
    #chord-hint {
        height: 1;
        background: $accent 30%;
        padding: 0 1;
        display: none;
    }
    #diff-indicator {
        height: 1;
        background: $warning 30%;
        padding: 0 1;
        display: none;
    }
    #body {
        height: 1fr;
    }
    .pane-title {
        text-style: bold;
        color: $accent;
        background: $boost;
        padding: 0 1;
    }
    .focus-pane:focus {
        /* $primary (a cool blue in both the light and dark themes) rather
           than $accent, which is the same yellow/orange as $warning --
           too jarring for a plain "this pane has focus" indicator. */
        background: $primary 25%;
    }
    #timeline-list {
        width: 32;
        /* must match _TIMELINE_LIST_WIDTH */
        border-right: solid $accent;
    }
    #right {
        width: 1fr;
        height: 1fr;
    }
    #detail-pane {
        height: 1fr;
        border-bottom: solid $accent;
    }
    #circuit-pane {
        height: 3fr;
        padding: 0 1;
        border-bottom: solid $accent;
        /* Default zoom draws the circuit at its natural (unlimited-fold)
           width -- scroll to the rest instead of wrapping it up front. */
        overflow-x: auto;
    }
    #circuit {
        width: auto;
    }
    #circuit-note {
        color: $text-muted;
    }
    #circuit-note.hidden {
        display: none;
    }
    #info-pane {
        height: 2fr;
        padding: 0 1;
    }
    #info-text {
        padding: 0;
    }
    #info-histogram {
        border-top: solid $accent;
        margin-top: 1;
        padding: 0;
    }
    #info-histogram.hidden {
        display: none;
        border-top: none;
        margin-top: 0;
    }
    #side {
        height: auto;
        max-height: 45%;
    }
    #watches-pane, #breakpoints-pane {
        width: 1fr;
        height: auto;
        max-height: 100%;
        border-right: solid $accent;
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("n", "step_next", "Next"),
        Binding("right", "step_next", "Next", show=False),
        Binding("space", "step_next", "Next", show=False),
        Binding("p", "step_prev", "Prev"),
        Binding("left", "step_prev", "Prev", show=False),
        # "[" / "]" rather than Home/End: plenty of laptop keyboards have no
        # dedicated Home/End keys at all (Fn-chord only, or missing).
        Binding("[", "step_first", "First"),
        Binding("]", "step_last", "Last"),
        # Real bindings, not just documentation: on_key claims "m"/"u"
        # itself before either of these actions would ever actually fire
        # (see on_key's docstring), but a Binding still has to exist for
        # the Footer to show the hint -- Footer only lists BINDINGS, it has
        # no idea a two-key chord is handled elsewhere. If on_key's
        # interception is ever bypassed, these do the same thing pressing
        # the key does (arm the chord), so they're not true no-ops either.
        Binding("m", "arm_mark", "Mark (a/b)"),
        Binding("u", "arm_unmark", "Unmark (a/b/u)"),
        Binding("w", "add_watch_guided", "Watch"),
        Binding("b", "add_breakpoint_guided", "Break"),
        Binding("W", "remove_watch_guided", "Unwatch"),
        Binding("B", "remove_breakpoint_guided", "Unbreak"),
        Binding("c", "continue_hit", "Continue"),
        Binding("x", "toggle_circuits", "All circuits"),
        Binding("D", "show_diff", "Diff"),
        Binding("plus", "zoom_in", "Zoom in"),
        Binding("minus", "zoom_out", "Zoom out"),
        Binding("0", "zoom_reset", "Reset zoom", show=False),
        Binding("colon", "open_command", "Command"),
        Binding("question_mark", "show_help", "Help", priority=True),
        Binding("q", "quit_app", "Quit"),
        Binding("ctrl+c", "quit_app", "Quit", show=False),
    ]

    # Every action above except "show_help" and "quit_app" belongs to the
    # main timeline view -- see check_action, which stops them from also
    # firing (and, since check_action's False also hides a binding from the
    # Footer, from being *advertised*) when a help/diff/element-picker/
    # command screen is on top. "arm_mark"/"arm_unmark" belong here too --
    # without them, the Footer kept showing "Mark (a/b)"/"Unmark (a/b)" as
    # available even while a sub-screen was on top, though on_key's own
    # `_on_main_screen` guard already refused to arm the chord there.
    _MAIN_SCREEN_ACTIONS = frozenset(
        {
            "step_next", "step_prev", "step_first", "step_last",
            "continue_hit", "toggle_circuits", "show_diff",
            "zoom_in", "zoom_out", "zoom_reset", "open_command",
            "add_watch_guided", "add_breakpoint_guided",
            "remove_watch_guided", "remove_breakpoint_guided",
            "arm_mark", "arm_unmark",
        }
    )  # fmt: skip

    def __init__(self, session: CliSession, subtitle: str) -> None:
        super().__init__()
        self.session = session
        self.sub_title = subtitle
        self._pending_leader: str | None = None

    def compose(self) -> ComposeResult:
        yield Static(id="status", markup=False)
        yield Static(id="chord-hint", markup=False)
        yield Static(id="diff-indicator", markup=False)
        with Horizontal(id="body"):
            yield ListView(id="timeline-list", classes="focus-pane")
            with Vertical(id="right"):
                # Stacked, not side by side: the circuit drawing is usually
                # wide, and a right-hand info column would either cramp it
                # or itself be too narrow to read -- both stay fully wide
                # and simply share the height instead.
                with Vertical(id="detail-pane"):
                    with VerticalScroll(id="circuit-pane", classes="focus-pane"):
                        yield Static("Circuit", classes="pane-title")
                        # Its own line, outside the (zoomable/scrollable)
                        # drawing itself -- session metadata, not part of
                        # the diagram, so it shouldn't scroll or reflow
                        # along with it.
                        yield Static(id="circuit-note", markup=False)
                        yield Static(id="circuit", markup=False)
                    with VerticalScroll(id="info-pane", classes="focus-pane"):
                        yield Static("Info", classes="pane-title")
                        yield Static(id="info-text", markup=False)
                        # Below the text, not beside it -- and hidden
                        # entirely (no rule, no space) when there's nothing
                        # to show, see refresh_panels.
                        yield Static(id="info-histogram", markup=False)
                with Horizontal(id="side"):
                    with VerticalScroll(id="watches-pane", classes="focus-pane"):
                        yield Static("Watches", classes="pane-title")
                        yield Static(id="watches", markup=False)
                    with VerticalScroll(id="breakpoints-pane", classes="focus-pane"):
                        yield Static("Breakpoints", classes="pane-title")
                        yield Static(id="breakpoints", markup=False)
        yield Footer()

    def on_mount(self) -> None:
        """Populate the timeline panel, draw the first frame, and focus it."""
        # Qiskit's text circuit drawer reads ``sys.stdout.encoding`` directly
        # (falling back to "utf8" only if that attribute exists and is
        # falsy); while this app is running, Textual has replaced
        # ``sys.stdout`` with a capture object that has no such attribute at
        # all, which raises rather than falling back. Give it one so drawing
        # a circuit (in the circuit or diff panel) doesn't crash mid-run.
        if not hasattr(sys.stdout, "encoding"):
            setattr(sys.stdout, "encoding", "utf-8")
        list_view = self.query_one("#timeline-list", ListView)
        for row in self.session.timeline_rows(max_width=_TIMELINE_LIST_WIDTH - 3):
            list_view.append(ListItem(Label(row, markup=False)))
        self.refresh_panels()
        # Focus the timeline panel so the single-key navigation bindings
        # (n/p/[/]/...) work the instant the app opens -- typing a command
        # is the deliberate action of pressing ":" to open the transient
        # command line, not the default state.
        list_view.focus()

    def refresh_panels(self) -> None:
        """Redraw every panel from the session's current state."""
        # The info pane's own current width, so a counts histogram scales to
        # actually fit it instead of assuming a fixed width always does; the
        # pane is already mounted and laid out by the time this ever runs.
        info_width = max(16, self.query_one("#info-pane").size.width - 2)
        frame = self.session.current_frame(info_width=info_width)
        self.query_one("#status", Static).update(
            Text(
                f"[{frame.index + 1}/{frame.total}] {frame.circuit_name} — "
                f"{frame.event_type} at {frame.timestamp:.3f} — {frame.description}"
            )
        )

        self.query_one("#circuit", Static).update(Text(frame.tracks_text))

        note_widget = self.query_one("#circuit-note", Static)
        if frame.other_circuits:
            names_not_shown = ", ".join(frame.other_circuits)
            note_widget.update(
                f"+{len(frame.other_circuits)} more not shown: {names_not_shown}"
            )
            note_widget.remove_class("hidden")
        else:
            note_widget.update("")
            note_widget.add_class("hidden")

        text_lines = [frame.event_text]
        if frame.bloch_text:
            text_lines += ["", frame.bloch_text]
        self.query_one("#info-text", Static).update(Text("\n".join(text_lines)))

        histogram_widget = self.query_one("#info-histogram", Static)
        if frame.histogram_text:
            histogram_widget.remove_class("hidden")
            content = Text("Counts histogram:\n")
            content.append_text(_underline_row_dividers(frame.histogram_text))
            histogram_widget.update(content)
        else:
            # No border, no column -- an empty bordered pane with nothing
            # in it just reads as a stray line, not a real divider.
            histogram_widget.add_class("hidden")
            histogram_widget.update("")

        watch_labels = self.session.watch_labels()
        watch_text = self.session.render_watches() if watch_labels else ""
        self.query_one("#watches", Static).update(
            Text(watch_text) if watch_text else Text("(no active watches)")
        )

        rows = self.session.breakpoint_rows()
        breakpoints_text = "\n".join(f"[{handle}] {desc}" for handle, desc in rows)
        self.query_one("#breakpoints", Static).update(
            Text(breakpoints_text) if rows else Text("(no active breakpoints)")
        )

        indicator = self.query_one("#diff-indicator", Static)
        status = self.session.diff_status_line()
        indicator.display = status is not None
        if status is not None:
            indicator.update(Text(status + "   (D, or : diff, to view)"))

        list_view = self.query_one("#timeline-list", ListView)
        if list_view.index != frame.index:
            list_view.index = frame.index

    def _toast(self, outcome_messages: list[str | Text]) -> None:
        for message in outcome_messages:
            plain = message.plain if isinstance(message, Text) else message
            if plain.strip():
                self.notify(plain, markup=False, timeout=5)

    def _run_command(self, line: str) -> None:
        """Execute one non-diff command line through the session, surface any
        resulting messages as toast notifications, and redraw every panel.
        A ``diff``/``d`` line is routed through :meth:`_run_diff_command`
        instead -- its content belongs on :class:`DiffScreen`, not in a toast.
        """
        outcome = self.session.execute(line)
        self._toast(outcome.messages)
        self.refresh_panels()
        if self.session.done:
            self.exit()

    def _run_diff_command(self, line: str) -> None:
        """Execute a typed ``diff``/``d`` line, opening :class:`DiffScreen`
        when it produces content to show (mirrors :meth:`action_show_diff`,
        the keyboard equivalent) rather than toasting what can be a long
        diff. ``diff hide`` and usage/error messages still just toast.
        """
        outcome = self.session.execute(line)
        self.refresh_panels()
        if self.session.diff_ready() and self.session.diff_expanded:
            status = self.session.diff_status_line() or ""
            content = self.session.render_diff() or ""
            self.push_screen(DiffScreen(status, content))
        else:
            self._toast(outcome.messages)
        if self.session.done:
            self.exit()

    @work
    async def action_open_command(self) -> None:
        """Open the transient ``:`` command line and run whatever line comes
        back (``None`` on cancel, which runs nothing). ``push_screen_wait``
        requires a worker context, hence ``@work`` rather than a plain
        ``async def`` action.
        """
        line = await self.push_screen_wait(CommandScreen())
        if line is None:
            return
        name = line.strip().split(maxsplit=1)[0] if line.strip() else ""
        if name in ("diff", "d"):
            self._run_diff_command(line)
        else:
            self._run_command(line)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Jump the playhead to the timeline row the user picked. Guarded to
        the main screen: the element picker has its own ``ListView`` (see
        :class:`ElementPickerScreen`, which stops its own selection event),
        but this still runs defensively in case anything else ever pushes a
        screen containing a ``ListView`` of its own.
        """
        if not self._on_main_screen:
            return
        self.session.go_to(event.index + 1)
        self.refresh_panels()

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        """Follow the timeline highlight live as arrow keys move it, so the
        circuit/info panels update while browsing -- no Enter needed.
        Guarded on the index actually changing: :meth:`refresh_panels`
        itself moves the highlight back onto the playhead after every other
        action (``n``, a typed command, ...), and without this check that
        assignment would immediately re-trigger this handler. Also guarded
        to the main screen, for the same reason as :meth:`on_list_view_selected`.
        """
        if not self._on_main_screen:
            return
        index = event.list_view.index
        if index is None or index == self.session.index:
            return
        self.session.go_to(index + 1)
        self.refresh_panels()

    @property
    def _on_main_screen(self) -> bool:
        """Whether the main timeline view -- not help, diff, the element
        picker, or the ``:`` command line -- is the screen on top."""
        return len(self.screen_stack) == 1

    def check_action(
        self,
        action: str,
        parameters: tuple[object, ...],  # pylint: disable=unused-argument
    ) -> bool | None:
        """Disable every main-screen action (:attr:`_MAIN_SCREEN_ACTIONS`)
        while a help/diff/element-picker/command screen is on top, so a key
        like "n" or "x" only ever drives whichever screen is actually being
        looked at -- not silently also the timeline view underneath it.
        "?" and quit stay available everywhere, and are not in that set.
        """
        if action in self._MAIN_SCREEN_ACTIONS and not self._on_main_screen:
            return False
        return True

    def action_step_next(self) -> None:
        """Advance to the next event."""
        self.session.go_next()
        self.refresh_panels()

    def action_step_prev(self) -> None:
        """Go back to the previous event."""
        self.session.go_prev()
        self.refresh_panels()

    def action_step_first(self) -> None:
        """Jump to the first event."""
        self.session.go_first()
        self.refresh_panels()

    def action_step_last(self) -> None:
        """Jump to the last event."""
        self.session.go_last()
        self.refresh_panels()

    # -- mark/unmark A/B: two-key "m"/"u" then "a"/"b" chords, see on_key --

    async def on_key(self, event: events.Key) -> None:
        """Complete or cancel a pending ``m``/``u`` -> ``a``/``b`` chord
        (mark or unmark that diff slot) before any other key handling.
        Two-key gestures rather than single ``a``/``b`` keys, so touching
        A/B diff state -- unlike plain navigation -- takes a deliberate
        small gesture. ``on_key`` runs ahead of the normal ``BINDINGS``
        dispatch for the same key (confirmed against this Textual version),
        so consuming the leader and its follow-up here is sufficient; no
        ``Binding`` for "m"/"u" is declared, since one would never fire once
        ``event.stop()``/``prevent_default()`` below claims the key first.
        The ``:`` command line's ``Input`` captures its own keys before
        they'd ever reach here, but nothing else pushed on the screen stack
        does (a ``ListView``/``Static`` doesn't eat plain "m"/"u"), so this
        is explicitly gated to the main screen -- a chord must not fire
        while help, diff, or the element picker is what's on screen.
        """
        if not self._on_main_screen:
            self._cancel_chord()
            return
        if self._pending_leader is not None:
            leader = self._pending_leader
            self._cancel_chord()
            if event.key in ("a", "b"):
                event.stop()
                event.prevent_default()
                if leader == "m":
                    self._start_mark(event.key)
                else:
                    self._start_unmark(event.key)
            elif leader == "u" and event.key == "u":
                # "uu", mirroring vim's doubled-key-for-"both lines" mnemonic
                # (dd/yy): unmark both slots at once instead of requiring two
                # separate "u"+"a" / "u"+"b" chords.
                event.stop()
                event.prevent_default()
                self._start_unmark("all")
            return
        if event.key in _CHORD_HINTS:
            event.stop()
            event.prevent_default()
            self._arm_chord(event.key)

    def _arm_chord(self, leader: str) -> None:
        """Arm a pending leader chord and show a hint for what to press
        next -- otherwise pressing "m"/"u" alone gives no on-screen sign
        anything happened. Stays armed indefinitely -- no timeout -- until
        "a"/"b" resolves it or any other key (Esc included) cancels it.
        """
        self._pending_leader = leader
        hint = self.query_one("#chord-hint", Static)
        hint.update(_CHORD_HINTS[leader])
        hint.display = True

    def _cancel_chord(self) -> None:
        """Clear any pending leader chord and its hint, on resolution (a
        follow-up key was pressed) or explicit cancellation (any other key,
        including Esc)."""
        self._pending_leader = None
        self.query_one("#chord-hint", Static).display = False

    def action_arm_mark(self) -> None:
        """Footer-visible entry point for the "m" mark chord (see the
        BINDINGS comment above) -- arms it the same way pressing "m" does."""
        self._arm_chord("m")

    def action_arm_unmark(self) -> None:
        """Footer-visible entry point for the "u" unmark chord."""
        self._arm_chord("u")

    def _start_mark(self, slot: str) -> None:
        self._mark_and_pick_element(slot)

    def _start_unmark(self, slot: str) -> None:
        """Clear diff slot ``slot`` ("a"/"b"/"all"), mirroring :meth:`_start_mark`."""
        label = {"a": "A", "b": "B", "all": "A and B"}[slot]
        try:
            self.session.unmark(slot)
        except CommandError as exc:
            self.notify(str(exc), severity="error", timeout=4)
            return
        self.notify(f"unmarked {label}", timeout=3)
        self.refresh_panels()

    @work
    async def _mark_and_pick_element(self, slot: str) -> None:
        """Pin the current event as diff slot ``slot``, then -- only if this
        stage actually has more than one candidate element (multiple
        circuits, or one circuit transpiled more than once) -- ask which one
        through :class:`ElementPickerScreen` instead of silently diffing
        whichever element happened to sort first.
        """
        label = "A" if slot == "a" else "B"
        try:
            stage_mark = self.session.mark(slot)
        except CommandError as exc:
            self.notify(str(exc), severity="error", timeout=4)
            return
        self.refresh_panels()
        elements = stage_mark.elements_at(self.session.timelines_by_id)
        if len(elements) > 1:
            chosen = await self.push_screen_wait(
                ElementPickerScreen(
                    f"Diff element {label}", elements, self.session.tracks[self.session.index]
                )
            )
            if chosen is not None:
                self.session.set_diff_element(slot, chosen)
        self.notify(
            f"marked {label}: {stage_mark.label(self.session.timelines_by_id)}",
            timeout=3,
        )
        self.refresh_panels()

    @work
    async def action_add_watch_guided(self) -> None:
        """Add a watch through the guided step wizard (see :mod:`._guided`)
        -- an arrow-navigable list of valid next tokens plus a dynamic
        hint, building the exact same ``watch kind ref`` line the ``:``
        command line accepts, without opening it and typing "watch" first."""
        text = await self.push_screen_wait(
            _guided.GuidedWizardScreen(
                "Add watch",
                self.session.watch_step_hint,
                self.session.watch_completions,
                self.session.watch_is_complete,
            )
        )
        if text is not None:
            self._run_command(f"watch {text}")

    @work
    async def action_add_breakpoint_guided(self) -> None:
        """Add a breakpoint through the same wizard (see :mod:`._guided`) --
        builds the same typed syntax (``seq >= 3``, ``watch qubit 0 >= 0.5``)
        one token at a time, from the exact same suggestions the ``:``
        command line offers."""
        text = await self.push_screen_wait(
            _guided.GuidedWizardScreen(
                "Add breakpoint",
                self.session.break_step_hint,
                self.session.break_completions,
                self.session.break_is_complete,
            )
        )
        if text is not None:
            self._run_command(f"break {text}")

    @work
    async def action_remove_watch_guided(self) -> None:
        """Remove a watch through the same wizard -- ``kind ref`` (only
        ever suggesting an *actually active* watch's kind/ref, see
        :meth:`~._cli_session.CliSession.unwatch_completions`) or ``all``,
        the CLI counterpart to the GUI's "Remove selected" button for the
        watch list."""
        if not self.session.watches:
            self.notify("no active watches", timeout=3)
            return
        text = await self.push_screen_wait(
            _guided.GuidedWizardScreen(
                "Remove watch",
                self.session.unwatch_step_hint,
                self.session.unwatch_completions,
                self.session.unwatch_is_complete,
            )
        )
        if text is not None:
            self._run_command(f"unwatch {text}")

    @work
    async def action_remove_breakpoint_guided(self) -> None:
        """Remove a breakpoint through the same wizard -- handle or
        ``all``, the CLI counterpart to the GUI's "Remove selected" button
        for the breakpoint list."""
        if not self.session.breakpoints:
            self.notify("no active breakpoints", timeout=3)
            return
        text = await self.push_screen_wait(
            _guided.GuidedWizardScreen(
                "Remove breakpoint",
                self.session.unbreak_step_hint,
                self.session.unbreak_completions,
                self.session.unbreak_is_complete,
            )
        )
        if text is not None:
            self._run_command(f"unbreak {text}")

    def action_continue_hit(self) -> None:
        """Jump to the next event matching any active breakpoint."""
        self._run_command("continue")

    def action_toggle_circuits(self) -> None:
        """Switch the circuit panel between "just the active circuit" and
        every tracked circuit's every track."""
        self.session.toggle_show_all_circuits()
        self.refresh_panels()

    def action_show_diff(self) -> None:
        """Open the full A/B diff as its own screen -- the watch diff, split
        circuit diagram, and gate/stage diff get the whole terminal instead
        of competing with the timeline/circuit/info panels for space.
        """
        if not self.session.diff_ready():
            self.notify("mark both a and b to compare (see: m a, m b)", timeout=4)
            return
        self.session.diff_expanded = True
        status = self.session.diff_status_line() or ""
        content = self.session.render_diff() or ""
        self.push_screen(DiffScreen(status, content))

    def action_zoom_in(self) -> None:
        """Widen the circuit drawing."""
        width = self.session.zoom_in()
        self.refresh_panels()
        self.notify(f"circuit zoom: fold={width}", timeout=2)

    def action_zoom_out(self) -> None:
        """Narrow the circuit drawing (more compact, more wrapping)."""
        width = self.session.zoom_out()
        self.refresh_panels()
        self.notify(f"circuit zoom: fold={width}", timeout=2)

    def action_zoom_reset(self) -> None:
        """Return the circuit drawing to its default width."""
        width = self.session.zoom_reset()
        self.refresh_panels()
        self.notify(f"circuit zoom reset: fold={width}", timeout=2)

    def action_show_help(self) -> None:
        """Open the command/key-binding reference overlay, or close it if
        it's already the screen on top -- "?" both opens and closes help."""
        if isinstance(self.screen, HelpScreen):
            self.pop_screen()
        else:
            self.push_screen(HelpScreen())

    def action_quit_app(self) -> None:
        """Exit the app."""
        self.exit()
