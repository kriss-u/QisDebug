"""The add/remove watch/breakpoint input popup for the TUI's
"w"/"b"/"W"/"B" actions.

Split out of :mod:`._tui` purely to keep that module under this project's
line-count threshold -- :class:`GuidedWizardScreen` is still TUI
presentation, just defined here rather than in it.

Watch/breakpoint syntax is a small multi-token grammar (``watch qubit 0``,
``break seq >= 3``), but typing it from memory with only a passive
suggestion strip to check against turned out to be the opposite of
guided: no visible list of what's valid, and no indication of when a
command was actually complete versus still missing a token. This screen
instead builds the command one token at a time: an always-visible,
arrow-key-navigable list (:class:`~textual.widgets.OptionList`, never a
collapsed dropdown) shows every valid next token, a hint line above it
explains what that token means in plain language, and a breadcrumb shows
the command assembled so far. Typing still works exactly as before --
free text that doesn't match a listed candidate is used verbatim (for
thresholds, custom gate names, anything not enumerable), and typing
several space-separated tokens before pressing Enter commits all of them
at once, so a user who already knows the syntax can still just type it.
The one thing that changed is who decides "done": once every mandatory
token is in, the list keeps offering a pinned "finish" entry rather than
the screen guessing when to close, mirroring the freedom typing a whole
line by hand always had; the screen closes on its own only once the
grammar itself has no more suggestions to offer.

Both this screen and the underlying grammar it walks are driven by the
same per-token suggestion functions
(:meth:`~._cli_session.CliSession.watch_completions`/``unwatch_completions``/
:meth:`~._cli_session.CliSession.break_completions`/``unbreak_completions``)
plus a matching per-token hint function
(:meth:`~._cli_session.CliSession.watch_step_hint`/``unwatch_step_hint``/
``break_step_hint``/``unbreak_step_hint``), so this screen, typing the
command out by hand, and the GUI's own condition builder can never
disagree about what's valid to type next or what it means. Submitting
hands the finished line straight to
:meth:`~._cli_session.CliSession.execute`, so each shortcut plus this
screen is just a guided way to type ``: watch ...``/``: unbreak ...``/etc.,
never a second parser.
"""

from __future__ import annotations

import os
from typing import Callable

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label, OptionList, Static
from textual.widgets.option_list import Option

_FINISH_ID = "\0finish"


class GuidedWizardScreen(ModalScreen[str | None]):
    """A centered popup that builds a command one token at a time.

    ``completions(tokens)``, ``hint_for(tokens)``, and ``is_complete(tokens)``
    are all called with the tokens committed so far (never including
    whatever is still being typed) on every keystroke and after every
    token commit: ``completions`` returns every valid next token (an empty
    list means "type this one freely -- nothing to enumerate", which is
    *not* the same as done -- a threshold immediately after a comparator
    is exactly this case), ``hint_for`` returns a one-line, plain-language
    explanation of what that next token is for, and ``is_complete`` says
    whether the tokens so far already parse as a full command. Dismisses
    with the finished, space-joined line once both ``is_complete`` is true
    and ``completions`` has nothing left to suggest, when the user picks
    the pinned "finish" entry, or ``None`` on Escape with nothing committed
    yet (Escape with tokens already committed steps back one token instead
    of closing outright).
    """

    DEFAULT_CSS = """
    GuidedWizardScreen {
        align: center middle;
    }
    #cmd-box {
        width: 70;
        height: auto;
        border: round $accent;
        background: $panel;
        padding: 1 2;
    }
    #cmd-title {
        text-style: bold;
    }
    #cmd-breadcrumb {
        color: $accent;
        padding-top: 1;
        padding-bottom: 1;
    }
    #cmd-hint {
        color: $text-muted;
        padding-bottom: 1;
    }
    #cmd-options {
        height: 10;
        border: round $primary 25%;
    }
    """

    BINDINGS = [Binding("escape", "cancel", "Back/cancel", show=False)]

    def __init__(
        self,
        title: str,
        hint_for: Callable[[list[str]], str],
        completions: Callable[[list[str]], list[str]],
        is_complete: Callable[[list[str]], bool],
    ) -> None:
        super().__init__()
        self._title = title
        self._hint_for = hint_for
        self._completions = completions
        self._is_complete = is_complete
        self._tokens: list[str] = []

    def compose(self) -> ComposeResult:
        with Vertical(id="cmd-box"):
            yield Label(self._title, id="cmd-title")
            yield Static("", id="cmd-breadcrumb")
            yield Label("", id="cmd-hint")
            yield Input(id="cmd-input")
            yield OptionList(id="cmd-options")

    def on_mount(self) -> None:
        """Focus the input and show the first step's candidates right away."""
        self.query_one("#cmd-input", Input).focus()
        self._refresh()

    def on_input_changed(self, _event: Input.Changed) -> None:
        """Recompute the filtered option list on every keystroke."""
        self._refresh()

    def _refresh(self) -> None:
        breadcrumb = " ".join(self._tokens)
        self.query_one("#cmd-breadcrumb", Static).update(
            breadcrumb if breadcrumb else "(nothing chosen yet)"
        )
        self.query_one("#cmd-hint", Label).update(self._hint_for(list(self._tokens)))
        options = self.query_one("#cmd-options", OptionList)
        options.clear_options()
        for value in self._current_candidates():
            options.add_option(Option(value, id=value))
        if self._tokens:
            options.add_option(Option(f"↵ finish: {breadcrumb}", id=_FINISH_ID))
        if options.option_count:
            options.highlighted = 0

    def _current_candidates(self) -> list[str]:
        partial = self.query_one("#cmd-input", Input).value
        candidates = self._completions(list(self._tokens))
        return [c for c in candidates if c.startswith(partial)] if partial else candidates

    async def on_key(self, event: events.Key) -> None:
        """Tab completes the token being typed to the longest common prefix
        of its matches, the same shell-style completion the plain typed
        command line offers; Up/Down move the option list's highlight
        without stealing focus away from the input, since typing to filter
        and arrowing to pick are meant to compose, not be alternatives.
        """
        if event.key == "tab":
            event.stop()
            event.prevent_default()
            self._tab_complete()
        elif event.key in ("up", "down"):
            event.stop()
            event.prevent_default()
            self._move_highlight(+1 if event.key == "down" else -1)

    def _tab_complete(self) -> None:
        matches = self._current_candidates()
        if not matches:
            return
        input_widget = self.query_one("#cmd-input", Input)
        input_widget.value = os.path.commonprefix(matches)
        input_widget.cursor_position = len(input_widget.value)
        self._refresh()

    def _move_highlight(self, delta: int) -> None:
        options = self.query_one("#cmd-options", OptionList)
        if not options.option_count:
            return
        current = options.highlighted or 0
        options.highlighted = (current + delta) % options.option_count

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Clicking (or Enter-selecting) an option commits it, same as
        typing its exact text and pressing Enter would."""
        self._commit_token(str(event.option.id))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Enter commits whatever was typed (splitting on whitespace lets
        several tokens be typed at once), or falls back to the option
        list's current highlight -- including the pinned finish entry --
        when nothing was typed."""
        typed = event.value.strip()
        if typed:
            self._tokens.extend(typed.split())
            self.query_one("#cmd-input", Input).value = ""
            self._finish_if_exhausted()
            return
        options = self.query_one("#cmd-options", OptionList)
        if options.highlighted is not None:
            self._commit_token(str(options.get_option_at_index(options.highlighted).id))
            return
        if self._tokens:
            self.dismiss(" ".join(self._tokens))
            return
        self.notify("nothing typed", severity="error", timeout=4)

    def _commit_token(self, token: str) -> None:
        if token == _FINISH_ID:
            self.dismiss(" ".join(self._tokens))
            return
        self._tokens.append(token)
        self.query_one("#cmd-input", Input).value = ""
        self._finish_if_exhausted()

    def _finish_if_exhausted(self) -> None:
        """Close automatically once the command already parses AND the
        grammar has no more suggestions left (e.g. ``watch``'s fixed
        two-token shape) -- the pinned "finish" entry stays the only way to
        stop early while an optional trailing clause (e.g. breakpoints'
        ``on CIRCUIT``) is still on offer.

        Checking ``is_complete`` too, not just an empty suggestion list, is
        the fix for a real bug: suggestions also run dry right after a
        comparator token, because the threshold that must follow is
        free-text and therefore unenumerable -- not because the command is
        done. Without the completeness check the wizard would close before
        the threshold was ever typed.
        """
        tokens = list(self._tokens)
        if tokens and self._is_complete(tokens) and not self._completions(tokens):
            self.dismiss(" ".join(self._tokens))
            return
        self._refresh()

    def action_cancel(self) -> None:
        """Escape steps back one token if any are committed, otherwise
        closes without running anything."""
        if self._tokens:
            self._tokens.pop()
            self.query_one("#cmd-input", Input).value = ""
            self._refresh()
        else:
            self.dismiss(None)
