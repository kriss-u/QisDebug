"""The add/remove watch/breakpoint input popup for the TUI's
"w"/"b"/"W"/"B" actions.

Split out of :mod:`._tui` purely to keep that module under this project's
line-count threshold -- :class:`CommandInputScreen` is still TUI
presentation, just defined here rather than in it.

This is deliberately a small, centered popup around one plain text field --
the same typed syntax the ``:`` command line already accepts (``watch qubit
0``, ``break seq >= 3``, ``unwatch all``, ``unbreak 1``) -- rather than a
form of dropdowns/spin-boxes: a terminal-native experience should feel like
typing at a prompt, with completion to speed that up, not like filling in a
GUI dialog. Live tab-completion is driven by whichever of
:meth:`~._cli_session.CliSession.watch_completions`/``unwatch_completions``/
:meth:`~._cli_session.CliSession.break_completions`/``unbreak_completions``
matches the action -- the exact same per-token suggestion contract already
covered for the typed command line -- so the popup and typing the whole
command out by hand can never disagree about what's valid to type next.
Submitting hands the finished line straight to
:meth:`~._cli_session.CliSession.execute`, so each shortcut plus this popup
is just a shortcut for typing ``: watch ...``/``: unbreak ...``/etc., never
a second parser.
"""

from __future__ import annotations

import os
from typing import Callable

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label, Static


class CommandInputScreen(ModalScreen[str | None]):
    """A small centered popup wrapping one text field with live,
    terminal-style tab-completion -- typing "qubit" then Tab then "0" then
    Enter, the same motion a real shell completion prompt gives, rather
    than picking from a dropdown.

    ``completions(tokens)`` is called on every keystroke with the tokens
    already fully typed (the word currently being typed is never included)
    and returns every valid next token; matches narrow as more of the
    current word is typed. Dismisses with the finished line on Enter, or
    ``None`` on Escape.
    """

    DEFAULT_CSS = """
    CommandInputScreen {
        align: center middle;
    }
    #cmd-box {
        width: 64;
        height: auto;
        border: round $accent;
        background: $panel;
        padding: 1 2;
    }
    #cmd-title {
        text-style: bold;
    }
    #cmd-hint {
        color: $text-muted;
        padding-bottom: 1;
    }
    #cmd-suggestions {
        color: $text-muted;
        padding-top: 1;
    }
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(
        self,
        title: str,
        hint: str,
        completions: Callable[[list[str]], list[str]],
        initial: str = "",
    ) -> None:
        super().__init__()
        self._title = title
        self._hint = hint
        self._completions = completions
        self._initial = initial

    def compose(self) -> ComposeResult:
        with Vertical(id="cmd-box"):
            yield Label(self._title, id="cmd-title")
            yield Label(self._hint, id="cmd-hint")
            yield Input(self._initial, id="cmd-input")
            yield Static(id="cmd-suggestions", markup=False)

    def on_mount(self) -> None:
        """Focus the input, cursor at the end of any pre-filled text, and
        show the completions for what's there already."""
        input_widget = self.query_one("#cmd-input", Input)
        input_widget.focus()
        input_widget.cursor_position = len(input_widget.value)
        self._refresh_suggestions(input_widget.value)

    def on_input_changed(self, event: Input.Changed) -> None:
        """Recompute the suggestion strip on every keystroke."""
        self._refresh_suggestions(event.value)

    @staticmethod
    def _split(text: str) -> tuple[list[str], str]:
        """(fully-typed tokens, the partial token currently being typed --
        empty when the line is blank or ends in whitespace, i.e. the next
        token hasn't been started yet)."""
        if text == "" or text.endswith(" "):
            return text.split(), ""
        tokens = text.split()
        return tokens[:-1], tokens[-1]

    def _matches(self, text: str) -> list[str]:
        committed, partial = self._split(text)
        candidates = self._completions(committed)
        return [c for c in candidates if c.startswith(partial)] if partial else candidates

    def _refresh_suggestions(self, text: str) -> None:
        matches = self._matches(text)
        suggestions = self.query_one("#cmd-suggestions", Static)
        suggestions.update("  ".join(matches) if matches else "(no more completions)")

    async def on_key(self, event: events.Key) -> None:
        """Tab completes the token being typed to the longest common prefix
        of its matches -- a single match also gets a trailing space, ready
        to type the next token -- the same completion a real shell gives.
        Runs ahead of this screen's own "tab" binding (focus-next), which
        would otherwise just move focus off the only field in this popup.
        """
        if event.key != "tab":
            return
        event.stop()
        event.prevent_default()
        input_widget = self.query_one("#cmd-input", Input)
        text = input_widget.value
        committed, _partial = self._split(text)
        matches = self._matches(text)
        if not matches:
            return
        common = os.path.commonprefix(matches)
        completed = f"{common} " if len(matches) == 1 else common
        prefix = f"{' '.join(committed)} " if committed else ""
        new_value = prefix + completed
        input_widget.value = new_value
        input_widget.cursor_position = len(new_value)
        self._refresh_suggestions(new_value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Enter submits the typed line, unless it's empty."""
        text = event.value.strip()
        if not text:
            self.notify("nothing typed", severity="error", timeout=4)
            return
        self.dismiss(text)

    def action_cancel(self) -> None:
        """Close without running anything."""
        self.dismiss(None)
