"""Shared helpers for the Textual TUI frontend's test modules
(``test_tui.py`` and ``test_tui_layout.py``)."""

from rich.text import Text
from textual.content import Content
from textual.widget import Widget
from textual.widgets import Static
from _helpers import traced_bell

from qisdebug._frontends._cli_session import CliSession
from qisdebug._frontends._tui import TimelineApp


def _plain(widget: Widget) -> str:
    """Plain text of a ``Static`` widget's current render.

    ``query_one`` only ever gives back the base ``Widget`` type, but every
    widget these tests query by id is actually a ``Static`` built from a
    plain string or markup, so it always renders either a Rich
    :class:`~rich.text.Text` or Textual's own :class:`~textual.content.Content`
    -- both expose ``.plain``. The asserts are the runtime checks that back
    those assumptions for the type checker.
    """
    assert isinstance(widget, Static)
    rendered = widget.render()
    assert isinstance(rendered, (Text, Content))
    return rendered.plain


def _make_app(debug, shots=5):
    """A ``TimelineApp`` (plus its session and timeline) over one traced bell circuit."""
    timeline = traced_bell(debug, shots=shots)
    session = CliSession([timeline])
    app = TimelineApp(session, "viz (test)")
    return app, session, timeline
