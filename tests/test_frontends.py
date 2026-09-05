"""Frontend factory: environment detection, forcing, and the CLI fallback."""

import builtins

import pytest
from _helpers import build_gui_window, traced_bell as _traced_bell

from qisdebug._frontends import detect_environment, get_frontend
from qisdebug._frontends.cli import CLIFrontend
from qisdebug._frontends.gui import QtFrontend


def test_detect_environment_prefers_gui(monkeypatch):
    """A usable display wins over CLI."""
    monkeypatch.setattr("qisdebug._frontends.detect.gui_available", lambda: True)
    assert detect_environment() == "gui"


def test_detect_environment_falls_back_to_cli(monkeypatch):
    """With no display, CLI is the universal fallback."""
    monkeypatch.setattr("qisdebug._frontends.detect.gui_available", lambda: False)
    assert detect_environment() == "cli"


def test_get_frontend_unknown_name_raises():
    """Forcing a nonsense frontend name is a clear ValueError, not a KeyError."""
    with pytest.raises(ValueError, match="Unknown qisdebug frontend"):
        get_frontend("holodeck")


def test_get_frontend_forced_unavailable_raises(monkeypatch):
    """Forcing GUI with no display raises RuntimeError explaining why."""
    monkeypatch.setattr("qisdebug._frontends.gui.gui_available", lambda: False)
    with pytest.raises(RuntimeError, match="no display is available"):
        get_frontend("gui")


def test_get_frontend_respects_env_var(monkeypatch):
    """QISDEBUG_FRONTEND is honored when no explicit argument is given."""
    monkeypatch.setenv("QISDEBUG_FRONTEND", "cli")
    assert isinstance(get_frontend(), CLIFrontend)


def test_get_frontend_argument_overrides_env_var(monkeypatch):
    """An explicit frontend argument takes precedence over the env var."""
    monkeypatch.setenv("QISDEBUG_FRONTEND", "gui")
    monkeypatch.setattr("qisdebug._frontends.gui.gui_available", lambda: True)
    assert isinstance(get_frontend("cli"), CLIFrontend)


def test_cli_frontend_is_always_available():
    """CLI never fails to be forced; it's the guaranteed fallback."""
    frontend = CLIFrontend()
    assert frontend.is_available() is True


def test_cli_frontend_navigates_events(debug, monkeypatch, capsys):
    """Feeding next/quit through stdin steps through events and then exits."""
    timeline = _traced_bell(debug)
    responses = iter(["n", "n", "q"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(responses))

    CLIFrontend().show([timeline])

    out = capsys.readouterr().out
    assert "1/" in out
    assert "2/" in out
    assert "3/" in out


def test_cli_frontend_handles_eof(debug, monkeypatch):
    """A closed stdin (e.g. piped input) exits cleanly instead of raising."""
    timeline = _traced_bell(debug)

    def raise_eof(prompt=""):
        raise EOFError

    monkeypatch.setattr(builtins, "input", raise_eof)
    CLIFrontend().show([timeline])  # should not raise


@pytest.mark.skipif(
    not QtFrontend().is_available(), reason="no display available for Qt"
)
def test_gui_frontend_builds_window_without_mainloop(debug, monkeypatch):
    """Smoke-test window construction without entering the blocking event loop."""
    from PySide6.QtWidgets import QApplication

    timeline = _traced_bell(debug)
    monkeypatch.setattr(QApplication, "exec", lambda self: None)
    QtFrontend().show([timeline])


def test_cli_frontend_mark_and_diff_appear_after_both_slots_set(
    debug, monkeypatch, capsys
):
    """mark a / mark b pin the current event; a collapsed "diff ready" line
    appears once both are set, and "diff full" expands it on demand."""
    timeline = _traced_bell(debug)
    responses = iter(["mark a", "n", "n", "mark b", "diff full", "q"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(responses))

    CLIFrontend().show([timeline])

    out = capsys.readouterr().out
    assert "marked a:" in out
    assert "marked b:" in out
    assert "diff ready --" in out
    assert "gates:" in out


def test_cli_frontend_unmark_clears_diff_panel(debug, monkeypatch, capsys):
    """unmark all drops both marks, so the diff indicator stops appearing."""
    timeline = _traced_bell(debug)
    responses = iter(["mark a", "n", "mark b", "unmark all", "q"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(responses))

    CLIFrontend().show([timeline])

    out = capsys.readouterr().out
    assert "cleared all marks" in out
    # After clearing, the final frame print (from "unmark all") has no diff indicator.
    assert out.rsplit("cleared all marks", 1)[1].count("diff ready --") == 0


@pytest.mark.skipif(
    not QtFrontend().is_available(), reason="no display available for Qt"
)
def test_gui_frontend_mark_and_diff_panel(debug, monkeypatch):
    """Mark A / Mark B populate the status label and diff text; clearing resets both.

    protected-access: this test exercises the GUI module's own mark/diff
    internals, which have no public API surface.
    """
    # pylint: disable=protected-access
    timeline = _traced_bell(debug)
    app, window = build_gui_window(monkeypatch, timeline)

    assert "(unmarked)" in window.mark_status.text()
    assert window.tabs.indexOf(window._diff_panel) < 0
    assert window.show_diff_button.isHidden()

    window._mark_a()
    assert window.tabs.indexOf(window._diff_panel) < 0
    assert window.show_diff_button.isHidden()

    window._goto(1)
    window._mark_b()

    assert "(unmarked)" not in window.mark_status.text()
    assert window.tabs.indexOf(window._diff_panel) >= 0
    assert "gates:" in window.diff_text.toPlainText()
    assert not window.show_diff_button.isHidden()

    timeline_tab = window.tabs.widget(0)
    assert timeline_tab is not None
    window.tabs.setCurrentWidget(timeline_tab)
    window._show_diff()
    assert window.tabs.currentWidget() is window._diff_panel

    window._clear_marks()
    assert window.marks == {"a": None, "b": None}
    assert window.tabs.indexOf(window._diff_panel) < 0


@pytest.mark.skipif(
    not QtFrontend().is_available(), reason="no display available for Qt"
)
def test_gui_frontend_question_mark_opens_help(debug, monkeypatch):
    """Regression test: the "?" shortcut previously didn't exist at all."""
    # pylint: disable=protected-access
    from PySide6.QtWidgets import QMessageBox

    timeline = _traced_bell(debug)
    app, window = build_gui_window(monkeypatch, timeline)

    calls = []
    monkeypatch.setattr(
        QMessageBox, "information", lambda *args, **kwargs: calls.append(args)
    )
    window._help_shortcut.activated.emit()
    assert len(calls) == 1


@pytest.mark.skipif(
    not QtFrontend().is_available(), reason="no display available for Qt"
)
def test_gui_frontend_collapses_to_active_circuit_by_default(debug, monkeypatch):
    """Only the circuit relevant to the current event is drawn until "Show
    all circuits" is checked."""
    # pylint: disable=protected-access
    timeline = _traced_bell(debug)
    app, window = build_gui_window(monkeypatch, timeline)

    assert window.show_all_circuits_checkbox.isChecked() is False
    assert "(no circuit recorded yet)" not in window.text.toPlainText()

    window.show_all_circuits_checkbox.setChecked(True)
    assert window._show_all_circuits is True

    window.show_all_circuits_checkbox.setChecked(False)
    assert window._show_all_circuits is False


@pytest.mark.skipif(
    not QtFrontend().is_available(), reason="no display available for Qt"
)
def test_gui_frontend_zoom_changes_circuit_font_size(debug, monkeypatch):
    # pylint: disable=protected-access
    timeline = _traced_bell(debug)
    app, window = build_gui_window(monkeypatch, timeline)

    default_size = window.text.font().pointSize()
    window._zoom_in()
    assert window.text.font().pointSize() == default_size + 1
    window._zoom_reset()
    assert window.text.font().pointSize() == default_size
    assert window.show_diff_button.isHidden()
    del app
