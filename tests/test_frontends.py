"""Frontend factory: environment detection, forcing, and the CLI fallback."""

import builtins

import pytest
from _helpers import build_bridge, traced_bell as _traced_bell

from qisdebug._frontends import detect_environment, get_frontend
from qisdebug._frontends.cli import CLIFrontend
from qisdebug._frontends.gui import WebviewFrontend


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
    not WebviewFrontend().is_available(), reason="no display available for pywebview"
)
def test_gui_frontend_builds_window_without_mainloop(debug, monkeypatch):
    """Smoke-test window construction without entering the blocking event loop."""
    import webview

    calls = []
    monkeypatch.setattr(webview, "create_window", lambda *a, **k: calls.append((a, k)))
    monkeypatch.setattr(webview, "start", lambda: None)

    timeline = _traced_bell(debug)
    WebviewFrontend().show([timeline])
    assert len(calls) == 1


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


def test_gui_bridge_mark_and_diff_panel(debug):
    """Mark A / Mark B populate the mark labels and diff text; clearing resets both."""
    timeline = _traced_bell(debug)
    bridge = build_bridge(timeline)

    state = bridge.state()
    assert state["marks"]["label_a"] == "(unmarked)"
    assert state["marks"]["both_marked"] is False

    state = bridge.mark("a")
    assert state["marks"]["both_marked"] is False

    bridge.goto(1)
    state = bridge.mark("b")

    assert state["marks"]["label_a"] != "(unmarked)"
    assert state["marks"]["both_marked"] is True
    assert "gates:" in state["marks"]["diff"]["stage_diff"]

    state = bridge.clear_marks()
    assert state["marks"]["both_marked"] is False
    assert bridge.marks == {"a": None, "b": None}


def test_gui_bridge_help_entries_include_question_mark():
    """Regression test: the "?" shortcut previously didn't exist at all."""
    from qisdebug._frontends.bridge import HELP_ENTRIES

    assert any(key == "?" for key, _ in HELP_ENTRIES)


def test_gui_bridge_collapses_to_active_circuit_by_default(debug):
    """Only the circuit relevant to the current event is drawn until "Show
    all circuits" is toggled on."""
    timeline = _traced_bell(debug)
    bridge = build_bridge(timeline)

    state = bridge.state()
    assert state["show_all_circuits"] is False
    assert "(no circuit recorded yet)" not in state["circuit_text"]

    state = bridge.set_show_all_circuits(True)
    assert state["show_all_circuits"] is True

    state = bridge.set_show_all_circuits(False)
    assert state["show_all_circuits"] is False


def test_gui_bridge_zoom_changes_circuit_font_size(debug):
    """Zooming the circuit view in and out adjusts its font size by one step."""
    timeline = _traced_bell(debug)
    bridge = build_bridge(timeline)

    default_size = bridge.state()["circuit_zoom"]
    state = bridge.zoom_in()
    assert state["circuit_zoom"] == default_size + 1
    state = bridge.zoom_reset()
    assert state["circuit_zoom"] == default_size
    assert state["marks"]["both_marked"] is False
