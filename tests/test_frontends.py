"""Frontend factory: environment detection, forcing, and the CLI fallback."""

import builtins

import pytest
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit_aer import AerSimulator

from qisdebug._frontends import _render, detect_environment, get_frontend
from qisdebug._frontends.cli import CLIFrontend
from qisdebug._frontends.gui import QtFrontend


def _traced_bell(debug):
    """Run a bell circuit under instrumentation and return its timeline."""
    debug.start()
    qc = QuantumCircuit(2, 2, name="viz")
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    AerSimulator().run(qc, shots=10).result()
    return debug.get(qc)


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


def test_cli_frontend_handles_eof(debug, monkeypatch, capsys):
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
    """mark a / mark b pin the current event; the diff panel appears once both are set."""
    timeline = _traced_bell(debug)
    responses = iter(["mark a", "n", "n", "mark b", "q"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(responses))

    CLIFrontend().show([timeline])

    out = capsys.readouterr().out
    assert "marked a:" in out
    assert "marked b:" in out
    assert "-- Diff A vs B --" in out
    assert "gates:" in out


def test_cli_frontend_unmark_clears_diff_panel(debug, monkeypatch, capsys):
    """unmark all drops both marks, so the diff panel stops appearing."""
    timeline = _traced_bell(debug)
    responses = iter(["mark a", "n", "mark b", "unmark all", "q"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(responses))

    CLIFrontend().show([timeline])

    out = capsys.readouterr().out
    assert "cleared all marks" in out
    # After clearing, the final frame print (from "unmark all") has no diff panel.
    assert out.rsplit("cleared all marks", 1)[1].count("-- Diff A vs B --") == 0


@pytest.mark.skipif(
    not QtFrontend().is_available(), reason="no display available for Qt"
)
def test_gui_frontend_mark_and_diff_panel(debug, monkeypatch):
    """Mark A / Mark B populate the status label and diff text; clearing resets both."""
    from PySide6.QtWidgets import QApplication

    from qisdebug._frontends import gui as gui_module

    timeline = _traced_bell(debug)
    monkeypatch.setattr(QApplication, "exec", lambda self: None)
    app = QApplication.instance() or QApplication([])
    merged, tracks, active_labels = _render.merge_timelines([timeline])
    view = gui_module._TimelineView(  # noqa: SLF001 (internal, but this is the frontend's own test)
        merged=merged,
        tracks=tracks,
        active_labels=active_labels,
        timelines_by_id={timeline.circuit_id: timeline},
    )
    window = gui_module._TimelineWindow("viz", view)  # noqa: SLF001

    assert "(unmarked)" in window.mark_status.text()
    assert window.tabs.indexOf(window._diff_panel) < 0  # noqa: SLF001

    window._mark_a()  # noqa: SLF001
    assert window.tabs.indexOf(window._diff_panel) < 0  # noqa: SLF001

    window._goto(1)  # noqa: SLF001
    window._mark_b()  # noqa: SLF001

    assert "(unmarked)" not in window.mark_status.text()
    assert window.tabs.indexOf(window._diff_panel) >= 0  # noqa: SLF001
    assert "gates:" in window.diff_text.toPlainText()

    window._clear_marks()  # noqa: SLF001
    assert window.marks == {"a": None, "b": None}
    assert window.tabs.indexOf(window._diff_panel) < 0  # noqa: SLF001
    del app
