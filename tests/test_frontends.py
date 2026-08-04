"""Frontend factory: environment detection, forcing, and the CLI fallback."""

import builtins

import pytest
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit_aer import AerSimulator

import qdebug
from conftest import get_timeline
from qdebug._frontends import detect_environment, get_frontend
from qdebug._frontends.cli import CLIFrontend
from qdebug._frontends.gui import QtFrontend


def _traced_bell():
    """Run a bell circuit under instrumentation and return its timeline."""
    qdebug.activate()
    qc = QuantumCircuit(2, 2, name="viz")
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    AerSimulator().run(qc, shots=10).result()
    return get_timeline(qc)


def test_detect_environment_prefers_gui(monkeypatch):
    """A usable display wins over CLI."""
    monkeypatch.setattr("qdebug._frontends.detect.gui_available", lambda: True)
    assert detect_environment() == "gui"


def test_detect_environment_falls_back_to_cli(monkeypatch):
    """With no display, CLI is the universal fallback."""
    monkeypatch.setattr("qdebug._frontends.detect.gui_available", lambda: False)
    assert detect_environment() == "cli"


def test_get_frontend_unknown_name_raises():
    """Forcing a nonsense frontend name is a clear ValueError, not a KeyError."""
    with pytest.raises(ValueError, match="Unknown qdebug frontend"):
        get_frontend("holodeck")


def test_get_frontend_forced_unavailable_raises(monkeypatch):
    """Forcing GUI with no display raises RuntimeError explaining why."""
    monkeypatch.setattr("qdebug._frontends.gui.gui_available", lambda: False)
    with pytest.raises(RuntimeError, match="no display is available"):
        get_frontend("gui")


def test_get_frontend_respects_env_var(monkeypatch):
    """QDEBUG_FRONTEND is honored when no explicit argument is given."""
    monkeypatch.setenv("QDEBUG_FRONTEND", "cli")
    assert isinstance(get_frontend(), CLIFrontend)


def test_get_frontend_argument_overrides_env_var(monkeypatch):
    """An explicit frontend argument takes precedence over the env var."""
    monkeypatch.setenv("QDEBUG_FRONTEND", "gui")
    monkeypatch.setattr("qdebug._frontends.gui.gui_available", lambda: True)
    assert isinstance(get_frontend("cli"), CLIFrontend)


def test_cli_frontend_is_always_available():
    """CLI never fails to be forced; it's the guaranteed fallback."""
    frontend = CLIFrontend()
    assert frontend.is_available() is True


def test_cli_frontend_navigates_events(monkeypatch, capsys):
    """Feeding next/quit through stdin steps through events and then exits."""
    timeline = _traced_bell()
    responses = iter(["n", "n", "q"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(responses))

    CLIFrontend().show([timeline])

    out = capsys.readouterr().out
    assert "1/" in out
    assert "2/" in out
    assert "3/" in out


def test_cli_frontend_handles_eof(monkeypatch, capsys):
    """A closed stdin (e.g. piped input) exits cleanly instead of raising."""
    timeline = _traced_bell()

    def raise_eof(prompt=""):
        raise EOFError

    monkeypatch.setattr(builtins, "input", raise_eof)
    CLIFrontend().show([timeline])  # should not raise


@pytest.mark.skipif(not QtFrontend().is_available(), reason="no display available for Qt")
def test_gui_frontend_builds_window_without_mainloop(monkeypatch):
    """Smoke-test window construction without entering the blocking event loop."""
    from PySide6.QtWidgets import QApplication

    timeline = _traced_bell()
    monkeypatch.setattr(QApplication, "exec", lambda self: None)
    QtFrontend().show([timeline])
