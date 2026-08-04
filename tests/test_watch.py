"""The interactive watch feature: tracking a qubit, clbit, or register."""

import builtins

import pytest
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit import ClassicalRegister, QuantumRegister
from qiskit_aer import AerSimulator

import qisdebug
from conftest import get_timeline
from qisdebug._frontends import _render
from qisdebug._frontends.cli import CLIFrontend
from qisdebug._frontends.gui import QtFrontend


def _traced_bell_with_registers():
    """Run a bell circuit built from named registers, fully instrumented."""
    qisdebug.activate()
    qreg = QuantumRegister(2, name="q")
    creg = ClassicalRegister(2, name="c")
    qc = QuantumCircuit(qreg, creg, name="viz")
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    AerSimulator().run(qc, shots=100).result()
    return get_timeline(qc)


def test_timeline_captures_register_metadata():
    """Register names map to their flat qubit/clbit indices."""
    timeline = _traced_bell_with_registers()
    assert timeline.qregs == {"q": [0, 1]}
    assert timeline.cregs == {"c": [0, 1]}


def test_watch_qubit_reports_gate_touches_before_any_state_exists():
    """Right after the h gate is recorded, no execution has happened yet."""
    timeline = _traced_bell_with_registers()
    target = _render.WatchTarget(circuit_id=timeline.circuit_id, kind="qubit", ref="0")
    after_h_seq = timeline.gate_events[0].seq
    text = _render.render_watch(target, timeline, after_h_seq)
    assert "gates touched: 1" in text
    assert "h q[0]" in text
    assert "P(1)" not in text  # snapshots are only seq'd once execution finishes


def test_watch_qubit_reports_state_once_execution_has_run():
    """Once playback reaches the end, the watch shows the post-h probability."""
    timeline = _traced_bell_with_registers()
    target = _render.WatchTarget(circuit_id=timeline.circuit_id, kind="qubit", ref="0")
    last_seq = timeline.events[-1].seq
    text = _render.render_watch(target, timeline, last_seq)
    assert "P(1)=0.5000" in text


def test_watch_qubit_bloch_vector_for_ground_state():
    """A qubit that's still |0> at the snapshot has a Bloch vector at z=+1."""
    timeline = _traced_bell_with_registers()
    target = _render.WatchTarget(circuit_id=timeline.circuit_id, kind="qubit", ref="1")
    snapshot = timeline.snapshots[0]
    text = _render.render_watch(target, timeline, snapshot.seq)
    assert "bloch=(+0.000, +0.000, +1.000)" in text
    before_anything = timeline.events[0].seq - 1
    assert _render.render_watch(target, timeline, before_anything) == (
        "* viz — qubit 1\n  gates touched: 0"
    )


def test_watch_register_aggregates_its_qubits():
    """Watching register q covers both of its qubits' gate touches and state."""
    timeline = _traced_bell_with_registers()
    target = _render.WatchTarget(circuit_id=timeline.circuit_id, kind="register", ref="q")
    last_seq = timeline.events[-1].seq
    text = _render.render_watch(target, timeline, last_seq)
    assert "gates touched: 4" in text  # h, cx, measure, measure
    assert "qubit 0:" in text
    assert "qubit 1:" in text


def test_watch_clbit_reports_measured_distribution():
    """Watching a clbit sees the measurement gate plus the counts-derived P(1)."""
    timeline = _traced_bell_with_registers()
    target = _render.WatchTarget(circuit_id=timeline.circuit_id, kind="clbit", ref="0")
    last_seq = timeline.events[-1].seq
    text = _render.render_watch(target, timeline, last_seq)
    assert "measure q[0] -> c[0]" in text
    assert "clbit 0: P(1)=" in text


def test_watch_unresolved_register_reports_not_found():
    """A watch on a register name that doesn't exist degrades cleanly."""
    timeline = _traced_bell_with_registers()
    target = _render.WatchTarget(circuit_id=timeline.circuit_id, kind="register", ref="nope")
    text = _render.render_watch(target, timeline, timeline.events[-1].seq)
    assert "not found on this circuit" in text


def test_render_watches_skips_unknown_circuit_ids():
    """A stale watch whose timeline was evicted is silently dropped, not an error."""
    timeline = _traced_bell_with_registers()
    target = _render.WatchTarget(circuit_id="does-not-exist", kind="qubit", ref="0")
    assert _render.render_watches([target], {timeline.circuit_id: timeline}, 0) == ""


def test_cli_watch_and_unwatch_commands(monkeypatch, capsys):
    """watch/unwatch/watches drive the tracked-value panel from the REPL."""
    timeline = _traced_bell_with_registers()
    responses = iter(["watch qubit 0", "l", "unwatch qubit 0", "watches", "q"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(responses))

    CLIFrontend().show([timeline])

    out = capsys.readouterr().out
    assert "watching viz — qubit 0" in out
    assert "P(1)=0.5000" in out
    assert "no active watches" in out


@pytest.mark.skipif(not QtFrontend().is_available(), reason="no display available for Qt")
def test_gui_frontend_add_and_remove_watch(monkeypatch):
    """Adding a watch through the combo boxes updates the text panel; removing clears it."""
    from PySide6.QtWidgets import QApplication

    from qisdebug._frontends import gui as gui_module

    timeline = _traced_bell_with_registers()
    monkeypatch.setattr(QApplication, "exec", lambda self: None)
    app = QApplication.instance() or QApplication([])
    merged, tracks, active_labels = _render.merge_timelines([timeline])
    window = gui_module._TimelineWindow(  # noqa: SLF001 (internal, but this is the frontend's own test)
        "viz", merged, tracks, active_labels, {timeline.circuit_id: timeline}
    )

    window.watch_kind.setCurrentText("qubit")
    window.watch_ref.setCurrentText("0")
    window._add_watch()  # noqa: SLF001
    assert len(window.watches) == 1
    assert "viz — qubit 0" in window.text.toPlainText()

    window.watch_list.setCurrentRow(0)
    window._remove_selected_watch()  # noqa: SLF001
    assert window.watches == []
    del app
