"""The interactive watch feature: tracking a qubit, clbit, or register."""

import builtins

import pytest
from _helpers import bell_circuit, build_gui_window
from qiskit import transpile
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit import ClassicalRegister, QuantumRegister
from qiskit_aer import AerSimulator

from qisdebug._frontends import _render
from qisdebug._frontends.cli import CLIFrontend
from qisdebug._frontends.gui import QtFrontend


def _traced_bell_with_registers(debug):
    """Run a bell circuit built from named registers, fully instrumented."""
    debug.start()
    qreg = QuantumRegister(2, name="q")
    creg = ClassicalRegister(2, name="c")
    qc = QuantumCircuit(qreg, creg, name="viz")
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    AerSimulator().run(qc, shots=100).result()
    return debug.get(qc)


def test_timeline_captures_register_metadata(debug):
    """Register names map to their flat qubit/clbit indices."""
    timeline = _traced_bell_with_registers(debug)
    assert timeline.qregs == {"q": [0, 1]}
    assert timeline.cregs == {"c": [0, 1]}


def test_watch_qubit_reports_gate_touches_before_any_state_exists(debug):
    """Right after the h gate is recorded, no execution has happened yet."""
    timeline = _traced_bell_with_registers(debug)
    target = _render.WatchTarget(circuit_id=timeline.circuit_id, kind="qubit", ref="0")
    after_h_seq = timeline.gate_events[0].seq
    text = _render.render_watch(target, timeline, after_h_seq)
    assert "gates touched: 1" in text
    assert "h q[0]" in text
    assert "P(1)" not in text  # snapshots are only seq'd once execution finishes


def test_watch_qubit_reports_state_once_execution_has_run(debug):
    """Once playback reaches the end, the watch shows the post-h probability."""
    timeline = _traced_bell_with_registers(debug)
    target = _render.WatchTarget(circuit_id=timeline.circuit_id, kind="qubit", ref="0")
    last_seq = timeline.events[-1].seq
    text = _render.render_watch(target, timeline, last_seq)
    assert "P(1)=0.5000" in text


def test_watch_qubit_bloch_vector_for_ground_state(debug):
    """A qubit that's still |0> at the snapshot has a Bloch vector at z=+1."""
    timeline = _traced_bell_with_registers(debug)
    target = _render.WatchTarget(circuit_id=timeline.circuit_id, kind="qubit", ref="1")
    snapshot = timeline.snapshots[0]
    text = _render.render_watch(target, timeline, snapshot.seq)
    assert "bloch=(+0.000, +0.000, +1.000)" in text
    before_anything = timeline.events[0].seq - 1
    assert _render.render_watch(target, timeline, before_anything) == "* viz — qubit 1"


def test_watch_register_aggregates_its_qubits(debug):
    """Watching register q covers both of its qubits' gate touches and state."""
    timeline = _traced_bell_with_registers(debug)
    target = _render.WatchTarget(
        circuit_id=timeline.circuit_id, kind="register", ref="q"
    )
    last_seq = timeline.events[-1].seq
    text = _render.render_watch(target, timeline, last_seq)
    assert "gates touched: 4" in text  # h, cx, measure, measure
    assert "qubit 0:" in text
    assert "qubit 1:" in text


def test_watch_clbit_reports_measured_distribution(debug):
    """Watching a clbit sees the measurement gate plus the counts-derived P(1)."""
    timeline = _traced_bell_with_registers(debug)
    target = _render.WatchTarget(circuit_id=timeline.circuit_id, kind="clbit", ref="0")
    last_seq = timeline.events[-1].seq
    text = _render.render_watch(target, timeline, last_seq)
    assert "measure q[0] -> c[0]" in text
    assert "clbit 0: P(1)=" in text


def test_watch_qubit_reports_touches_on_every_transpile_track(debug):
    """A qubit watch reports gates touched on construction *and* transpiled
    lineages, not just construction -- a transpiled circuit's gates aren't
    their own gate_added events, so they must show up here too."""
    debug.start()
    qc = bell_circuit("viz")
    transpile(qc, basis_gates=["cx", "u"], optimization_level=1)
    timeline = debug.get(qc)

    target = _render.WatchTarget(circuit_id=timeline.circuit_id, kind="qubit", ref="0")
    last_seq = timeline.events[-1].seq
    text = _render.render_watch(target, timeline, last_seq)

    assert "[Circuit (construction)] gates touched:" in text
    assert "[Circuit (transpiled)] gates touched:" in text


def test_watch_track_reports_latest_pass_stats(debug):
    """A ``kind="track"`` watch shows the latest transpile pass's stats for
    that circuit lineage, not any bit-level state."""
    debug.start()
    qc = bell_circuit("viz")
    transpile(qc, basis_gates=["cx", "u"], optimization_level=1)
    timeline = debug.get(qc)

    target = _render.WatchTarget(
        circuit_id=timeline.circuit_id, kind="track", ref="transpiled"
    )
    last_seq = timeline.events[-1].seq
    text = _render.render_watch(target, timeline, last_seq)

    assert "viz — track transpiled" in text
    assert "latest pass:" in text
    assert "depth:" in text and "size:" in text and "ops:" in text


def test_watch_track_before_transpile_started_reports_not_found(debug):
    """A track watch on a lineage that doesn't exist yet at ``upto_seq``
    says so instead of showing stale or empty stats."""
    debug.start()
    qc = bell_circuit("viz")
    timeline = debug.get(qc)

    target = _render.WatchTarget(
        circuit_id=timeline.circuit_id, kind="track", ref="transpiled"
    )
    before_anything = timeline.events[0].seq - 1
    text = _render.render_watch(target, timeline, before_anything)
    assert "not found on this circuit" in text


def test_watch_unresolved_register_reports_not_found(debug):
    """A watch on a register name that doesn't exist degrades cleanly."""
    timeline = _traced_bell_with_registers(debug)
    target = _render.WatchTarget(
        circuit_id=timeline.circuit_id, kind="register", ref="nope"
    )
    text = _render.render_watch(target, timeline, timeline.events[-1].seq)
    assert "not found on this circuit" in text


def test_render_watches_skips_unknown_circuit_ids(debug):
    """A stale watch whose timeline was evicted is silently dropped, not an error."""
    timeline = _traced_bell_with_registers(debug)
    target = _render.WatchTarget(circuit_id="does-not-exist", kind="qubit", ref="0")
    assert _render.render_watches([target], {timeline.circuit_id: timeline}, 0) == ""


def test_render_watch_diff_marks_changed_lines(debug):
    """A watch whose value differs between the two seqs shows up as a
    -/+ pair; a watch whose gates-touched count hasn't changed yet between
    the two seqs stays unmarked plain text.
    """
    timeline = _traced_bell_with_registers(debug)
    target = _render.WatchTarget(circuit_id=timeline.circuit_id, kind="qubit", ref="0")
    before_anything = timeline.events[0].seq - 1
    last_seq = timeline.events[-1].seq

    text = _render.render_watch_diff(
        [target], {timeline.circuit_id: timeline}, before_anything, last_seq
    )

    assert "    * viz — qubit 0" in text
    assert "  + " in text and "P(1)=0.5000" in text


def test_render_watch_diff_unchanged_watch_has_no_diff_markers(debug):
    """Diffing a watch against the same seq twice reports it as unchanged."""
    timeline = _traced_bell_with_registers(debug)
    target = _render.WatchTarget(circuit_id=timeline.circuit_id, kind="qubit", ref="0")
    last_seq = timeline.events[-1].seq

    text = _render.render_watch_diff(
        [target], {timeline.circuit_id: timeline}, last_seq, last_seq
    )

    assert "  - " not in text
    assert "  + " not in text
    assert "P(1)=0.5000" in text


def test_render_watch_diff_skips_unknown_circuit_ids(debug):
    """Mirrors :func:`render_watches`: a stale watch is dropped, not an error."""
    timeline = _traced_bell_with_registers(debug)
    target = _render.WatchTarget(circuit_id="does-not-exist", kind="qubit", ref="0")
    assert (
        _render.render_watch_diff([target], {timeline.circuit_id: timeline}, 0, 0) == ""
    )


def test_cli_watch_and_unwatch_commands(debug, monkeypatch, capsys):
    """watch/unwatch/watches drive the tracked-value panel from the REPL."""
    timeline = _traced_bell_with_registers(debug)
    responses = iter(["watch qubit 0", "l", "unwatch qubit 0", "watches", "q"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(responses))

    CLIFrontend().show([timeline])

    out = capsys.readouterr().out
    assert "watching viz — qubit 0" in out
    assert "P(1)=0.5000" in out
    assert "no active watches" in out


@pytest.mark.skipif(
    not QtFrontend().is_available(), reason="no display available for Qt"
)
def test_gui_frontend_add_and_remove_watch(debug, monkeypatch):
    """Adding a watch through the combo boxes updates the watch text; removing clears it."""
    timeline = _traced_bell_with_registers(debug)
    app, window = build_gui_window(monkeypatch, timeline)

    window.watch_kind.setCurrentText("qubit")
    window.watch_ref.setCurrentText("0")
    window._add_watch()  # pylint: disable=protected-access
    assert len(window.watches) == 1
    assert "viz — qubit 0" in window.watch_text.toPlainText()

    window.watch_list.setCurrentRow(0)
    window._remove_selected_watch()  # pylint: disable=protected-access
    assert not window.watches
    assert window.watch_text.toPlainText() == ""
    del app
