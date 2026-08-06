"""Conditional breakpoints: the predicate model, live pause/resume, and both
frontends' navigation over an already-captured timeline.
"""

import builtins
import threading

import pytest
from _helpers import bell_circuit, build_gui_window, traced_bell
from qiskit.circuit import QuantumCircuit

from qisdebug._breakpoints import (
    AnomalyCondition,
    GateMatchCondition,
    SeqThresholdCondition,
    WatchThresholdCondition,
    matches,
)
from qisdebug._events import (
    ExecutionFinishedEvent,
    GateEvent,
    StateSnapshotEvent,
    Timeline,
)
from qisdebug._frontends import _breakpoints as frontend_bp
from qisdebug._frontends import _render
from qisdebug._frontends.cli import CLIFrontend
from qisdebug._frontends.gui import QtFrontend
from qisdebug._store import BreakpointStopped

# -- Pure predicate matching --------------------------------------------


def test_gate_match_condition_matches_name_and_any_qubits():
    timeline = Timeline(circuit_id="c1")
    condition = GateMatchCondition(gate="h")
    assert matches(condition, GateEvent(seq=1, gate="h", qubits=(3,)), timeline)
    assert not matches(condition, GateEvent(seq=1, gate="x", qubits=(3,)), timeline)


def test_gate_match_condition_qubits_must_match_as_a_set():
    timeline = Timeline(circuit_id="c1")
    condition = GateMatchCondition(gate="cx", qubits=(0, 1))
    assert matches(condition, GateEvent(seq=1, gate="cx", qubits=(0, 1)), timeline)
    assert matches(condition, GateEvent(seq=1, gate="cx", qubits=(1, 0)), timeline)
    assert not matches(condition, GateEvent(seq=1, gate="cx", qubits=(0, 2)), timeline)


def test_condition_circuit_id_scopes_the_match():
    timeline = Timeline(circuit_id="c1")
    condition = GateMatchCondition(gate="h", circuit_id="other-circuit")
    assert not matches(condition, GateEvent(seq=1, gate="h", qubits=(0,)), timeline)


def test_seq_threshold_condition_matches_at_or_after_seq():
    timeline = Timeline(circuit_id="c1")
    condition = SeqThresholdCondition(seq=5)
    assert not matches(condition, GateEvent(seq=4, gate="h", qubits=(0,)), timeline)
    assert matches(condition, GateEvent(seq=5, gate="h", qubits=(0,)), timeline)
    assert matches(condition, GateEvent(seq=6, gate="h", qubits=(0,)), timeline)


def test_watch_threshold_condition_matches_qubit_probability():
    timeline = Timeline(circuit_id="c1", num_qubits=1)
    event = StateSnapshotEvent(seq=1, step=1, num_qubits=1, statevector=[0.0, 1.0])
    above = WatchThresholdCondition(
        circuit_id="c1", watch_kind="qubit", ref="0", comparator="gt", threshold=0.9
    )
    below = WatchThresholdCondition(
        circuit_id="c1", watch_kind="qubit", ref="0", comparator="lt", threshold=0.1
    )
    assert matches(above, event, timeline)
    assert not matches(below, event, timeline)


def test_watch_threshold_condition_qubit_ignores_non_snapshot_events():
    timeline = Timeline(circuit_id="c1")
    condition = WatchThresholdCondition(
        circuit_id="c1", watch_kind="qubit", ref="0", comparator="gt", threshold=0.0
    )
    assert not matches(condition, GateEvent(seq=1, gate="h", qubits=(0,)), timeline)


def test_watch_threshold_condition_matches_clbit_probability():
    timeline = Timeline(circuit_id="c1")
    event = ExecutionFinishedEvent(seq=1, backend="aer", counts={"1": 10, "0": 0})
    condition = WatchThresholdCondition(
        circuit_id="c1", watch_kind="clbit", ref="0", comparator="ge", threshold=0.5
    )
    assert matches(condition, event, timeline)


def test_watch_threshold_condition_register_resolves_to_its_bits():
    timeline = Timeline(circuit_id="c1", num_qubits=2, qregs={"q": [0, 1]})
    event = StateSnapshotEvent(seq=1, step=1, num_qubits=2, statevector=[0, 0, 0, 1])
    condition = WatchThresholdCondition(
        circuit_id="c1", watch_kind="register", ref="q", comparator="gt", threshold=0.9
    )
    assert matches(condition, event, timeline)


def test_anomaly_condition_matches_non_normalized_snapshot():
    timeline = Timeline(circuit_id="c1")
    event = StateSnapshotEvent(seq=1, step=0, statevector=[2.0, 0.0])
    assert matches(AnomalyCondition(), event, timeline)


def test_anomaly_condition_severity_filter():
    timeline = Timeline(circuit_id="c1")
    event = ExecutionFinishedEvent(seq=1, backend="aer", success=True, counts={})
    assert matches(AnomalyCondition(min_severity="warning"), event, timeline)
    assert not matches(AnomalyCondition(min_severity="error"), event, timeline)


# -- Live pause/resume across threads ------------------------------------


def test_add_breakpoint_pauses_the_hit_thread_until_resumed(debug):
    debug.start()
    debug.add_breakpoint(GateMatchCondition(gate="cx"))
    qc = QuantumCircuit(2)
    qc.h(0)
    progressed = []

    def build():
        qc.cx(0, 1)
        progressed.append("past cx")

    worker = threading.Thread(target=build)
    worker.start()
    try:
        hit = debug.wait_for_breakpoint(timeout=5)
        assert hit is not None
        assert hit.condition.gate == "cx"
        assert not progressed
        assert debug.store.breakpoints.is_paused
        debug.resume()
        worker.join(timeout=5)
        assert not worker.is_alive()
        assert progressed == ["past cx"]
    finally:
        debug.resume()
        worker.join(timeout=5)


def test_stop_raises_breakpoint_stopped_in_the_paused_thread(debug):
    debug.start()
    debug.add_breakpoint(GateMatchCondition(gate="cx"))
    qc = QuantumCircuit(2)
    qc.h(0)
    caught = []

    def build():
        try:
            qc.cx(0, 1)
        except BreakpointStopped as exc:
            caught.append(exc)

    worker = threading.Thread(target=build)
    worker.start()
    try:
        assert debug.wait_for_breakpoint(timeout=5) is not None
        debug.store.breakpoints.stop()
        worker.join(timeout=5)
        assert not worker.is_alive()
        assert len(caught) == 1
    finally:
        debug.resume()
        worker.join(timeout=5)


def test_add_breakpoint_enforces_max_breakpoints(debug):
    debug.configure(max_breakpoints=1)
    debug.add_breakpoint(SeqThresholdCondition(seq=1))
    with pytest.raises(ValueError, match="max_breakpoints"):
        debug.add_breakpoint(SeqThresholdCondition(seq=2))


def test_remove_and_list_breakpoints(debug):
    handle = debug.add_breakpoint(SeqThresholdCondition(seq=1))
    assert [h for h, _ in debug.list_breakpoints()] == [handle]
    debug.remove_breakpoint(handle)
    assert debug.list_breakpoints() == []


def test_wait_for_breakpoint_times_out_when_nothing_hits(debug):
    debug.start()
    debug.add_breakpoint(GateMatchCondition(gate="does-not-exist"))
    bell_circuit("viz")  # never touches qisdebug's construction hook path meaningfully
    assert debug.wait_for_breakpoint(timeout=0.05) is None


# -- Frontend navigation over an already-captured timeline ---------------


def test_find_next_hit_locates_the_first_matching_event(debug):
    timeline = traced_bell(debug, shots=5)
    merged, _tracks, _labels = _render.merge_timelines([timeline])
    conditions = {"1": GateMatchCondition(gate="cx")}
    hit = frontend_bp.find_next_hit(merged, conditions, start_index=0)
    assert hit is not None
    index, handle = hit
    assert handle == "1"
    assert merged[index][1].describe().startswith("cx")


def test_find_next_hit_returns_none_when_nothing_matches(debug):
    timeline = traced_bell(debug, shots=5)
    merged, _tracks, _labels = _render.merge_timelines([timeline])
    conditions = {"1": GateMatchCondition(gate="does-not-exist")}
    assert frontend_bp.find_next_hit(merged, conditions, start_index=0) is None


def test_describe_condition_renders_each_kind():
    assert "gate cx" in frontend_bp.describe_condition(GateMatchCondition(gate="cx"))
    assert "seq >= 5" in frontend_bp.describe_condition(SeqThresholdCondition(seq=5))
    assert "gt 0.9" in frontend_bp.describe_condition(
        WatchThresholdCondition(
            circuit_id="c1", watch_kind="qubit", ref="0", comparator="gt", threshold=0.9
        )
    )
    assert "anomaly >= error" in frontend_bp.describe_condition(
        AnomalyCondition(min_severity="error")
    )


def test_parse_condition_args_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown breakpoint kind"):
        frontend_bp.parse_condition_args("nonsense", [], "c1")


def test_parse_condition_args_builds_gate_condition():
    condition = frontend_bp.parse_condition_args("gate", ["cx", "0", "1"], "c1")
    assert isinstance(condition, GateMatchCondition)
    assert condition.gate == "cx"
    assert condition.qubits == (0, 1)
    assert condition.circuit_id == "c1"


# -- CLI integration -------------------------------------------------------


def test_cli_break_continue_and_unbreak_commands(debug, monkeypatch, capsys):
    timeline = traced_bell(debug, shots=5)
    responses = iter(
        ["break gate cx", "continue", "breakpoints", "unbreak 1", "breakpoints", "q"]
    )
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(responses))

    CLIFrontend().show([timeline])

    out = capsys.readouterr().out
    assert "breakpoint 1: gate cx" in out
    assert "hit breakpoint 1: gate cx" in out
    assert "[1] gate cx" in out
    assert "removed breakpoint 1" in out
    assert "no active breakpoints" in out


def test_cli_break_reports_usage_on_bad_input(debug, monkeypatch, capsys):
    timeline = traced_bell(debug, shots=5)
    responses = iter(["break nonsense", "q"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(responses))

    CLIFrontend().show([timeline])

    out = capsys.readouterr().out
    assert "unknown breakpoint kind" in out


# -- GUI integration --------------------------------------------------------


@pytest.mark.skipif(
    not QtFrontend().is_available(), reason="no display available for Qt"
)
def test_gui_frontend_add_breakpoint_and_run_to_next(debug, monkeypatch):
    timeline = traced_bell(debug, shots=5)
    app, window = build_gui_window(monkeypatch, timeline)

    window.breakpoint_kind.setCurrentText("gate")
    window.breakpoint_args.setText("cx")
    window._add_breakpoint()  # pylint: disable=protected-access
    assert len(window.breakpoints) == 1
    assert window.breakpoint_list.count() == 1

    window._run_to_next_breakpoint()  # pylint: disable=protected-access
    _source, event = window.controller.current
    assert event.describe().startswith("cx")

    window.breakpoint_list.setCurrentRow(0)
    window._remove_selected_breakpoint()  # pylint: disable=protected-access
    assert not window.breakpoints
    del app


@pytest.mark.skipif(
    not QtFrontend().is_available(), reason="no display available for Qt"
)
def test_gui_frontend_add_breakpoint_rejects_bad_args(debug, monkeypatch):
    timeline = traced_bell(debug, shots=5)
    app, window = build_gui_window(monkeypatch, timeline)

    window.breakpoint_kind.setCurrentText("seq")
    window.breakpoint_args.setText("not-a-number")
    window._add_breakpoint()  # pylint: disable=protected-access
    assert not window.breakpoints
    del app
