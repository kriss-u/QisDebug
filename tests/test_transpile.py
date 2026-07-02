"""Transpile hook: per-pass recording, id propagation, callback chaining."""

from qiskit import transpile
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator

import qdebug
from qdebug import EventType


def _bell(name="bell"):
    """Return a 2-qubit bell circuit with measurements."""
    qc = QuantumCircuit(2, 2, name=name)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def test_transpile_records_start_passes_and_finish():
    """A transpile() call produces started, per-pass, and finished events."""
    qdebug.activate()
    qc = _bell()
    transpiled = transpile(qc, basis_gates=["cx", "u"], optimization_level=1)

    timeline = qdebug.timeline(qc)
    started = timeline.filter(EventType.TRANSPILE_STARTED)
    passes = timeline.filter(EventType.TRANSPILE_PASS)
    finished = timeline.filter(EventType.TRANSPILE_FINISHED)
    assert len(started) == 1
    assert len(finished) == 1
    assert len(passes) > 0
    assert all(p.pass_name for p in passes)
    assert finished[0].duration > 0
    assert finished[0].size == len(transpiled.data)


def test_transpiled_output_joins_the_same_timeline():
    """The id propagates to the output, linking transpile to execution."""
    qdebug.activate()
    qc = _bell("linked")
    transpiled = transpile(qc, backend=AerSimulator(), optimization_level=1)
    assert getattr(transpiled, "_qdebug_id") == qdebug.timeline(qc).circuit_id

    AerSimulator().run(transpiled, shots=10).result()
    timeline = qdebug.timeline(qc)
    assert timeline.filter(EventType.TRANSPILE_FINISHED)
    assert timeline.filter(EventType.EXECUTION_FINISHED)


def test_user_callback_still_runs():
    """QDebug chains, not replaces, a user-supplied transpile callback."""
    qdebug.activate()
    seen = []

    def callback(**details):
        seen.append(details["pass_"].name())

    transpile(_bell(), basis_gates=["cx", "u"], callback=callback)
    assert seen


def test_changed_snapshot_policy_keeps_some_circuits():
    """With the default policy, passes that changed the dag carry a snapshot."""
    qdebug.activate()
    qc = _bell()
    transpile(qc, basis_gates=["cx", "u"], optimization_level=1)
    passes = qdebug.timeline(qc).filter(EventType.TRANSPILE_PASS)
    with_snapshot = [p for p in passes if p.circuit is not None]
    without_snapshot = [p for p in passes if p.circuit is None]
    assert with_snapshot
    assert all(p.changed for p in with_snapshot)
    assert all(not p.changed for p in without_snapshot)


def test_snapshot_policy_none_records_passes_without_circuits():
    """transpile_snapshots='none' still records pass events, minus snapshots."""
    qdebug.configure(transpile_snapshots="none")
    qdebug.activate()
    qc = _bell()
    transpile(qc, basis_gates=["cx", "u"])
    timeline = qdebug.timeline(qc)
    passes = timeline.filter(EventType.TRANSPILE_PASS)
    assert passes
    assert all(p.circuit is None for p in passes)
    assert timeline.filter(EventType.TRANSPILE_FINISHED)[0].circuit is None


def test_layout_widening_is_detected_as_a_change():
    """Passes that only add wires (ancilla allocation) still get snapshots.

    Depth, size and op counts are identical before and after widening, so
    this exercises the structural part of the change fingerprint.
    """
    qdebug.activate()
    qc = _bell()
    transpile(qc, coupling_map=[[0, 1], [1, 2], [2, 3], [3, 4]], basis_gates=["cx", "u"])
    passes = qdebug.timeline(qc).filter(EventType.TRANSPILE_PASS)
    widened = [p for p in passes if p.circuit is not None and p.circuit.num_qubits == 5]
    assert widened
    assert all(p.changed for p in widened)


def test_layout_permutation_is_detected_as_a_change():
    """A pure qubit relabeling (same depth, size and ops) is still a change."""
    qdebug.activate()
    qc = _bell()
    transpile(qc, coupling_map=[[0, 1], [1, 0]], initial_layout=[1, 0], basis_gates=["cx", "u"])
    passes = qdebug.timeline(qc).filter(EventType.TRANSPILE_PASS)
    apply_layout = [p for p in passes if p.pass_name == "ApplyLayout"]
    assert apply_layout
    assert apply_layout[0].changed
    assert apply_layout[0].circuit is not None


def test_finished_event_carries_the_final_circuit():
    """The finished event freezes a copy of the transpiled output."""
    qdebug.activate()
    qc = _bell()
    transpiled = transpile(qc, basis_gates=["cx", "u"], optimization_level=1)
    finished = qdebug.timeline(qc).filter(EventType.TRANSPILE_FINISHED)[0]
    assert finished.circuit is not None
    assert finished.circuit == transpiled
    assert getattr(finished.circuit, "_qdebug_id", None) is None
    size_before = finished.circuit.size()
    transpiled.x(0)
    assert finished.circuit.size() == size_before


def test_preset_pass_manager_run_is_recorded():
    """Directly running a preset PassManager records the same telemetry."""
    qdebug.activate()
    qc = _bell("pm-run")
    pm = generate_preset_pass_manager(optimization_level=1, basis_gates=["cx", "u"])
    out = pm.run(qc)
    timeline = qdebug.timeline(qc)
    assert timeline.filter(EventType.TRANSPILE_STARTED)
    assert timeline.filter(EventType.TRANSPILE_PASS)
    assert timeline.filter(EventType.TRANSPILE_FINISHED)
    assert getattr(out, "_qdebug_id") == timeline.circuit_id


def test_transpiling_a_list_of_circuits_records_start_and_finish():
    """Multi-circuit transpiles record boundaries for every circuit."""
    qdebug.activate()
    circuits = [_bell(f"batch-{i}") for i in range(2)]
    transpile(circuits, basis_gates=["cx", "u"])
    for qc in circuits:
        timeline = qdebug.timeline(qc)
        assert timeline.filter(EventType.TRANSPILE_STARTED)
        assert timeline.filter(EventType.TRANSPILE_FINISHED)


def test_backend_internal_compilation_records_no_transpile_events():
    """Aer's own control-flow compilation during run() stays off the timeline.

    Aer compiles on a worker thread where context-var suspension is not
    visible, so this exercises the suspended-circuit-id path instead.
    """
    qdebug.activate()
    qc = QuantumCircuit(2, 2, name="dynamic")
    qc.h(0)
    qc.measure(0, 0)
    with qc.if_test((qc.clbits[0], 1)):
        qc.x(1)
    qc.measure(1, 1)
    AerSimulator().run(qc, shots=10).result()
    timeline = qdebug.timeline(qc)
    assert timeline.filter(EventType.EXECUTION_FINISHED)
    assert not timeline.transpile_events
    notes = timeline.filter(EventType.NOTE)
    assert any("backend-internal" in n.message for n in notes)


def test_hand_built_pass_manager_is_recorded():
    """A user-constructed PassManager, not just presets, is instrumented."""
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import Depth, Optimize1qGatesDecomposition

    qdebug.activate()
    qc = _bell("custom-pm")
    pm = PassManager([Depth(), Optimize1qGatesDecomposition()])
    pm.run(qc)
    timeline = qdebug.timeline(qc)
    assert timeline.filter(EventType.TRANSPILE_STARTED)
    passes = timeline.filter(EventType.TRANSPILE_PASS)
    assert {p.pass_name for p in passes} == {"Depth", "Optimize1qGatesDecomposition"}
    assert timeline.filter(EventType.TRANSPILE_FINISHED)


def test_construction_of_transpiled_circuit_not_polluted():
    """Internal circuits built during transpilation create no timelines."""
    qdebug.activate()
    qc = _bell("clean")
    before = {t.circuit_id for t in qdebug.timelines()}
    transpile(qc, basis_gates=["cx", "u"], optimization_level=2)
    after = {t.circuit_id for t in qdebug.timelines()}
    assert after == before
