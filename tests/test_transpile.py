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
    passes = qdebug.timeline(qc).filter(EventType.TRANSPILE_PASS)
    assert passes
    assert all(p.circuit is None for p in passes)


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


def test_construction_of_transpiled_circuit_not_polluted():
    """Internal circuits built during transpilation create no timelines."""
    qdebug.activate()
    qc = _bell("clean")
    before = {t.circuit_id for t in qdebug.timelines()}
    transpile(qc, basis_gates=["cx", "u"], optimization_level=2)
    after = {t.circuit_id for t in qdebug.timelines()}
    assert after == before
