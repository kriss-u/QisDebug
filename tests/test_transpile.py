"""Transpile hook: per-pass recording, id propagation, callback chaining."""

from qiskit import transpile
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator

from qisdebug import EventType


def _bell(name="bell"):
    """Return a 2-qubit bell circuit with measurements."""
    qc = QuantumCircuit(2, 2, name=name)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def test_transpile_records_start_passes_and_finish(debug):
    """A transpile() call produces started, per-pass, and finished events."""
    debug.start()
    qc = _bell()
    transpiled = transpile(qc, basis_gates=["cx", "u"], optimization_level=1)

    timeline = debug.get(qc)
    started = timeline.filter(EventType.TRANSPILE_STARTED)
    passes = timeline.filter(EventType.TRANSPILE_PASS)
    finished = timeline.filter(EventType.TRANSPILE_FINISHED)
    assert len(started) == 1
    assert len(finished) == 1
    assert len(passes) > 0
    assert all(p.pass_name for p in passes)
    assert finished[0].duration > 0
    assert finished[0].size == len(transpiled.data)


def test_transpiled_output_joins_the_same_timeline(debug):
    """The id propagates to the output, linking transpile to execution."""
    debug.start()
    qc = _bell("linked")
    transpiled = transpile(qc, backend=AerSimulator(), optimization_level=1)
    assert getattr(transpiled, "_qisdebug_id") == debug.get(qc).circuit_id

    AerSimulator().run(transpiled, shots=10).result()
    timeline = debug.get(qc)
    assert timeline.filter(EventType.TRANSPILE_FINISHED)
    assert timeline.filter(EventType.EXECUTION_FINISHED)


def test_user_callback_still_runs(debug):
    """QisDebug chains, not replaces, a user-supplied transpile callback."""
    debug.start()
    seen = []

    def callback(pass_, dag, time, property_set, count):
        seen.append(pass_.name())

    transpile(_bell(), basis_gates=["cx", "u"], callback=callback)
    assert seen


def test_changed_snapshot_policy_keeps_some_circuits(debug):
    """With the default policy, passes that changed the dag carry a snapshot."""
    debug.start()
    qc = _bell()
    transpile(qc, basis_gates=["cx", "u"], optimization_level=1)
    passes = debug.get(qc).filter(EventType.TRANSPILE_PASS)
    with_snapshot = [p for p in passes if p.circuit is not None]
    without_snapshot = [p for p in passes if p.circuit is None]
    assert with_snapshot
    assert all(p.changed for p in with_snapshot)
    assert all(not p.changed for p in without_snapshot)


def test_snapshot_policy_none_records_passes_without_circuits(debug):
    """transpile_snapshots='none' still records pass events, minus snapshots."""
    debug.configure(transpile_snapshots="none")
    debug.start()
    qc = _bell()
    transpile(qc, basis_gates=["cx", "u"])
    timeline = debug.get(qc)
    passes = timeline.filter(EventType.TRANSPILE_PASS)
    assert passes
    assert all(p.circuit is None for p in passes)
    assert timeline.filter(EventType.TRANSPILE_FINISHED)[0].circuit is None


def test_layout_widening_is_detected_as_a_change(debug):
    """Passes that only add wires (ancilla allocation) still get snapshots.

    Depth, size and op counts are identical before and after widening, so
    this exercises the structural part of the change fingerprint.
    """
    debug.start()
    qc = _bell()
    transpile(
        qc, coupling_map=[[0, 1], [1, 2], [2, 3], [3, 4]], basis_gates=["cx", "u"]
    )
    passes = debug.get(qc).filter(EventType.TRANSPILE_PASS)
    widened = [p for p in passes if p.circuit is not None and p.circuit.num_qubits == 5]
    assert widened
    assert all(p.changed for p in widened)


def test_layout_permutation_is_detected_as_a_change(debug):
    """A pure qubit relabeling (same depth, size and ops) is still a change."""
    debug.start()
    qc = _bell()
    transpile(
        qc,
        coupling_map=[[0, 1], [1, 0]],
        initial_layout=[1, 0],
        basis_gates=["cx", "u"],
    )
    passes = debug.get(qc).filter(EventType.TRANSPILE_PASS)
    apply_layout = [p for p in passes if p.pass_name == "ApplyLayout"]
    assert apply_layout
    assert apply_layout[0].changed
    assert apply_layout[0].circuit is not None


def test_finished_event_carries_the_final_circuit(debug):
    """The finished event freezes a copy of the transpiled output."""
    debug.start()
    qc = _bell()
    transpiled = transpile(qc, basis_gates=["cx", "u"], optimization_level=1)
    finished = debug.get(qc).filter(EventType.TRANSPILE_FINISHED)[0]
    assert finished.circuit is not None
    assert finished.circuit == transpiled
    assert getattr(finished.circuit, "_qisdebug_id", None) is None
    size_before = finished.circuit.size()
    transpiled.x(0)
    assert finished.circuit.size() == size_before


def test_preset_pass_manager_run_is_recorded(debug):
    """Directly running a preset PassManager records the same telemetry."""
    debug.start()
    qc = _bell("pm-run")
    pm = generate_preset_pass_manager(optimization_level=1, basis_gates=["cx", "u"])
    out = pm.run(qc)
    timeline = debug.get(qc)
    assert timeline.filter(EventType.TRANSPILE_STARTED)
    assert timeline.filter(EventType.TRANSPILE_PASS)
    assert timeline.filter(EventType.TRANSPILE_FINISHED)
    assert getattr(out, "_qisdebug_id") == timeline.circuit_id


def test_transpiling_a_list_of_circuits_records_start_and_finish(debug):
    """Multi-circuit transpiles record boundaries for every circuit."""
    debug.start()
    circuits = [_bell(f"batch-{i}") for i in range(2)]
    transpile(circuits, basis_gates=["cx", "u"])
    for qc in circuits:
        timeline = debug.get(qc)
        assert timeline.filter(EventType.TRANSPILE_STARTED)
        assert timeline.filter(EventType.TRANSPILE_FINISHED)


def _dynamic(name="dynamic"):
    """Return a circuit with control flow, which Aer compiles internally."""
    qc = QuantumCircuit(2, 2, name=name)
    qc.h(0)
    qc.measure(0, 0)
    with qc.if_test((qc.clbits[0], 1)):
        qc.x(1)
    qc.measure(1, 1)
    return qc


def test_backend_internal_compilation_is_recorded_as_backend_phase(debug):
    """Aer's own control-flow compilation during run() lands as phase='backend'.

    Aer compiles on a worker thread where context-var suspension is not
    visible, so this exercises the suspended-circuit-id path.
    """
    debug.start()
    qc = _dynamic()
    AerSimulator().run(qc, shots=10).result()
    timeline = debug.get(qc)
    assert timeline.filter(EventType.EXECUTION_FINISHED)
    assert timeline.transpile_events
    assert all(e.phase == "backend" for e in timeline.transpile_events)
    passes = timeline.filter(EventType.TRANSPILE_PASS)
    assert any("jump" in p.ops or "mark" in p.ops for p in passes if p.ops)


def test_backend_compilation_capture_can_be_disabled(debug):
    """With capture off, backend-internal runs leave only a note event."""
    debug.configure(capture_backend_compilation=False)
    debug.start()
    qc = _dynamic("dynamic-off")
    AerSimulator().run(qc, shots=10).result()
    timeline = debug.get(qc)
    assert not timeline.transpile_events
    assert any("backend-internal" in n.message for n in timeline.filter(EventType.NOTE))


def test_directly_applied_pass_is_recorded(debug):
    """SomePass()(circuit) never enters PassManager.run but is still captured."""
    from qiskit.transpiler.passes import Optimize1qGatesDecomposition

    debug.start()
    qc = QuantumCircuit(1, name="standalone")
    qc.h(0)
    qc.h(0)
    out = Optimize1qGatesDecomposition(basis=["u"])(qc)
    timeline = debug.get(qc)
    passes = timeline.filter(EventType.TRANSPILE_PASS)
    assert len(passes) == 1
    assert passes[0].pass_name == "Optimize1qGatesDecomposition"
    assert passes[0].changed
    assert passes[0].circuit is not None
    assert getattr(out, "_qisdebug_id") == timeline.circuit_id


def test_hand_built_pass_manager_is_recorded(debug):
    """A user-constructed PassManager, not just presets, is instrumented."""
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import Depth, Optimize1qGatesDecomposition

    debug.start()
    qc = _bell("custom-pm")
    pm = PassManager([Depth(), Optimize1qGatesDecomposition()])
    pm.run(qc)
    timeline = debug.get(qc)
    assert timeline.filter(EventType.TRANSPILE_STARTED)
    passes = timeline.filter(EventType.TRANSPILE_PASS)
    assert {p.pass_name for p in passes} == {"Depth", "Optimize1qGatesDecomposition"}
    assert timeline.filter(EventType.TRANSPILE_FINISHED)


def test_property_set_diff_recorded_on_changed_pass(debug):
    """Layout/routing passes leave a property_changes diff against the previous pass."""
    debug.start()
    qc = _bell()
    transpile(qc, coupling_map=[[0, 1], [1, 0]], basis_gates=["cx", "u"])
    passes = debug.get(qc).filter(EventType.TRANSPILE_PASS)
    with_diff = [p for p in passes if p.property_changes]
    assert with_diff
    assert any(
        p.property_changes.added or p.property_changes.changed for p in with_diff
    )
    assert any(p.property_set is not None for p in passes)


def test_capture_property_set_none_disables_capture(debug):
    """capture_property_set='none' records passes without property_set data."""
    debug.configure(capture_property_set="none")
    debug.start()
    qc = _bell()
    transpile(qc, coupling_map=[[0, 1], [1, 0]], basis_gates=["cx", "u"])
    passes = debug.get(qc).filter(EventType.TRANSPILE_PASS)
    assert passes
    assert all(p.property_changes is None for p in passes)
    assert all(p.property_set is None for p in passes)


def test_capture_property_set_all_keeps_every_pass_snapshot(debug):
    """capture_property_set='all' snapshots every pass, changed or not."""
    debug.configure(capture_property_set="all")
    debug.start()
    qc = _bell()
    transpile(qc, coupling_map=[[0, 1], [1, 0]], basis_gates=["cx", "u"])
    passes = debug.get(qc).filter(EventType.TRANSPILE_PASS)
    assert passes
    assert all(p.property_set is not None for p in passes)
    assert all(p.property_changes is not None for p in passes)


def test_construction_of_transpiled_circuit_not_polluted(debug):
    """Internal circuits built during transpilation create no timelines."""
    debug.start()
    qc = _bell("clean")
    before = {t.circuit_id for t in debug.timelines()}
    transpile(qc, basis_gates=["cx", "u"], optimization_level=2)
    after = {t.circuit_id for t in debug.timelines()}
    assert after == before
