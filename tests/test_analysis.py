"""Bug-finding logic: statevector/counts comparison and anomaly detection."""

import math

import pytest
from _helpers import traced_bell
from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit_aer import AerSimulator

from qisdebug._analysis import (
    compare_counts,
    compare_statevector,
    diff_timelines,
    expand_sparse_statevector,
    find_anomalies,
)
from qisdebug._events import EventType, StateSnapshotEvent


def _traced_bell(debug):
    """Run a bell circuit with a fixed seed, for deterministic count comparisons."""
    return traced_bell(debug, name="bell", shots=20000, seed_simulator=1234)


def test_compare_statevector_matches_expected_bell_state(debug):
    """A statevector matching the expected amplitudes reports full fidelity."""
    timeline = _traced_bell(debug)
    final = timeline.snapshots[-1]
    expected = [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)]
    diff = compare_statevector(final, expected)
    assert diff.matched
    assert diff.fidelity == pytest.approx(1.0, abs=1e-6)
    assert not diff.mismatches


def test_compare_statevector_flags_wrong_state(debug):
    """A statevector far from expected reports low fidelity and mismatches."""
    timeline = _traced_bell(debug)
    final = timeline.snapshots[-1]
    expected = [1.0, 0, 0, 0]  # |00> instead of the actual Bell state
    diff = compare_statevector(final, expected)
    assert not diff.matched
    assert diff.fidelity < 0.9
    assert diff.mismatches


def test_compare_statevector_rejects_wrong_length(debug):
    """Comparing against a mismatched-length expected vector is a ValueError."""
    timeline = _traced_bell(debug)
    final = timeline.snapshots[-1]
    with pytest.raises(ValueError):
        compare_statevector(final, [1.0, 0.0])


def test_expand_sparse_statevector_fills_named_labels_and_zeros_the_rest():
    """Basis states missing from the sparse mapping expand to zero amplitude."""
    dense = expand_sparse_statevector(
        {"00": [1 / math.sqrt(2), 0], "11": [1 / math.sqrt(2), 0]}, num_qubits=2
    )
    assert dense == [1 / math.sqrt(2), 0j, 0j, 1 / math.sqrt(2)]


def test_expand_sparse_statevector_rejects_wrong_width_label():
    """A basis label with the wrong bit width is a clear ValueError."""
    with pytest.raises(ValueError, match="2-character binary string"):
        expand_sparse_statevector({"1": [1.0, 0.0]}, num_qubits=2)


def test_compare_counts_matches_balanced_bell_outcomes(debug):
    """Measured bell counts match the expected 50/50 distribution."""
    timeline = _traced_bell(debug)
    finished = timeline.filter(EventType.EXECUTION_FINISHED)[0]
    assert finished.counts is not None
    diff = compare_counts(finished.counts, {"00": 0.5, "11": 0.5})
    assert diff.matched
    assert diff.total_variation_distance < 0.1


def test_compare_counts_flags_unexpected_distribution():
    """A skewed distribution reports a mismatch with nonzero TVD."""
    diff = compare_counts({"00": 900, "11": 100}, {"00": 0.5, "11": 0.5})
    assert not diff.matched
    assert diff.total_variation_distance == pytest.approx(0.4, abs=1e-6)
    assert {d.outcome for d in diff.deltas} == {"00", "11"}


def test_find_anomalies_reports_failed_execution(debug):
    """A failed execution event is flagged as an anomaly."""
    timeline = _traced_bell(debug)
    finished = timeline.filter(EventType.EXECUTION_FINISHED)[0]
    finished.success = False
    anomalies = find_anomalies(timeline)
    assert any("execution failed" in a.message for a in anomalies)


def test_find_anomalies_flags_non_normalized_statevector(debug):
    """A statevector whose norm isn't 1 is flagged as an anomaly."""
    timeline = _traced_bell(debug)
    snapshot = timeline.snapshots[-1]
    tampered = StateSnapshotEvent(
        seq=snapshot.seq,
        step=snapshot.step,
        gate_index=snapshot.gate_index,
        gate=snapshot.gate,
        num_qubits=snapshot.num_qubits,
        statevector=[2.0, 0, 0, 0],
    )
    idx = timeline.events.index(snapshot)
    timeline.events[idx] = tampered
    anomalies = find_anomalies(timeline)
    assert any("norm is" in a.message for a in anomalies)


def test_find_anomalies_clean_bell_circuit_has_no_findings(debug):
    """A normal, successful bell run has nothing to flag."""
    timeline = _traced_bell(debug)
    anomalies = find_anomalies(timeline)
    assert not anomalies


def _traced(debug, name, build):
    """Build a circuit with ``build``, run it, and return its timeline."""
    debug.start()
    qc = QuantumCircuit(2, 2, name=name)
    build(qc)
    AerSimulator().run(qc, shots=20000, seed_simulator=1234).result()
    return debug.get(qc)


def _build_bell(qc):
    """Add a standard bell-pair circuit body to ``qc`` in place."""
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])


def test_diff_timelines_identical_circuits_are_all_equal(debug):
    """Two runs of the same circuit diff as entirely equal gates."""
    timeline_a = _traced(debug, "bell_a", _build_bell)
    timeline_b = _traced(debug, "bell_b", _build_bell)

    diff = diff_timelines(timeline_a, timeline_b)

    assert [entry.op for entry in diff.gate_diff] == ["equal"] * len(
        timeline_a.gate_events
    )
    assert diff.statevector_diff is not None
    assert diff.statevector_diff.fidelity == pytest.approx(1.0, abs=1e-6)
    assert diff.counts_diff is not None
    assert diff.counts_diff.matched


def test_diff_timelines_detects_inserted_gate(debug):
    """An extra gate in one circuit shows up as an "insert" diff entry."""
    timeline_a = _traced(debug, "bell_plain", _build_bell)

    def _build_with_extra_x(qc):
        qc.h(0)
        qc.x(1)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])

    timeline_b = _traced(debug, "bell_extra_x", _build_with_extra_x)

    diff = diff_timelines(timeline_a, timeline_b)

    ops = [entry.op for entry in diff.gate_diff]
    assert "insert" in ops
    inserted = next(entry for entry in diff.gate_diff if entry.op == "insert")
    assert inserted.b is not None
    assert inserted.b.gate == "x"
    assert diff.statevector_diff is not None
    assert not diff.statevector_diff.matched


def test_diff_timelines_reports_depth_and_size_from_last_transpile(debug):
    """Depth/size deltas come from each timeline's most recent transpile event."""
    debug.start()
    qc_a = QuantumCircuit(2, name="shape_a")
    qc_a.h(0)
    qc_a.cx(0, 1)
    transpile(qc_a, basis_gates=["cx", "u"], optimization_level=1)

    qc_b = QuantumCircuit(2, name="shape_b")
    qc_b.h(0)
    qc_b.cx(0, 1)
    qc_b.cx(0, 1)
    transpile(qc_b, basis_gates=["cx", "u"], optimization_level=1)

    diff = diff_timelines(debug.get(qc_a), debug.get(qc_b))

    assert diff.depth_a is not None
    assert diff.depth_b is not None
    assert diff.size_a is not None
    assert diff.size_b is not None


def test_diff_timelines_no_snapshots_or_counts_reports_none():
    """Timelines with no events, snapshots, or counts diff to all-None extras."""
    from qisdebug._events import Timeline

    timeline_a = Timeline(circuit_id="a", name="a")
    timeline_b = Timeline(circuit_id="b", name="b")

    diff = diff_timelines(timeline_a, timeline_b)

    assert not diff.gate_diff
    assert diff.statevector_diff is None
    assert diff.counts_diff is None
    assert diff.depth_a is None
    assert diff.depth_b is None


def test_timeline_diff_summary_line_and_describe_are_readable(debug):
    """summary_line() and describe() mention fidelity and the circuit names."""
    timeline_a = _traced(debug, "bell_a2", _build_bell)
    timeline_b = _traced(debug, "bell_b2", _build_bell)

    diff = diff_timelines(timeline_a, timeline_b)

    assert "equal" in diff.summary_line()
    assert "fidelity" in diff.summary_line()
    assert "bell_a2" in diff.describe()
    assert "bell_b2" in diff.describe()


def test_diff_timelines_seq_bounds_diff_two_moments_of_one_circuit(debug):
    """seq_a/seq_b bound a diff to two moments within a single circuit's history."""
    debug.start()
    qc = QuantumCircuit(2, name="replay")
    qc.h(0)
    seq_after_h = debug.get(qc).events[-1].seq
    qc.cx(0, 1)
    seq_after_cx = debug.get(qc).events[-1].seq

    diff = diff_timelines(
        debug.get(qc), debug.get(qc), seq_a=seq_after_h, seq_b=seq_after_cx
    )

    assert diff.seq_a == seq_after_h
    assert diff.seq_b == seq_after_cx
    assert [entry.op for entry in diff.gate_diff] == ["equal", "insert"]
    inserted = next(entry for entry in diff.gate_diff if entry.op == "insert")
    assert inserted.b is not None
    assert inserted.b.gate == "cx"


def test_diff_timelines_seq_bounds_use_snapshot_and_counts_upto_seq(debug):
    """seq_a bounds which snapshot is used, not just which gates are diffed."""
    timeline = _traced(debug, "bell_replay", _build_bell)
    # snapshots[0] is the initial |00> state, snapshots[1] the state right
    # after h(0) -- both captured during execution, strictly after every
    # gate_added event's own seq.
    seq_after_h = timeline.snapshots[1].seq

    diff = diff_timelines(timeline, timeline, seq_a=seq_after_h, seq_b=None)

    # As of just the h gate, the state is a superposition, not the final
    # bell state -- diffing it against the (identical) timeline's final
    # state must show a mismatch, proving snapshot_upto bounded the "a" side.
    assert diff.statevector_diff is not None
    assert not diff.statevector_diff.matched


def test_diff_timelines_matching_explicit_seqs_across_two_circuits(debug):
    """Explicit seq_a/seq_b bound each circuit independently before diffing."""
    timeline_a = _traced(debug, "bell_seq_a", _build_bell)
    timeline_b = _traced(debug, "bell_seq_b", _build_bell)
    seq_a = timeline_a.gate_events[0].seq
    seq_b = timeline_b.gate_events[0].seq

    diff = diff_timelines(timeline_a, timeline_b, seq_a=seq_a, seq_b=seq_b)

    assert diff.gate_diff == [entry for entry in diff.gate_diff if entry.op == "equal"]
    assert len(diff.gate_diff) == 1


def test_diff_timelines_different_qubit_counts_reports_mismatch_not_crash(debug):
    """Diffing two stages whose statevector snapshots have different qubit
    counts is a legitimate "these differ" outcome, not a crash -- unlike
    compare_statevector's own hard error for a caller-supplied expected
    vector of the wrong shape (see test_compare_statevector_rejects_wrong_length),
    diffing two circuits that simply have different widths is exactly the
    kind of difference a diff should report, not raise on."""
    debug.start()
    qc_a = QuantumCircuit(2, name="two_qubit")
    qc_a.h(0)
    qc_a.cx(0, 1)
    AerSimulator().run(qc_a, shots=10).result()

    qc_b = QuantumCircuit(3, name="three_qubit")
    qc_b.h(0)
    qc_b.h(1)
    qc_b.cx(1, 2)
    AerSimulator().run(qc_b, shots=10).result()
    debug.end()

    timeline_a = debug.get(qc_a)
    timeline_b = debug.get(qc_b)

    diff = diff_timelines(timeline_a, timeline_b)

    assert diff.statevector_diff is not None
    assert diff.statevector_diff.matched is False
    assert diff.statevector_diff.fidelity == 0.0
    assert diff.statevector_diff.note is not None
    assert "2" in diff.statevector_diff.note and "3" in diff.statevector_diff.note
    # describe() must not raise either -- this is what render_state_diff
    # (the TUI/CLI diff screen's own text) calls.
    assert "MISMATCH" in diff.statevector_diff.describe()
