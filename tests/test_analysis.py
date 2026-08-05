"""Bug-finding logic: statevector/counts comparison and anomaly detection."""

import math

import pytest
from qiskit.circuit import QuantumCircuit
from qiskit_aer import AerSimulator

from qisdebug._analysis import (
    compare_counts,
    compare_statevector,
    expand_sparse_statevector,
    find_anomalies,
)
from qisdebug._events import EventType, StateSnapshotEvent


def _traced_bell(debug):
    debug.start()
    qc = QuantumCircuit(2, 2, name="bell")
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    AerSimulator().run(qc, shots=1000).result()
    return debug.get(qc)


def test_compare_statevector_matches_expected_bell_state(debug):
    timeline = _traced_bell(debug)
    final = timeline.snapshots[-1]
    expected = [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)]
    diff = compare_statevector(final, expected)
    assert diff.matched
    assert diff.fidelity == pytest.approx(1.0, abs=1e-6)
    assert not diff.mismatches


def test_compare_statevector_flags_wrong_state(debug):
    timeline = _traced_bell(debug)
    final = timeline.snapshots[-1]
    expected = [1.0, 0, 0, 0]  # |00> instead of the actual Bell state
    diff = compare_statevector(final, expected)
    assert not diff.matched
    assert diff.fidelity < 0.9
    assert diff.mismatches


def test_compare_statevector_rejects_wrong_length(debug):
    timeline = _traced_bell(debug)
    final = timeline.snapshots[-1]
    with pytest.raises(ValueError):
        compare_statevector(final, [1.0, 0.0])


def test_expand_sparse_statevector_fills_named_labels_and_zeros_the_rest():
    dense = expand_sparse_statevector(
        {"00": [1 / math.sqrt(2), 0], "11": [1 / math.sqrt(2), 0]}, num_qubits=2
    )
    assert dense == [1 / math.sqrt(2), 0j, 0j, 1 / math.sqrt(2)]


def test_expand_sparse_statevector_rejects_wrong_width_label():
    with pytest.raises(ValueError, match="2-character binary string"):
        expand_sparse_statevector({"1": [1.0, 0.0]}, num_qubits=2)


def test_compare_counts_matches_balanced_bell_outcomes(debug):
    timeline = _traced_bell(debug)
    finished = timeline.filter(EventType.EXECUTION_FINISHED)[0]
    assert finished.counts is not None
    diff = compare_counts(finished.counts, {"00": 0.5, "11": 0.5})
    assert diff.matched
    assert diff.total_variation_distance < 0.1


def test_compare_counts_flags_unexpected_distribution():
    diff = compare_counts({"00": 900, "11": 100}, {"00": 0.5, "11": 0.5})
    assert not diff.matched
    assert diff.total_variation_distance == pytest.approx(0.4, abs=1e-6)
    assert {d.outcome for d in diff.deltas} == {"00", "11"}


def test_find_anomalies_reports_failed_execution(debug):
    timeline = _traced_bell(debug)
    finished = timeline.filter(EventType.EXECUTION_FINISHED)[0]
    finished.success = False
    anomalies = find_anomalies(timeline)
    assert any("execution failed" in a.message for a in anomalies)


def test_find_anomalies_flags_non_normalized_statevector(debug):
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
    timeline = _traced_bell(debug)
    anomalies = find_anomalies(timeline)
    assert anomalies == []
