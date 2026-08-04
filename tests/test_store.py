"""Telemetry store behavior: eviction, clearing, suspension, ordering."""

import json

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from qiskit.circuit.quantumcircuit import QuantumCircuit

import qisdebug
from conftest import get_timeline
from qisdebug._store import STORE


def test_eviction_keeps_only_most_recent_circuits():
    """Once the cap is hit, the oldest timelines are evicted first."""
    qisdebug.configure(max_circuits=5)
    qisdebug.activate()
    for i in range(8):
        qc = QuantumCircuit(1, name=f"loop-{i}")
        qc.h(0)
    names = {t.name for t in qisdebug.timelines()}
    assert len(names) == 5
    assert names == {f"loop-{i}" for i in range(3, 8)}


def test_clear_drops_all_timelines():
    """clear() empties the store without deactivating the hooks."""
    qisdebug.activate()
    qc = QuantumCircuit(1)
    qc.h(0)
    assert qisdebug.timelines()
    qisdebug.clear()
    assert qisdebug.timelines() == []
    assert qisdebug.timeline(qc) is None


def test_suspension_blocks_recording():
    """Appends made under STORE.suspended() leave no telemetry behind."""
    qisdebug.activate()
    qc = QuantumCircuit(1)
    with STORE.suspended():
        qc.h(0)
    assert qisdebug.timeline(qc) is None
    qc.x(0)
    assert [e.gate for e in get_timeline(qc).gate_events] == ["x"]


def test_timeline_lookup_by_object_and_id():
    """Timelines resolve by circuit object, id string, and latest-touched."""
    qisdebug.activate()
    qc = QuantumCircuit(1)
    qc.h(0)
    by_object = get_timeline(qc)
    assert by_object is get_timeline(by_object.circuit_id)
    assert qisdebug.timeline() is by_object


def test_summary_and_export_are_serializable():
    """summary() mentions the circuit and export() survives json.dumps."""
    qisdebug.activate()
    qc = QuantumCircuit(2, 2, name="bell")
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    timeline = get_timeline(qc)
    assert "bell" in timeline.summary()
    assert "gate_added" in timeline.summary()
    payload = timeline.export()
    text = json.dumps(payload)
    assert "gate_added" in text
    assert len(payload["events"]) == len(timeline.events)


def test_circuit_at_rebuilds_partial_circuit():
    """circuit_at(n) reconstructs the circuit after its first n gates."""
    qisdebug.activate()
    qc = QuantumCircuit(2, name="rebuild")
    qc.h(0)
    qc.cx(0, 1)
    partial = get_timeline(qc).circuit_at(1)
    assert partial is not None
    assert [ci.operation.name for ci in partial.data] == ["h"]
    full = get_timeline(qc).circuit_at()
    assert full is not None
    assert [ci.operation.name for ci in full.data] == ["h", "cx"]


@settings(
    deadline=None,
    max_examples=25,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    total=st.integers(min_value=1, max_value=20),
    cap=st.integers(min_value=1, max_value=10),
)
def test_eviction_never_exceeds_cap(total, cap):
    """For any cap and insert count, the store never grows past the cap."""
    qisdebug.deactivate()
    qisdebug.clear()
    qisdebug.configure(max_circuits=cap)
    qisdebug.activate()
    try:
        for i in range(total):
            qc = QuantumCircuit(1, name=f"c{i}")
            qc.x(0)
        assert len(qisdebug.timelines()) == min(total, cap)
        sequences = [e.seq for t in qisdebug.timelines() for e in t.events]
        assert len(set(sequences)) == len(sequences)
    finally:
        qisdebug.deactivate()
        qisdebug.clear()
