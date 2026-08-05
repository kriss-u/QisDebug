"""Telemetry store behavior: eviction, clearing, suspension, ordering."""

import json

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from qiskit.circuit.quantumcircuit import QuantumCircuit

from qisdebug._store import TelemetryStore


def test_eviction_keeps_only_most_recent_circuits(debug):
    """Once the cap is hit, the oldest timelines are evicted first."""
    debug.configure(max_circuits=5)
    debug.start()
    for i in range(8):
        qc = QuantumCircuit(1, name=f"loop-{i}")
        qc.h(0)
    names = {t.name for t in debug.timelines()}
    assert len(names) == 5
    assert names == {f"loop-{i}" for i in range(3, 8)}


def test_clear_drops_all_timelines(debug):
    """clear() empties the store without deactivating the hooks."""
    debug.start()
    qc = QuantumCircuit(1)
    qc.h(0)
    assert debug.timelines()
    debug.clear()
    assert debug.timelines() == []
    import pytest
    import qisdebug

    with pytest.raises(qisdebug.TimelineNotFoundError):
        debug.get(qc)


def test_suspension_blocks_recording(debug):
    """Appends made under store.suspended() leave no telemetry behind."""
    debug.start()
    qc = QuantumCircuit(1)
    with debug.store.suspended():
        qc.h(0)
    import pytest
    import qisdebug

    with pytest.raises(qisdebug.TimelineNotFoundError):
        debug.get(qc)
    qc.x(0)
    assert [e.gate for e in debug.get(qc).gate_events] == ["x"]


def test_timeline_lookup_by_object_and_id(debug):
    """Timelines resolve by circuit object, id string, and latest-touched."""
    debug.start()
    qc = QuantumCircuit(1)
    qc.h(0)
    by_object = debug.get(qc)
    assert by_object is debug.get(by_object.circuit_id)
    assert by_object is debug.store.latest()


def test_summary_and_export_are_serializable(debug):
    """summary() mentions the circuit and export() survives json.dumps."""
    debug.start()
    qc = QuantumCircuit(2, 2, name="bell")
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    timeline = debug.get(qc)
    assert "bell" in timeline.summary()
    assert "gate_added" in timeline.summary()
    payload = timeline.export()
    text = json.dumps(payload)
    assert "gate_added" in text
    assert len(payload["events"]) == len(timeline.events)


def test_circuit_at_rebuilds_partial_circuit(debug):
    """circuit_at(n) reconstructs the circuit after its first n gates."""
    debug.start()
    qc = QuantumCircuit(2, name="rebuild")
    qc.h(0)
    qc.cx(0, 1)
    partial = debug.get(qc).circuit_at(1)
    assert partial is not None
    assert [ci.operation.name for ci in partial.data] == ["h"]
    full = debug.get(qc).circuit_at()
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
    store = TelemetryStore()
    for i in range(total):
        qc = QuantumCircuit(1, name=f"c{i}")
        store.timeline_for(qc, max_circuits=cap)
    assert len(store.all_timelines()) == min(total, cap)
    sequences = [e.seq for t in store.all_timelines() for e in t.events]
    assert len(set(sequences)) == len(sequences)
