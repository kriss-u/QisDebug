"""Execution hook: statevector capture on Aer, result stripping, primitives."""

import numpy as np
import pytest
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit_aer import AerSimulator

import qisdebug
from conftest import get_timeline
from qisdebug import EventType


def _bell(name="bell"):
    """Return a 2-qubit bell circuit with measurements."""
    qc = QuantumCircuit(2, 2, name=name)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def test_backend_run_records_and_captures_statevectors():
    """An Aer run records submission, snapshots per gate, and completion."""
    qisdebug.activate()
    qc = _bell()
    result = AerSimulator().run(qc, shots=200).result()

    counts = result.get_counts()
    assert set(counts) <= {"00", "11"}
    assert sum(counts.values()) == 200

    timeline = get_timeline(qc)
    submitted = timeline.filter(EventType.EXECUTION_SUBMITTED)
    assert len(submitted) == 1
    assert submitted[0].backend == "aer_simulator"
    assert submitted[0].shots == 200
    assert submitted[0].instrumented

    snapshots = timeline.snapshots
    assert [s.step for s in snapshots] == [0, 1, 2]
    assert [s.gate for s in snapshots] == [None, "h", "cx"]
    np.testing.assert_allclose(snapshots[0].amplitudes(), [1, 0, 0, 0], atol=1e-9)
    inv_sqrt2 = 1 / np.sqrt(2)
    np.testing.assert_allclose(snapshots[1].amplitudes(), [inv_sqrt2, inv_sqrt2, 0, 0], atol=1e-9)
    np.testing.assert_allclose(snapshots[2].amplitudes(), [inv_sqrt2, 0, 0, inv_sqrt2], atol=1e-9)

    finished = timeline.filter(EventType.EXECUTION_FINISHED)
    assert len(finished) == 1
    assert finished[0].success
    assert finished[0].counts == counts
    assert finished[0].duration is not None and finished[0].duration >= 0


def test_snapshot_data_is_stripped_from_user_result():
    """The user-visible result contains no qisdebug snapshot entries."""
    qisdebug.activate()
    result = AerSimulator().run(_bell(), shots=50).result()
    assert set(result.data(0).keys()) == {"counts"}


def test_basic_simulator_run_records_execution_events():
    """The non-Aer BasicSimulator gets timing telemetry, minus snapshots."""
    from qiskit.providers.basic_provider import BasicSimulator

    qisdebug.activate()
    qc = _bell("basic")
    counts = BasicSimulator().run(qc, shots=50).result().get_counts()
    assert sum(counts.values()) == 50
    timeline = get_timeline(qc)
    submitted = timeline.filter(EventType.EXECUTION_SUBMITTED)
    assert len(submitted) == 1
    assert not submitted[0].instrumented
    finished = timeline.filter(EventType.EXECUTION_FINISHED)
    assert len(finished) == 1
    assert finished[0].counts == counts
    assert not timeline.snapshots


def test_backend_sampler_v2_records_execution_events():
    """BackendSamplerV2 pubs are tracked like the other primitives."""
    from qiskit.primitives import BackendSamplerV2

    qisdebug.activate()
    qc = _bell("backend-sampler")
    sampler = BackendSamplerV2(backend=AerSimulator())
    sampler.run([qc], shots=64).result()
    timeline = get_timeline(qc)
    submitted = timeline.filter(EventType.EXECUTION_SUBMITTED)
    assert any(s.backend == "BackendSamplerV2" for s in submitted)
    assert timeline.filter(EventType.EXECUTION_FINISHED)


def test_noise_model_construction_does_not_flood_the_store():
    """from_backend builds many internal circuits; none may become timelines.

    Without suspension around those factories, the internal noise circuits
    would evict the user's real timelines from the LRU store.
    """
    from qiskit.providers.fake_provider import GenericBackendV2

    qisdebug.activate()
    qc = _bell("kept")
    before = {t.circuit_id for t in qisdebug.timelines()}
    AerSimulator.from_backend(GenericBackendV2(num_qubits=4, seed=11))
    after = {t.circuit_id for t in qisdebug.timelines()}
    assert after == before
    assert len(get_timeline(qc).gate_events) == 4


def test_result_called_twice_records_once():
    """Repeated job.result() calls do not duplicate telemetry."""
    qisdebug.activate()
    qc = _bell()
    job = AerSimulator().run(qc, shots=50)
    first = job.result()
    second = job.result()
    assert first is second
    assert len(get_timeline(qc).filter(EventType.EXECUTION_FINISHED)) == 1


def test_circuit_without_measurements_still_snapshots():
    """Snapshot capture works when the circuit produces no counts."""
    qisdebug.activate()
    qc = QuantumCircuit(1, name="nomeasure")
    qc.h(0)
    AerSimulator().run(qc, shots=1).result()
    timeline = get_timeline(qc)
    assert [s.step for s in timeline.snapshots] == [0, 1]
    finished = timeline.filter(EventType.EXECUTION_FINISHED)
    assert len(finished) == 1
    assert finished[0].counts is None


def test_wide_circuit_skips_capture_with_note():
    """Circuits above max_statevector_qubits run normally, without capture."""
    qisdebug.configure(max_statevector_qubits=2)
    qisdebug.activate()
    qc = QuantumCircuit(3, 3, name="wide")
    qc.h(0)
    qc.measure(range(3), range(3))
    result = AerSimulator().run(qc, shots=25).result()
    assert sum(result.get_counts().values()) == 25
    timeline = get_timeline(qc)
    assert timeline.snapshots == []
    notes = timeline.filter(EventType.NOTE)
    assert any("max_statevector_qubits" in n.message for n in notes)
    assert len(timeline.filter(EventType.EXECUTION_FINISHED)) == 1


def test_mid_circuit_measurement_skips_capture():
    """Nondeterministic circuits skip snapshots but still run and record."""
    qisdebug.activate()
    qc = QuantumCircuit(2, 2, name="midmeasure")
    qc.h(0)
    qc.measure(0, 0)
    qc.x(1)
    qc.measure(1, 1)
    result = AerSimulator().run(qc, shots=30).result()
    assert sum(result.get_counts().values()) == 30
    timeline = get_timeline(qc)
    assert timeline.snapshots == []
    assert any("mid-circuit" in n.message for n in timeline.filter(EventType.NOTE))


def test_run_list_of_circuits():
    """Multi-circuit runs record independent telemetry per circuit."""
    qisdebug.activate()
    first = _bell("multi-a")
    second = _bell("multi-b")
    result = AerSimulator().run([first, second], shots=40).result()
    assert sum(result.get_counts(0).values()) == 40
    assert sum(result.get_counts(1).values()) == 40
    for qc in (first, second):
        timeline = get_timeline(qc)
        assert len(timeline.snapshots) == 3
        assert len(timeline.filter(EventType.EXECUTION_FINISHED)) == 1


def test_statevector_sampler_records_events():
    """Primitive runs record submission and completion telemetry."""
    qisdebug.activate()
    qc = QuantumCircuit(1, 1, name="prim")
    qc.h(0)
    qc.measure(0, 0)
    job = StatevectorSampler().run([qc], shots=64)
    job.result()
    timeline = get_timeline(qc)
    submitted = timeline.filter(EventType.EXECUTION_SUBMITTED)
    finished = timeline.filter(EventType.EXECUTION_FINISHED)
    assert any(e.backend == "StatevectorSampler" for e in submitted)
    assert any(e.backend == "StatevectorSampler" for e in finished)
    assert submitted[0].shots == 64


def test_aer_sampler_records_events():
    """The Aer SamplerV2 primitive is instrumented when available."""
    aer_primitives = pytest.importorskip("qiskit_aer.primitives")
    qisdebug.activate()
    qc = QuantumCircuit(1, 1, name="aerprim")
    qc.h(0)
    qc.measure(0, 0)
    job = aer_primitives.SamplerV2().run([qc], shots=32)
    job.result()
    timeline = get_timeline(qc)
    assert any(
        e.backend == "SamplerV2" for e in timeline.filter(EventType.EXECUTION_SUBMITTED)
    )
    assert timeline.filter(EventType.EXECUTION_FINISHED)


def test_capture_can_be_disabled():
    """With capture_statevectors=False no instrumented copy is built."""
    qisdebug.configure(capture_statevectors=False)
    qisdebug.activate()
    qc = _bell("nocapture")
    result = AerSimulator().run(qc, shots=20).result()
    assert sum(result.get_counts().values()) == 20
    timeline = get_timeline(qc)
    assert timeline.snapshots == []
    assert not timeline.filter(EventType.EXECUTION_SUBMITTED)[0].instrumented
