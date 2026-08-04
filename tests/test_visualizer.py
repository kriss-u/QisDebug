"""Visualizer: decoupled from instrumentation."""

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit_aer import AerSimulator

import qdebug
from conftest import get_timeline
from qdebug import visualizer


def _traced_bell():
    """Run a bell circuit under instrumentation and return its timeline."""
    qdebug.activate()
    qc = QuantumCircuit(2, 2, name="viz")
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    AerSimulator().run(qc, shots=10).result()
    return get_timeline(qc)


def test_format_statevector_renders_amplitudes():
    """Snapshot formatting shows basis states with probabilities."""
    timeline = _traced_bell()
    text = visualizer.format_statevector(timeline.snapshots[-1])
    assert "|00>" in text
    assert "p=0.5000" in text
