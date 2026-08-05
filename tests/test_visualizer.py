"""Visualizer: decoupled from instrumentation."""

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit_aer import AerSimulator

from qisdebug import visualizer


def _traced_bell(debug):
    """Run a bell circuit under instrumentation and return its timeline."""
    debug.start()
    qc = QuantumCircuit(2, 2, name="viz")
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    AerSimulator().run(qc, shots=10).result()
    return debug.get(qc)


def test_format_statevector_renders_amplitudes(debug):
    """Snapshot formatting shows basis states with probabilities."""
    timeline = _traced_bell(debug)
    text = visualizer.format_statevector(timeline.snapshots[-1])
    assert "|00>" in text
    assert "p=0.5000" in text
