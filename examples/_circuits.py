"""Small canonical circuits shared between examples (``quickstart.py`` and
``multi_circuit.py``)."""

from qiskit import QuantumCircuit


def ghz_circuit(name: str = "ghz") -> QuantumCircuit:
    """Three-qubit GHZ state, measured."""
    qc = QuantumCircuit(3, 3, name=name)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure(range(3), range(3))
    return qc
