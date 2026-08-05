"""QisDebug with multiple circuits: each QuantumCircuit gets its own timeline.

Run with::

    python examples/multi_circuit.py

QisDebug tracks every circuit touched while it is active, keyed off a hidden id
attached to the circuit instance. This example builds and runs several
unrelated circuits, shows how to list and inspect all of their timelines, and
opens a single viewer with all of their events merged into one chronological
timeline.
"""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

import qisdebug


def bell() -> QuantumCircuit:
    """Two-qubit Bell pair."""
    qc = QuantumCircuit(2, 2, name="bell")
    qc.h(0)
    qc.cx(0, 1)
    qc.measure(range(2), range(2))
    return qc


def ghz() -> QuantumCircuit:
    """Three-qubit GHZ state."""
    qc = QuantumCircuit(3, 3, name="ghz")
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure(range(3), range(3))
    return qc


def qft(num_qubits: int = 3) -> QuantumCircuit:
    """Textbook quantum Fourier transform, uncomputed by measuring in order."""
    qc = QuantumCircuit(num_qubits, num_qubits, name="qft")
    for target in range(num_qubits):
        qc.h(target)
        for control in range(target + 1, num_qubits):
            qc.cp(3.14159 / 2 ** (control - target), control, target)
    qc.measure(range(num_qubits), range(num_qubits))
    return qc


def main() -> None:
    """Trace three independent circuits and report on each of their timelines."""
    debug = qisdebug.QisDebug()
    debug.start()

    circuits = [bell(), ghz(), qft()]
    backend = AerSimulator()
    for qc in circuits:
        transpiled = transpile(qc, basis_gates=["cx", "u"], optimization_level=1)
        counts = backend.run(transpiled, shots=1000).result().get_counts()
        print(f"{qc.name}: {counts}")

    print(f"\n{len(debug.timelines())} timelines tracked:")
    for timeline in debug.timelines():
        print(
            f"  - {timeline.name:<6} {len(timeline.events)} events, id={timeline.circuit_id}"
        )

    print("\nper-circuit summaries:")
    for qc in circuits:
        print(f"\n{debug.get(qc).summary()}")

    debug.show(circuits)

    debug.end()


if __name__ == "__main__":
    main()
