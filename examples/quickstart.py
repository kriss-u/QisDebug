"""QisDebug quickstart: trace a circuit from construction through execution.

Run with::

    python examples/quickstart.py

Note that after ``debug.start()`` the circuit-building, transpile and run
calls below are completely ordinary Qiskit code.
"""

import json

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

import qisdebug


def main() -> None:
    """Trace a GHZ circuit and print everything QisDebug captured."""
    debug = qisdebug.QisDebug()
    debug.start()

    qc = QuantumCircuit(3, 3, name="ghz")
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure(range(3), range(3))

    backend = AerSimulator()
    transpiled = transpile(qc, basis_gates=["cx", "u"], optimization_level=1)
    counts = backend.run(transpiled, shots=1000).result().get_counts()
    print(f"counts: {counts}\n")

    timeline = debug.get(qc)
    print(timeline.summary())

    print("\nfull event sequence:")
    for event in timeline.events:
        print(f"  [{event.seq:>3}] {event.type.value:<22} {event.describe()}")

    print("\nstate evolution:")
    for snapshot in timeline.snapshots:
        probabilities = {
            format(i, f"0{snapshot.num_qubits}b"): round(p, 3)
            for i, p in enumerate(snapshot.probabilities())
            if p > 1e-9
        }
        label = snapshot.gate or "initial"
        print(f"  step {snapshot.step} ({label:<7}) {probabilities}")

    exported = debug.export(qc)
    print(
        f"\nexport: {len(json.dumps(exported)) / 1024:.1f} KiB of JSON, "
        f"{len(exported[0]['events'])} events"
    )

    debug.show()

    debug.end()


if __name__ == "__main__":
    main()
