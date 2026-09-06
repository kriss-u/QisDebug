"""QisDebug on a buggy circuit: a 3-qubit phase estimation that gives the
wrong answer, traced end to end with a timeline.

Run with::

    python examples/phase_estimation_bug.py

The circuit estimates a phase of theta = pi/2, which should peak the
measurement distribution at eval register value 010 (2/8). Instead it comes
out wrong, because the controlled rotations below use ``crz`` where the
algorithm calls for ``cp``: ``crz`` puts an extra relative phase on the
*control* qubit that a controlled phase gate does not, so the inverse QFT
decodes a shifted phase.

Rather than diffing against a hand-written fix, this uses ``debug.get()``
to pull the timeline back out and inspect the exact controlled gates that
were applied, which is enough to see the bug without already knowing the
answer.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from _qpe import inverse_qft_and_measure, prepare_eigenstate_and_superposition, qpe_registers

import qisdebug


def phase_estimation(theta: float) -> QuantumCircuit:
    """3-qubit QPE circuit for an eigenphase of ``theta``."""
    eval_reg, target_reg, cr = qpe_registers()
    qc = QuantumCircuit(eval_reg, target_reg, cr, name="qpe")
    prepare_eigenstate_and_superposition(qc, eval_reg, target_reg)

    qc.crz(4 * theta, eval_reg[0], target_reg[0])
    qc.crz(2 * theta, eval_reg[1], target_reg[0])
    qc.crz(1 * theta, eval_reg[2], target_reg[0])

    inverse_qft_and_measure(qc, eval_reg, cr)
    return qc


def main() -> None:
    """Run the buggy QPE circuit and inspect its timeline for the ``crz``
    gates responsible for the wrong measurement peak."""
    debug = qisdebug.QisDebug()
    debug.start()

    qc = phase_estimation(theta=np.pi / 2)
    counts = AerSimulator().run(qc, shots=10000).result().get_counts()
    top = max(counts, key=counts.get)
    print(f"counts: {counts}")
    print(f"expected peak at 010, got {top!r} instead\n")

    timeline = debug.get(qc)
    print(timeline.summary())

    print("\ncontrolled rotations applied to the target qubit:")
    for event in timeline.events:
        if event.type.value == "gate_added" and event.describe().startswith("crz"):
            print(f"  [{event.seq:>3}] {event.describe()}")

    debug.show(qc)

    debug.end()


if __name__ == "__main__":
    main()
