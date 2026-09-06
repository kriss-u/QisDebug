"""Shared 3-qubit phase-estimation circuit-building blocks used by
``bug_1.py``, ``bug_1_fixed.py``, and ``phase_estimation_bug.py`` -- the
controlled-rotation step (``crz`` vs. ``cp``) is each script's own point of
difference, so only that step is left out of here.
"""

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister


def qpe_registers() -> tuple[QuantumRegister, QuantumRegister, ClassicalRegister]:
    """3-qubit eval register, 1-qubit target register, and its classical register."""
    return (
        QuantumRegister(3, name="eval"),
        QuantumRegister(1, name="target"),
        ClassicalRegister(3, name="meas"),
    )


def prepare_eigenstate_and_superposition(
    qc: QuantumCircuit, eval_reg: QuantumRegister, target_reg: QuantumRegister
) -> None:
    """Set the target qubit to the |1> eigenstate and put the eval register
    into an equal superposition."""
    qc.x(target_reg[0])
    qc.h(eval_reg[0])
    qc.h(eval_reg[1])
    qc.h(eval_reg[2])


def inverse_qft_and_measure(
    qc: QuantumCircuit, eval_reg: QuantumRegister, cr: ClassicalRegister
) -> None:
    """Append the 3-qubit inverse QFT to the eval register and measure it."""
    qc.swap(eval_reg[0], eval_reg[2])
    qc.h(eval_reg[0])
    qc.cp(-np.pi / 2, eval_reg[0], eval_reg[1])
    qc.h(eval_reg[1])
    qc.cp(-np.pi / 4, eval_reg[0], eval_reg[2])
    qc.cp(-np.pi / 2, eval_reg[1], eval_reg[2])
    qc.h(eval_reg[2])
    qc.measure(eval_reg, cr)
