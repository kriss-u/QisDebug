"""Buggy 3-qubit quantum phase estimation: uses ``crz`` for the
controlled-eigenvalue kickback where a controlled-phase gate (``cp``) is
required, producing skewed measurement counts. See ``bug_1_fixed.py`` for
the corrected version and ``phase_estimation_bug.py`` for the same bug in a
more general, parametrized form.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from _qpe import inverse_qft_and_measure, prepare_eigenstate_and_superposition, qpe_registers

eval_reg, target_reg, cr = qpe_registers()
qc = QuantumCircuit(eval_reg, target_reg, cr)
prepare_eigenstate_and_superposition(qc, eval_reg, target_reg)

theta = np.pi / 2

qc.crz(4 * theta, eval_reg[0], target_reg[0])
qc.crz(2 * theta, eval_reg[1], target_reg[0])
qc.crz(1 * theta, eval_reg[2], target_reg[0])

inverse_qft_and_measure(qc, eval_reg, cr)

sim = AerSimulator()
result = sim.run(qc, shots=10000).result()
print(result.get_counts())
