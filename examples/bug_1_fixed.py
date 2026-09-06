"""Corrected version of ``bug_1.py``: uses controlled-phase (``cp``) gates
for the eigenvalue kickback instead of ``crz``, producing the expected
measurement counts for 3-qubit quantum phase estimation.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from _qpe import inverse_qft_and_measure, prepare_eigenstate_and_superposition, qpe_registers

eval_reg, target_reg, cr = qpe_registers()
qc = QuantumCircuit(eval_reg, target_reg, cr)
prepare_eigenstate_and_superposition(qc, eval_reg, target_reg)

theta = np.pi / 2

qc.cp(1 * theta, eval_reg[0], target_reg[0])
qc.cp(2 * theta, eval_reg[1], target_reg[0])
qc.cp(4 * theta, eval_reg[2], target_reg[0])

inverse_qft_and_measure(qc, eval_reg, cr)

sim = AerSimulator()
result = sim.run(qc, shots=10000).result()
print(result.get_counts())
