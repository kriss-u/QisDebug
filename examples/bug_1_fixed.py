"""Corrected version of ``bug_1.py``: uses controlled-phase (``cp``) gates
for the eigenvalue kickback instead of ``crz``, producing the expected
measurement counts for 3-qubit quantum phase estimation.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator

eval_reg = QuantumRegister(3, name="eval")
target_reg = QuantumRegister(1, name="target")
cr = ClassicalRegister(3, name="meas")

qc = QuantumCircuit(eval_reg, target_reg, cr)

qc.x(target_reg[0])

qc.h(eval_reg[0])
qc.h(eval_reg[1])
qc.h(eval_reg[2])

theta = np.pi / 2

qc.cp(1 * theta, eval_reg[0], target_reg[0])
qc.cp(2 * theta, eval_reg[1], target_reg[0])
qc.cp(4 * theta, eval_reg[2], target_reg[0])

qc.swap(eval_reg[0], eval_reg[2])

qc.h(eval_reg[0])
qc.cp(-np.pi / 2, eval_reg[0], eval_reg[1])
qc.h(eval_reg[1])
qc.cp(-np.pi / 4, eval_reg[0], eval_reg[2])
qc.cp(-np.pi / 2, eval_reg[1], eval_reg[2])
qc.h(eval_reg[2])

qc.measure(eval_reg, cr)

sim = AerSimulator()
result = sim.run(qc, shots=10000).result()
print(result.get_counts())
