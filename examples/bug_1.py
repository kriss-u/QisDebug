"""Buggy 3-qubit quantum phase estimation: uses ``crz`` for the
controlled-eigenvalue kickback where a controlled-phase gate (``cp``) is
required, producing skewed measurement counts. See ``bug_1_fixed.py`` for
the corrected version and ``phase_estimation_bug.py`` for the same bug in a
more general, parametrized form.
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

qc.crz(4 * theta, eval_reg[0], target_reg[0])
qc.crz(2 * theta, eval_reg[1], target_reg[0])
qc.crz(1 * theta, eval_reg[2], target_reg[0])

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
