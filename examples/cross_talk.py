"""A QFT-based adder of two 2-qubit registers (``a`` and ``b``) into ``a``,
using controlled-phase gates between the registers -- an example of
debugging cross-register gate interactions with qisdebug.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator

reg_a = QuantumRegister(2, name="a")
reg_b = QuantumRegister(2, name="b")
cr = ClassicalRegister(2, name="sum")

qc = QuantumCircuit(reg_a, reg_b, cr)

qc.x(reg_a[1])
qc.x(reg_b[0])

qc.h(reg_a[1])
qc.cp(np.pi / 2, reg_a[1], reg_a[0])
qc.h(reg_a[0])
qc.swap(reg_a[0], reg_a[1])

qc.cp(np.pi / 2, reg_b[0], reg_a[0])
qc.cp(np.pi, reg_b[1], reg_a[0])

qc.cp(np.pi, reg_b[0], reg_a[1])
qc.cp(2 * np.pi, reg_b[1], reg_a[1])

qc.swap(reg_a[0], reg_a[1])
qc.h(reg_a[0])
qc.cp(-np.pi / 2, reg_a[1], reg_a[0])
qc.h(reg_a[1])

qc.measure(reg_a, cr)

sim = AerSimulator()
result = sim.run(qc, shots=10000).result()
print(result.get_counts())
