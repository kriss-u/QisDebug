# QisDebug

Zero-touch instrumentation and timeline debugging for [Qiskit](https://www.ibm.com/quantum/qiskit).

QisDebug attaches full circuit and execution visibility to your existing
Qiskit code the way OpenTelemetry or Sentry attach to a web framework: one
import, one activation call, and no changes to how you build, transpile, or
run circuits.

```python
import qisdebug

qisdebug.activate()

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

qc = QuantumCircuit(2, 2, name="bell")
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])

backend = AerSimulator()
counts = backend.run(transpile(qc, backend), shots=1000).result().get_counts()

print(qisdebug.timeline(qc).summary())
```

```{toctree}
:maxdepth: 2
:caption: Contents

install
api
contributing
```
