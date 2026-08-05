"""QisDebug on a realistic protocol: teleportation over a noisy hardware mock.

Run with::

    python examples/teleport_noisy_hardware.py

This mirrors how someone would actually use QisDebug on a research circuit:

- Two independent subcircuits (an arbitrary payload state and an EPR pair)
  are built separately, then composed into a third, larger teleportation
  circuit -- a common pattern when assembling circuits from reusable pieces.
- The combined circuit is transpiled for and run on a noisy simulator built
  from ``FakeManilaV2``, a mock of a real 5-qubit IBM device that ships with
  qiskit-ibm-runtime, so the noise model reflects real calibration data
  (T1/T2, readout error, gate error) rather than a synthetic one.
- The same circuit is also run on a noiseless simulator so the effect of the
  noise model is visible directly in the measurement statistics.

Along the way this demonstrates all three QisDebug hooks and one of its
graceful-degradation paths: teleportation requires a mid-circuit measurement
and a classically conditioned correction, which makes statevector capture
non-deterministic, so QisDebug skips it for that circuit and records why
instead of guessing or crashing.
"""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeManilaV2

import qisdebug


def payload_state(theta: float = 0.9, phi: float = 1.3) -> QuantumCircuit:
    """Single-qubit circuit preparing the arbitrary state to be teleported."""
    qc = QuantumCircuit(1, name="payload")
    qc.ry(theta, 0)
    qc.rz(phi, 0)
    return qc


def epr_pair() -> QuantumCircuit:
    """Two-qubit Bell pair shared between "sender" and "receiver"."""
    qc = QuantumCircuit(2, name="epr_pair")
    qc.h(0)
    qc.cx(0, 1)
    return qc


def teleportation_circuit(
    payload: QuantumCircuit, epr: QuantumCircuit
) -> QuantumCircuit:
    """Compose the payload and EPR subcircuits into a full teleportation circuit.

    Qubit 0 carries the payload state, qubits 1 and 2 are the shared EPR
    pair. A Bell measurement on qubits 0 and 1 collapses the state; the
    classically conditioned corrections on qubit 2 reconstruct the original
    payload state there.
    """
    qc = QuantumCircuit(3, 3, name="teleport")
    qc.compose(payload, qubits=[0], inplace=True)
    qc.compose(epr, qubits=[1, 2], inplace=True)
    qc.cx(0, 1)
    qc.h(0)
    qc.measure(0, 0)
    qc.measure(1, 1)
    with qc.if_test((qc.clbits[1], 1)):
        qc.x(2)
    with qc.if_test((qc.clbits[0], 1)):
        qc.z(2)
    qc.measure(2, 2)
    return qc


def ideal_receiver_marginal(payload: QuantumCircuit) -> dict[str, float]:
    """Expected P(0)/P(1) on the receiver qubit if teleportation is exact."""
    probs = Statevector(payload).probabilities()
    return {"0": probs[0], "1": probs[1]}


def receiver_marginal(counts: dict[str, int]) -> dict[str, float]:
    """Marginal distribution of the receiver qubit (leftmost bit) from counts."""
    totals = {"0": 0, "1": 0}
    for bitstring, count in counts.items():
        totals[bitstring[0]] += count
    shots = sum(totals.values())
    return {bit: total / shots for bit, total in totals.items()}


def main() -> None:
    """Build, transpile and run a teleportation circuit assembled from two subcircuits."""
    debug = qisdebug.QisDebug()
    debug.start()

    payload = payload_state()
    epr = epr_pair()
    teleport = teleportation_circuit(payload, epr)

    device = FakeManilaV2()
    pass_manager = generate_preset_pass_manager(optimization_level=1, backend=device)
    transpiled = pass_manager.run(teleport)

    noisy_backend = AerSimulator.from_backend(device)
    ideal_backend = AerSimulator()
    shots = 4000

    # Run the payload circuit on its own too: it is a pure unitary with no
    # measurement, so QisDebug can capture its full statevector evolution,
    # which is what the "payload state evolution" section below reads.
    ideal_backend.run(payload, shots=1).result()

    noisy_counts = noisy_backend.run(transpiled, shots=shots).result().get_counts()
    ideal_counts = ideal_backend.run(transpiled, shots=shots).result().get_counts()

    expected = ideal_receiver_marginal(payload)
    print(f"expected receiver marginal (analytic): {expected}")
    print(f"ideal simulator receiver marginal:      {receiver_marginal(ideal_counts)}")
    print(f"{device.name} noisy receiver marginal:   {receiver_marginal(noisy_counts)}")

    print("\nper-circuit timelines:")
    timelines: dict[str, qisdebug.Timeline] = {}
    for qc in (payload, epr, teleport):
        timeline = debug.get(qc)
        timelines[qc.name] = timeline
        print(f"\n{timeline.summary()}")
        for event in timeline.events:
            print(f"  [{event.seq:>3}] {event.type.value:<22} {event.describe()}")

    payload_timeline = timelines["payload"]
    print("\npayload state evolution (pure unitary, so statevector capture applies):")
    for snapshot in payload_timeline.snapshots:
        probs = {
            format(i, f"0{snapshot.num_qubits}b"): round(p, 3)
            for i, p in enumerate(snapshot.probabilities())
            if p > 1e-9
        }
        print(f"  step {snapshot.step} ({snapshot.gate or 'initial':<7}) {probs}")

    teleport_timeline = timelines["teleport"]
    print(
        f"\nteleport circuit has {len(teleport_timeline.snapshots)} statevector snapshots "
        "(mid-circuit measurement + classical control disable capture; see the note event above)"
    )

    debug.show([payload_timeline, timelines["epr_pair"], teleport_timeline])

    debug.end()


if __name__ == "__main__":
    main()
