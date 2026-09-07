"""Circuit-building and GUI-bridge helpers shared across the test suite."""

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit_aer import AerSimulator


def bell_circuit(name="bell"):
    """Return a 2-qubit bell circuit with measurements, uninstrumented."""
    qc = QuantumCircuit(2, 2, name=name)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def traced_bell(debug, name="viz", shots=10, **run_kwargs):
    """Build, run, and return the timeline for a bell circuit under instrumentation."""
    debug.start()
    qc = bell_circuit(name)
    AerSimulator().run(qc, shots=shots, **run_kwargs).result()
    return debug.get(qc)


def build_bridge(timeline, name="viz"):
    """Construct a :class:`~qisdebug._frontends.bridge.TimelineBridge` directly.

    The bridge is plain Python (no webview, no event loop), so unlike the
    old Qt ``_TimelineWindow`` this needs no app instance or teardown --
    callers just build one and call its methods directly.
    """
    from qisdebug._frontends import _render
    from qisdebug._frontends.bridge import TimelineBridge

    merged, tracks, active_labels = _render.merge_timelines([timeline])
    return TimelineBridge(
        title=name,
        merged=merged,
        tracks=tracks,
        active_labels=active_labels,
        timelines_by_id={timeline.circuit_id: timeline},
    )
