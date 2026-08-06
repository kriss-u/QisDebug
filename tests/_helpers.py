"""Circuit-building and GUI-window helpers shared across the test suite."""

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


def build_gui_window(monkeypatch, timeline, name="viz"):
    """Construct a ``_TimelineWindow`` directly, bypassing the blocking Qt event loop.

    Returns ``(app, window)``; callers are responsible for ``del app`` once done,
    matching the cleanup every direct-construction GUI test already performs.
    """
    from PySide6.QtWidgets import QApplication

    from qisdebug._frontends import _render
    from qisdebug._frontends import gui as gui_module

    monkeypatch.setattr(QApplication, "exec", lambda self: None)
    app = QApplication.instance() or QApplication([])
    merged, tracks, active_labels = _render.merge_timelines([timeline])
    # protected-access: this helper exists only for the frontend module's own
    # tests, which have no public constructor or API for these internals.
    view = gui_module._TimelineView(  # pylint: disable=protected-access
        merged=merged,
        tracks=tracks,
        active_labels=active_labels,
        timelines_by_id={timeline.circuit_id: timeline},
    )
    window = gui_module._TimelineWindow(name, view)  # pylint: disable=protected-access
    return app, window
