"""Activation lifecycle: patch application, idempotence, and full restoration."""

import qiskit
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.transpiler.passmanager import PassManager

import qdebug


def _current_patch_targets():
    """Snapshot the attributes QDebug patches, for identity comparison."""
    from qiskit_aer.backends.aerbackend import AerBackend

    return {
        "append": QuantumCircuit.append,
        "_append": QuantumCircuit._append,
        "pm_run": PassManager.run,
        "aer_run": AerBackend.run,
    }


def test_activate_reports_all_layers():
    """With qiskit and qiskit-aer installed, all three layers attach."""
    report = qdebug.activate()
    assert report.active
    assert report.qiskit_version == qiskit.__version__
    assert report.layers == {"construction": True, "execution": True, "transpile": True}
    assert report.warnings == []
    assert qdebug.is_active()


def test_activate_is_idempotent():
    """A second activate() neither re-patches nor double-records."""
    first = qdebug.activate()
    second = qdebug.activate()
    assert first is second

    qc = QuantumCircuit(1)
    qc.h(0)
    qc.x(0)
    timeline = qdebug.timeline(qc)
    assert timeline is not None
    assert len(timeline.gate_events) == 2


def test_deactivate_restores_original_methods():
    """deactivate() puts back the exact original function objects."""
    originals = _current_patch_targets()
    qdebug.activate()
    patched = _current_patch_targets()
    for name in originals:
        assert patched[name] is not originals[name], name
    qdebug.deactivate()
    restored = _current_patch_targets()
    for name in originals:
        assert restored[name] is originals[name], name
    assert not qdebug.is_active()


def test_deactivate_without_activate_is_noop():
    """Deactivating an inactive QDebug does nothing and does not raise."""
    qdebug.deactivate()
    qdebug.deactivate()
    assert not qdebug.is_active()


def test_no_recording_while_inactive():
    """Importing qdebug alone must not instrument anything."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    assert qdebug.timeline(qc) is None
    assert qdebug.timelines() == []


def test_no_recording_after_deactivate():
    """Circuits built after deactivation leave no telemetry."""
    qdebug.activate()
    qdebug.deactivate()
    qdebug.clear()
    qc = QuantumCircuit(1)
    qc.h(0)
    assert qdebug.timeline(qc) is None
