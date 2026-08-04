"""Activation lifecycle: patch application, idempotence, and full restoration."""

import qiskit
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.transpiler.passmanager import PassManager

import qisdebug


def _current_patch_targets():
    """Snapshot the attributes QisDebug patches, for identity comparison."""
    from qiskit_aer.backends.aerbackend import AerBackend

    return {
        "append": QuantumCircuit.append,
        "_append": QuantumCircuit._append,
        "pm_run": PassManager.run,
        "aer_run": AerBackend.run,
    }


def test_activate_reports_all_layers():
    """With qiskit and qiskit-aer installed, all three layers attach."""
    report = qisdebug.activate()
    assert report.active
    assert report.qiskit_version == qiskit.__version__
    assert report.layers == {"construction": True, "execution": True, "transpile": True}
    assert report.warnings == []
    assert qisdebug.is_active()


def test_activate_is_idempotent():
    """A second activate() neither re-patches nor double-records."""
    first = qisdebug.activate()
    second = qisdebug.activate()
    assert first is second

    qc = QuantumCircuit(1)
    qc.h(0)
    qc.x(0)
    timeline = qisdebug.timeline(qc)
    assert timeline is not None
    assert len(timeline.gate_events) == 2


def test_deactivate_restores_original_methods():
    """deactivate() puts back the exact original function objects."""
    originals = _current_patch_targets()
    qisdebug.activate()
    patched = _current_patch_targets()
    for name in originals:
        assert patched[name] is not originals[name], name
    qisdebug.deactivate()
    restored = _current_patch_targets()
    for name in originals:
        assert restored[name] is originals[name], name
    assert not qisdebug.is_active()


def test_deactivate_without_activate_is_noop():
    """Deactivating an inactive QisDebug does nothing and does not raise."""
    qisdebug.deactivate()
    qisdebug.deactivate()
    assert not qisdebug.is_active()


def test_no_recording_while_inactive():
    """Importing qisdebug alone must not instrument anything."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    assert qisdebug.timeline(qc) is None
    assert qisdebug.timelines() == []


def test_no_recording_after_deactivate():
    """Circuits built after deactivation leave no telemetry."""
    qisdebug.activate()
    qisdebug.deactivate()
    qisdebug.clear()
    qc = QuantumCircuit(1)
    qc.h(0)
    assert qisdebug.timeline(qc) is None
