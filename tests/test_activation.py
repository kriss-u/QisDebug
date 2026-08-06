"""Activation lifecycle: patch application, idempotence, and full restoration."""

import gc
import warnings

import pytest
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


def test_activate_reports_all_layers(debug):
    """With qiskit and qiskit-aer installed, all three layers attach."""
    report = debug.start()
    assert report.active
    assert report.qiskit_version == qiskit.__version__
    assert report.layers == {"construction": True, "execution": True, "transpile": True}
    assert report.warnings == []
    assert debug.active


def test_activate_is_idempotent(debug):
    """A second start() neither re-patches nor double-records."""
    first = debug.start()
    second = debug.start()
    assert first is second

    qc = QuantumCircuit(1)
    qc.h(0)
    qc.x(0)
    timeline = debug.get(qc)
    assert len(timeline.gate_events) == 2


def test_deactivate_restores_original_methods(debug):
    """end() puts back the exact original function objects."""
    originals = _current_patch_targets()
    debug.start()
    patched = _current_patch_targets()
    for name in originals:
        assert patched[name] is not originals[name], name
    debug.end()
    restored = _current_patch_targets()
    for name in originals:
        assert restored[name] is originals[name], name
    assert not debug.active


def test_deactivate_without_activate_is_noop(debug):
    """Ending an inactive QisDebug does nothing and does not raise."""
    debug.end()
    debug.end()
    assert not debug.active


def test_no_recording_while_inactive(debug):
    """Constructing a QisDebug alone must not instrument anything."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    with pytest.raises(qisdebug.TimelineNotFoundError):
        debug.get(qc)
    assert debug.timelines() == []


def test_no_recording_after_deactivate(debug):
    """Circuits built after deactivation leave no telemetry."""
    debug.start()
    debug.end()
    debug.clear()
    qc = QuantumCircuit(1)
    qc.h(0)
    with pytest.raises(qisdebug.TimelineNotFoundError):
        debug.get(qc)


def test_get_raises_for_untracked_circuit(debug):
    """debug.get() raises TimelineNotFoundError instead of returning None."""
    debug.start()
    qc = QuantumCircuit(1)
    with pytest.raises(qisdebug.TimelineNotFoundError):
        debug.get(qc)


def test_diff_compares_two_tracked_timelines(debug):
    """debug.diff() is a thin wrapper: resolves both targets, delegates to _analysis."""
    debug.start()
    qc_a = QuantumCircuit(2, name="diff_a")
    qc_a.h(0)
    qc_a.cx(0, 1)
    qc_b = QuantumCircuit(2, name="diff_b")
    qc_b.h(0)
    qc_b.cx(0, 1)

    result = debug.diff(qc_a, qc_b)

    assert result.circuit_id_a == debug.get(qc_a).circuit_id
    assert result.circuit_id_b == debug.get(qc_b).circuit_id
    assert [entry.op for entry in result.gate_diff] == ["equal", "equal"]


def test_diff_accepts_names_and_raises_for_untracked(debug):
    """debug.diff() resolves circuit names the same way debug.get() does."""
    debug.start()
    qc = QuantumCircuit(1, name="named")
    qc.h(0)
    with pytest.raises(qisdebug.TimelineNotFoundError):
        debug.diff("named", "missing")


def test_diff_with_seq_bounds_compares_two_moments_of_one_circuit(debug):
    """debug.diff(a, a, seq_a=..., seq_b=...) diffs two points in one circuit's replay."""
    debug.start()
    qc = QuantumCircuit(2, name="replay")
    qc.h(0)
    first_seq = debug.get(qc).events[-1].seq
    qc.cx(0, 1)
    second_seq = debug.get(qc).events[-1].seq

    result = debug.diff(qc, qc, seq_a=first_seq, seq_b=second_seq)

    assert result.seq_a == first_seq
    assert result.seq_b == second_seq
    assert [entry.op for entry in result.gate_diff] == ["equal", "insert"]


def test_two_instances_patch_concurrently_with_independent_stores():
    """Two started QisDebug instances share patches but never mix telemetry."""
    first = qisdebug.QisDebug()
    second = qisdebug.QisDebug()
    try:
        first.start()
        second.start()

        qc_a = QuantumCircuit(1, name="a")
        qc_a.h(0)
        qc_b = QuantumCircuit(1, name="b")
        qc_b.x(0)

        first_timeline = first.get(qc_a)
        assert [e.gate for e in first_timeline.gate_events] == ["h"]
        second_timeline = second.get(qc_a)
        assert [e.gate for e in second_timeline.gate_events] == ["h"]

        assert {t.name for t in first.timelines()} == {"a", "b"}
        assert {t.name for t in second.timelines()} == {"a", "b"}
    finally:
        first.end()
        second.end()


def test_one_instance_ending_leaves_the_other_recording():
    """Ending one of two concurrently started instances doesn't unpatch the other."""
    first = qisdebug.QisDebug()
    second = qisdebug.QisDebug()
    try:
        first.start()
        second.start()
        first.end()
        assert not first.active
        assert second.active

        qc = QuantumCircuit(1, name="after-first-ends")
        qc.h(0)

        with pytest.raises(qisdebug.TimelineNotFoundError):
            first.get(qc)
        timeline = second.get(qc)
        assert [e.gate for e in timeline.gate_events] == ["h"]
    finally:
        first.end()
        second.end()


def test_gc_without_end_warns_resource_warning():
    """A started QisDebug that's garbage collected without end() warns."""
    instance = qisdebug.QisDebug()
    instance.start()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            del instance
            gc.collect()
        assert any(issubclass(w.category, ResourceWarning) for w in caught)
    finally:
        # The finalizer already ran and released the patches; nothing left to end.
        pass
