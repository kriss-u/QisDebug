"""Agent-facing query functions in qdebug._mcp.tools.

These call straight into the same STORE/Timeline/_analysis logic the CLI and
GUI frontends use, so the checks here are about the tool layer's contract
(JSON-serializable in/out, correct lookups, useful errors) rather than
re-testing statevector math already covered by test_analysis.py.
"""

import textwrap

import pytest

import qdebug
from conftest import get_timeline
from qdebug._mcp import tools


def _write_bell_script(tmp_path):
    script = tmp_path / "bell.py"
    script.write_text(
        textwrap.dedent(
            """
            from qiskit.circuit import QuantumCircuit
            from qiskit_aer import AerSimulator

            qc = QuantumCircuit(2, 2, name="bell")
            qc.h(0)
            qc.cx(0, 1)
            qc.measure([0, 1], [0, 1])
            AerSimulator().run(qc, shots=200).result()
            """
        )
    )
    return str(script)


def _run_bell_and_get_id(tmp_path) -> str:
    """Run the bell script and return the id of the timeline named "bell".

    ``AerSimulator()`` construction can register incidental internal
    circuits of its own alongside the user's, so tests locate the circuit
    they care about by name rather than assuming it's the only timeline.
    """
    tools.run_script(_write_bell_script(tmp_path))
    return next(t for t in tools.list_timelines() if t["name"] == "bell")["circuit_id"]


def test_run_script_activates_captures_and_deactivates(tmp_path):
    assert not qdebug.is_active()
    result = tools.run_script(_write_bell_script(tmp_path))
    assert not qdebug.is_active()
    assert result["error"] is None
    bell = next(t for t in result["timelines"] if t["name"] == "bell")
    assert bell["anomalies"] == []


def test_run_script_reports_exceptions_without_leaking_them(tmp_path):
    script = tmp_path / "broken.py"
    script.write_text("raise ValueError('boom')\n")
    result = tools.run_script(str(script))
    assert result["error"] == "ValueError: boom"
    assert result["timelines"] == []
    assert not qdebug.is_active()


def test_list_and_get_timeline_round_trip(tmp_path):
    circuit_id = _run_bell_and_get_id(tmp_path)
    assert circuit_id in {t["circuit_id"] for t in tools.list_timelines()}

    summary = tools.get_timeline_summary(circuit_id)
    assert "bell" in summary

    exported = tools.get_timeline(circuit_id)
    assert exported["circuit_id"] == circuit_id
    assert any(e["type"] == "gate_added" for e in exported["events"])


def test_get_timeline_unknown_id_raises_lookup_error():
    with pytest.raises(LookupError):
        tools.get_timeline("not-a-real-id")


def test_get_circuit_text_reflects_step(tmp_path):
    circuit_id = _run_bell_and_get_id(tmp_path)
    after_h = tools.get_circuit_text(circuit_id, step=1)
    full = tools.get_circuit_text(circuit_id)
    assert "X" not in after_h  # cx's target box hasn't been drawn onto qubit 1 yet
    assert "X" in full


def test_compare_statevector_via_tool_layer(tmp_path):
    circuit_id = _run_bell_and_get_id(tmp_path)
    timeline = get_timeline(circuit_id)
    final_step = timeline.snapshots[-1].step

    root_half = 0.7071067811865476
    good = tools.compare_statevector(
        circuit_id, final_step, [root_half, 0, 0, root_half], [0, 0, 0, 0]
    )
    assert good["matched"]

    bad = tools.compare_statevector(circuit_id, final_step, [1.0, 0, 0, 0], [0, 0, 0, 0])
    assert not bad["matched"]


def test_compare_counts_via_tool_layer(tmp_path):
    circuit_id = _run_bell_and_get_id(tmp_path)
    # 200 shots alone can swing a couple percentage points from the ideal
    # 50/50 split, so use a tolerance wide enough to absorb sampling noise
    # while still catching a real semantic bug (which shows up as ~50pp off).
    result = tools.compare_counts(circuit_id, {"00": 0.5, "11": 0.5}, atol=0.15)
    assert result["matched"]


def test_find_anomalies_via_tool_layer(tmp_path):
    circuit_id = _run_bell_and_get_id(tmp_path)
    assert tools.find_anomalies(circuit_id) == []


def test_clear_empties_the_store(tmp_path):
    tools.run_script(_write_bell_script(tmp_path))
    assert tools.list_timelines()
    assert tools.clear() == "cleared"
    assert tools.list_timelines() == []
