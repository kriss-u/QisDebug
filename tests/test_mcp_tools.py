"""Agent-facing query functions in qdebug._mcp.tools.

These call straight into the same STORE/Timeline/_analysis logic the CLI and
GUI frontends use, so the checks here are about the tool layer's contract
(JSON-serializable in/out, correct lookups, useful errors) rather than
re-testing statevector math already covered by test_analysis.py.
"""

import json
import textwrap

import pytest
from pydantic import ValidationError

import qdebug
from conftest import get_timeline
from qdebug._mcp import tools


@pytest.fixture(autouse=True)
def clean_watches():
    """Watches are MCP-session state, not store state, so reset them separately."""
    tools.clear()
    yield
    tools.clear()


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


def test_clear_empties_the_store_and_watches(tmp_path):
    circuit_id = _run_bell_and_get_id(tmp_path)
    tools.add_watch(circuit_id, "qubit", 0)
    assert tools.list_timelines()
    assert tools.list_watches()
    assert tools.clear() == "cleared"
    assert tools.list_timelines() == []
    assert tools.list_watches() == []


def test_run_script_scoped_by_scripts_own_activate_calls(tmp_path):
    script = tmp_path / "scoped.py"
    script.write_text(
        textwrap.dedent(
            """
            import qdebug
            from qiskit.circuit import QuantumCircuit

            qc = QuantumCircuit(2, name="scoped")
            qc.h(0)
            qdebug.activate()
            qc.cx(0, 1)
            qdebug.deactivate()
            qc.h(1)
            """
        )
    )
    result = tools.run_script(str(script), auto_activate=False)
    assert result["error"] is None
    scoped = next(t for t in result["timelines"] if t["name"] == "scoped")
    events = tools.get_events(scoped["circuit_id"], types=["gate_added"])
    assert [e["gate"] for e in events] == ["cx"]


def test_run_script_passes_argv(tmp_path):
    script = tmp_path / "argv.py"
    script.write_text(
        textwrap.dedent(
            """
            import sys
            import qdebug
            from qiskit.circuit import QuantumCircuit

            qc = QuantumCircuit(1, name=sys.argv[1])
            qc.x(0)
            """
        )
    )
    result = tools.run_script(str(script), argv=["from-argv"])
    assert result["error"] is None
    assert any(t["name"] == "from-argv" for t in result["timelines"])


def test_get_events_filters_and_pages(tmp_path):
    circuit_id = _run_bell_and_get_id(tmp_path)
    gates = tools.get_events(circuit_id, types=["gate_added"])
    assert [e["gate"] for e in gates] == ["h", "cx", "measure", "measure"]

    first_two = tools.get_events(circuit_id, types=["gate_added"], limit=2)
    assert [e["gate"] for e in first_two] == ["h", "cx"]

    rest = tools.get_events(circuit_id, types=["gate_added"], since_seq=first_two[-1]["seq"])
    assert [e["gate"] for e in rest] == ["measure", "measure"]


def test_get_state_snapshot_returns_structured_state(tmp_path):
    circuit_id = _run_bell_and_get_id(tmp_path)
    final = tools.get_state_snapshot(circuit_id)
    assert final["num_qubits"] == 2
    assert len(final["amplitudes"]) == 4
    assert final["probabilities"][0] == pytest.approx(0.5, abs=1e-6)
    assert final["probabilities"][3] == pytest.approx(0.5, abs=1e-6)
    assert "|00>" in final["formatted"]

    initial = tools.get_state_snapshot(circuit_id, step=0)
    assert initial["probabilities"][0] == pytest.approx(1.0, abs=1e-6)


def test_get_state_snapshot_unknown_step_lists_available(tmp_path):
    circuit_id = _run_bell_and_get_id(tmp_path)
    with pytest.raises(ValueError, match="available steps"):
        tools.get_state_snapshot(circuit_id, step=99)


def test_watch_lifecycle_reports_qubit_state(tmp_path):
    circuit_id = _run_bell_and_get_id(tmp_path)
    message = tools.add_watch(circuit_id, "qubit", 0)
    assert "qubit 0" in message

    listed = tools.list_watches()
    assert len(listed) == 1
    assert listed[0]["kind"] == "qubit"

    reports = tools.read_watches()
    assert len(reports) == 1
    assert "P(1)=0.5" in reports[0]["text"]

    assert "unwatched" in tools.remove_watch(circuit_id, "qubit", 0)
    assert tools.list_watches() == []


def test_watch_upto_seq_bounds_the_read(tmp_path):
    circuit_id = _run_bell_and_get_id(tmp_path)
    tools.add_watch(circuit_id, "qubit", 1)
    first_gate_seq = tools.get_events(circuit_id, types=["gate_added"], limit=1)[0]["seq"]
    report = tools.read_watches(upto_seq=first_gate_seq)[0]
    assert "gates touched: 0" in report["text"]


def test_add_watch_validates_targets(tmp_path):
    circuit_id = _run_bell_and_get_id(tmp_path)
    with pytest.raises(ValueError, match="out of range"):
        tools.add_watch(circuit_id, "qubit", 5)
    with pytest.raises(ValueError, match="available"):
        tools.add_watch(circuit_id, "register", "nope")
    with pytest.raises(ValueError, match="not watching"):
        tools.remove_watch(circuit_id, "clbit", 0)


def test_save_timeline_round_trips_through_json(tmp_path):
    circuit_id = _run_bell_and_get_id(tmp_path)
    out = tmp_path / "evidence.json"
    result = tools.save_timeline(circuit_id, str(out))
    assert result["events"] > 0
    payload = json.loads(out.read_text())
    assert payload["circuit_id"] == circuit_id
    assert len(payload["events"]) == result["events"]


def test_configuration_round_trip():
    assert tools.get_configuration()["max_statevector_qubits"] == 12
    updated = tools.set_configuration({"max_statevector_qubits": 20})
    assert updated["max_statevector_qubits"] == 20
    assert tools.get_configuration()["max_statevector_qubits"] == 20


def test_set_configuration_rejects_bad_values():
    with pytest.raises(ValidationError):
        tools.set_configuration({"max_statevector_qubits": -1})
    assert tools.get_configuration()["max_statevector_qubits"] == 12
