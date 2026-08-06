"""Agent-facing query functions in qisdebug._mcp.tools.

These call straight into the same store/Timeline/_analysis logic the CLI and
GUI frontends use, so the checks here are about the tool layer's contract
(JSON-serializable in/out, correct lookups, useful errors) rather than
re-testing statevector math already covered by test_analysis.py.

protected-access is disabled file-wide: by design (see the MCP server section
of the project's CLAUDE.md) ``tools._debug``/``tools._watches`` are the one
module-level instance scoping the whole server process, with no public
accessor -- these tests have to reach into them directly to reset state
between runs and to assert on session lifecycle.
"""

# pylint: disable=protected-access

import json
import textwrap
import time

import pytest
from pydantic import ValidationError

from qisdebug._config import QisDebugConfig
from qisdebug._mcp import tools


@pytest.fixture(autouse=True)
def clean_mcp_debug():
    """The MCP tools module owns a process-wide `_debug`/`_watches`, mirroring
    one real MCP server process; reset all of it around each test since there
    is no per-test fixture instance to isolate them for us.
    """

    def _reset():
        tools._debug.end()
        tools._debug.clear()
        tools._debug.config = QisDebugConfig()
        tools._watches.clear()
        tools._runs.clear()
        for handle, _condition in list(tools._debug.list_breakpoints()):
            tools._debug.remove_breakpoint(handle)
        tools._debug.store.breakpoints.resume()

    _reset()
    yield
    _reset()


def _wait_until(predicate, timeout=5.0):
    """Poll ``predicate`` until it's truthy or ``timeout`` elapses; returns
    the last (possibly falsy) result, matching background-run tests'
    "wait for the worker thread to reach some state" needs without sleeping
    a fixed, possibly-flaky duration.
    """
    deadline = time.monotonic() + timeout
    result = predicate()
    while not result and time.monotonic() < deadline:
        time.sleep(0.01)
        result = predicate()
    return result


def _wait_for_status(run_id, status):
    """Poll ``poll_run`` until it reports ``status``; returns the last response."""
    return _wait_until(
        lambda: (lambda s: s if s["status"] == status else None)(tools.poll_run(run_id))
    )


def _wait_while_paused(run_id):
    """Poll ``poll_run`` until it stops reporting "paused"; returns the last response."""
    return _wait_until(
        lambda: (lambda s: s if s["status"] != "paused" else None)(
            tools.poll_run(run_id)
        )
    )


def _write_bell_script(tmp_path):
    script = tmp_path / "bell.py"
    script.write_text(textwrap.dedent("""
            from qiskit.circuit import QuantumCircuit
            from qiskit_aer import AerSimulator

            qc = QuantumCircuit(2, 2, name="bell")
            qc.h(0)
            qc.cx(0, 1)
            qc.measure([0, 1], [0, 1])
            AerSimulator().run(qc, shots=200).result()
            """))
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
    """Activation and deactivation around run_script leave no session active before or after."""
    assert not tools._debug.active
    result = tools.run_script(_write_bell_script(tmp_path))
    assert not tools._debug.active
    assert result["error"] is None
    bell = next(t for t in result["timelines"] if t["name"] == "bell")
    assert not bell["anomalies"]


def test_run_script_reports_exceptions_without_leaking_them(tmp_path):
    """An exception inside the target script is reported as a message, not re-raised."""
    script = tmp_path / "broken.py"
    script.write_text("raise ValueError('boom')\n")
    result = tools.run_script(str(script))
    assert result["error"] == "ValueError: boom"
    assert not result["timelines"]
    assert not tools._debug.active


def test_list_and_get_timeline_round_trip(tmp_path):
    """list_timelines/get_timeline_summary/get_timeline agree on the same circuit."""
    circuit_id = _run_bell_and_get_id(tmp_path)
    assert circuit_id in {t["circuit_id"] for t in tools.list_timelines()}

    summary = tools.get_timeline_summary(circuit_id)
    assert "bell" in summary

    exported = tools.get_timeline(circuit_id)
    assert exported["circuit_id"] == circuit_id
    assert any(e["type"] == "gate_added" for e in exported["events"])


def test_get_timeline_unknown_id_raises_lookup_error():
    """Looking up an id that was never tracked is a LookupError."""
    with pytest.raises(LookupError):
        tools.get_timeline("not-a-real-id")


def test_get_circuit_text_reflects_step(tmp_path):
    """get_circuit_text(step=...) draws only the gates recorded up to that step."""
    circuit_id = _run_bell_and_get_id(tmp_path)
    after_h = tools.get_circuit_text(circuit_id, step=1)
    full = tools.get_circuit_text(circuit_id)
    assert "X" not in after_h  # cx's target box hasn't been drawn onto qubit 1 yet
    assert "X" in full


def _write_transpile_script(tmp_path):
    script = tmp_path / "transpile_bell.py"
    script.write_text(textwrap.dedent("""
            from qiskit import transpile
            from qiskit.circuit import QuantumCircuit

            qc = QuantumCircuit(2, 2, name="routed")
            qc.h(0)
            qc.cx(0, 1)
            qc.measure([0, 1], [0, 1])
            transpile(qc, coupling_map=[[0, 1], [1, 0]], basis_gates=["cx", "u"])
            """))
    return str(script)


def _run_transpile_and_get_id(tmp_path) -> str:
    tools.run_script(_write_transpile_script(tmp_path))
    return next(t for t in tools.list_timelines() if t["name"] == "routed")[
        "circuit_id"
    ]


def test_get_property_set_returns_full_snapshot(tmp_path):
    """get_property_set returns the full property_set snapshot for a captured pass."""
    circuit_id = _run_transpile_and_get_id(tmp_path)
    passes = [
        e
        for e in tools.get_events(circuit_id, types=["transpile_pass"])
        if e["property_changes"]
    ]
    assert passes
    snapshot = tools.get_property_set(circuit_id, passes[0]["seq"])
    assert isinstance(snapshot, dict)
    assert snapshot


def test_get_property_set_unknown_seq_raises_value_error(tmp_path):
    """Asking for a seq with no transpile_pass event at all is a ValueError."""
    circuit_id = _run_transpile_and_get_id(tmp_path)
    with pytest.raises(ValueError, match="no transpile_pass event"):
        tools.get_property_set(circuit_id, 999_999)


def test_get_property_set_uncaptured_seq_raises_value_error(tmp_path):
    """Asking for a snapshot when capture_property_set='none' is a ValueError."""
    tools._debug.configure(capture_property_set="none")
    circuit_id = _run_transpile_and_get_id(tmp_path)
    passes = tools.get_events(circuit_id, types=["transpile_pass"])
    assert passes
    with pytest.raises(ValueError, match="no property_set snapshot"):
        tools.get_property_set(circuit_id, passes[0]["seq"])


def test_compare_statevector_via_tool_layer(tmp_path):
    """compare_statevector via the tool layer matches/rejects the same as the direct API."""
    circuit_id = _run_bell_and_get_id(tmp_path)
    timeline = tools._debug.get(circuit_id)
    final_step = timeline.snapshots[-1].step

    root_half = 0.7071067811865476
    good = tools.compare_statevector(
        circuit_id, {"00": [root_half, 0], "11": [root_half, 0]}, step=final_step
    )
    assert good["matched"]

    bad = tools.compare_statevector(circuit_id, {"00": [1.0, 0]}, step=final_step)
    assert not bad["matched"]


def test_compare_statevector_by_seq(tmp_path):
    """compare_statevector accepts seq as an alternative to step."""
    circuit_id = _run_bell_and_get_id(tmp_path)
    h_seq = tools.get_events(circuit_id, types=["gate_added"], limit=1)[0]["seq"]

    result = tools.compare_statevector(circuit_id, {"00": [1.0, 0]}, seq=h_seq)
    assert not result["matched"]  # state right after h(0) is a superposition, not |00>

    root_half = 0.7071067811865476
    after_h = tools.compare_statevector(
        circuit_id, {"00": [root_half, 0], "01": [root_half, 0]}, seq=h_seq
    )
    assert after_h["matched"]


def test_compare_statevector_rejects_step_and_seq_together(tmp_path):
    """Passing both step and seq to compare_statevector is a ValueError."""
    circuit_id = _run_bell_and_get_id(tmp_path)
    with pytest.raises(ValueError, match="not both"):
        tools.compare_statevector(circuit_id, {"00": [1.0, 0]}, step=0, seq=1)


def test_compare_counts_via_tool_layer(tmp_path):
    """compare_counts via the tool layer reports a match within the given tolerance."""
    circuit_id = _run_bell_and_get_id(tmp_path)
    # 200 shots alone can swing a couple percentage points from the ideal
    # 50/50 split, so use a tolerance wide enough to absorb sampling noise
    # while still catching a real semantic bug (which shows up as ~50pp off).
    result = tools.compare_counts(circuit_id, {"00": 0.5, "11": 0.5}, atol=0.15)
    assert result["matched"]


def test_find_anomalies_via_tool_layer(tmp_path):
    """find_anomalies via the tool layer finds nothing wrong with a clean bell run."""
    circuit_id = _run_bell_and_get_id(tmp_path)
    assert not tools.find_anomalies(circuit_id)


def _write_two_bell_script(tmp_path):
    script = tmp_path / "two_bells.py"
    script.write_text(textwrap.dedent("""
            from qiskit.circuit import QuantumCircuit
            from qiskit_aer import AerSimulator

            for name in ("bell_a", "bell_b"):
                qc = QuantumCircuit(2, 2, name=name)
                qc.h(0)
                qc.cx(0, 1)
                qc.measure([0, 1], [0, 1])
                AerSimulator().run(qc, shots=20000, seed_simulator=1234).result()
            """))
    return str(script)


def test_diff_timelines_via_tool_layer(tmp_path):
    """diff_timelines via the tool layer reports an all-equal gate diff for identical runs."""
    tools.run_script(_write_two_bell_script(tmp_path))
    id_a = next(t for t in tools.list_timelines() if t["name"] == "bell_a")[
        "circuit_id"
    ]
    id_b = next(t for t in tools.list_timelines() if t["name"] == "bell_b")[
        "circuit_id"
    ]

    result = tools.diff_timelines(id_a, id_b)

    assert [entry["op"] for entry in result["gate_diff"]] == ["equal"] * len(
        result["gate_diff"]
    )
    assert result["statevector_diff"]["matched"]
    assert result["counts_diff"]["matched"]


def test_diff_timelines_with_seq_bounds_diffs_two_moments_of_one_circuit(tmp_path):
    """diff_timelines with seq bounds diffs two moments of the same circuit's history."""
    circuit_id = _run_bell_and_get_id(tmp_path)
    gate_events = tools.get_events(circuit_id, types=["gate_added"], limit=100)
    first_seq = gate_events[0]["seq"]
    last_seq = gate_events[-1]["seq"]

    result = tools.diff_timelines(
        circuit_id, circuit_id, seq_a=first_seq, seq_b=last_seq
    )

    assert result["seq_a"] == first_seq
    assert result["seq_b"] == last_seq
    ops = [entry["op"] for entry in result["gate_diff"]]
    assert "insert" in ops


def test_diff_timelines_unknown_id_raises_lookup_error(tmp_path):
    """Diffing against an unknown circuit id is a LookupError."""
    circuit_id = _run_bell_and_get_id(tmp_path)
    with pytest.raises(LookupError):
        tools.diff_timelines(circuit_id, "not-a-real-id")


def test_clear_empties_the_store_and_watches(tmp_path):
    """clear() empties both the timeline store and the watch list."""
    circuit_id = _run_bell_and_get_id(tmp_path)
    tools.add_watch(circuit_id, "qubit", 0)
    assert tools.list_timelines()
    assert tools.list_watches()
    assert tools.clear() == "cleared"
    assert not tools.list_timelines()
    assert not tools.list_watches()


def test_run_script_with_auto_activate_false_leaves_this_sessions_debug_untouched(
    tmp_path,
):
    """auto_activate=False no longer scopes via a removed global activate/deactivate.

    A target script cannot reach the MCP tool's own `_debug` instance at all
    now (there's no global to call into) -- the closest it could get is
    starting its own private `QisDebug()`, which this call has no way to
    observe or report back. So the only behavior left to verify is that this
    tool's own `_debug` session (and its store) is completely unaffected by
    the run: nothing gets started, and nothing the script does shows up in
    `_debug.timelines()`.
    """
    script = tmp_path / "scoped.py"
    script.write_text(textwrap.dedent("""
            from qiskit.circuit import QuantumCircuit

            qc = QuantumCircuit(2, name="scoped")
            qc.h(0)
            qc.cx(0, 1)
            qc.h(1)
            """))
    before = tools._debug.active
    result = tools.run_script(str(script), auto_activate=False)
    assert result["error"] is None
    assert tools._debug.active == before
    assert not tools._debug.active
    assert not result["timelines"]
    assert not tools._debug.timelines()


def test_run_script_passes_argv(tmp_path):
    """run_script forwards argv to the target script's sys.argv."""
    script = tmp_path / "argv.py"
    script.write_text(textwrap.dedent("""
            import sys
            from qiskit.circuit import QuantumCircuit

            qc = QuantumCircuit(1, name=sys.argv[1])
            qc.x(0)
            """))
    result = tools.run_script(str(script), argv=["from-argv"])
    assert result["error"] is None
    assert any(t["name"] == "from-argv" for t in result["timelines"])


def test_get_events_filters_and_pages(tmp_path):
    """get_events supports filtering by type, paging with limit, and resuming via since_seq."""
    circuit_id = _run_bell_and_get_id(tmp_path)
    gates = tools.get_events(circuit_id, types=["gate_added"])
    assert [e["gate"] for e in gates] == ["h", "cx", "measure", "measure"]

    first_two = tools.get_events(circuit_id, types=["gate_added"], limit=2)
    assert [e["gate"] for e in first_two] == ["h", "cx"]

    rest = tools.get_events(
        circuit_id, types=["gate_added"], since_seq=first_two[-1]["seq"]
    )
    assert [e["gate"] for e in rest] == ["measure", "measure"]


def test_get_state_snapshot_returns_structured_state(tmp_path):
    """get_state_snapshot returns structured amplitudes/probabilities for a given step."""
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
    """Asking for an unavailable step lists the steps that are actually available."""
    circuit_id = _run_bell_and_get_id(tmp_path)
    with pytest.raises(ValueError, match="available steps"):
        tools.get_state_snapshot(circuit_id, step=99)


def test_get_state_snapshot_by_seq_matches_equivalent_step(tmp_path):
    """get_state_snapshot accepts seq as an alternative to step, returning the same data."""
    circuit_id = _run_bell_and_get_id(tmp_path)
    h_seq = tools.get_events(circuit_id, types=["gate_added"], limit=1)[0]["seq"]

    by_seq = tools.get_state_snapshot(circuit_id, seq=h_seq)
    by_step = tools.get_state_snapshot(circuit_id, step=1)
    assert by_seq == by_step


def test_get_state_snapshot_rejects_non_gate_seq(tmp_path):
    """get_state_snapshot rejects a seq that isn't a gate_added event."""
    circuit_id = _run_bell_and_get_id(tmp_path)
    with pytest.raises(ValueError, match="gate_added event"):
        tools.get_state_snapshot(circuit_id, seq=999_999)


def test_watch_lifecycle_reports_qubit_state(tmp_path):
    """add_watch/list_watches/read_watches/remove_watch round-trip a qubit watch."""
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
    assert not tools.list_watches()


def test_watch_upto_seq_bounds_the_read(tmp_path):
    """read_watches(upto_seq=...) bounds the report to gates touched by that point."""
    circuit_id = _run_bell_and_get_id(tmp_path)
    tools.add_watch(circuit_id, "qubit", 1)
    first_gate_seq = tools.get_events(circuit_id, types=["gate_added"], limit=1)[0][
        "seq"
    ]
    report = tools.read_watches(upto_seq=first_gate_seq)[0]
    assert "gates touched: 0" in report["text"]


def test_add_watch_validates_targets(tmp_path):
    """add_watch/remove_watch validate their targets before accepting them."""
    circuit_id = _run_bell_and_get_id(tmp_path)
    with pytest.raises(ValueError, match="out of range"):
        tools.add_watch(circuit_id, "qubit", 5)
    with pytest.raises(ValueError, match="available"):
        tools.add_watch(circuit_id, "register", "nope")
    with pytest.raises(ValueError, match="not watching"):
        tools.remove_watch(circuit_id, "clbit", 0)


def test_save_timeline_round_trips_through_json(tmp_path):
    """save_timeline writes a JSON file whose contents match the in-memory export."""
    circuit_id = _run_bell_and_get_id(tmp_path)
    out = tmp_path / "evidence.json"
    result = tools.save_timeline(circuit_id, str(out))
    assert result["events"] > 0
    payload = json.loads(out.read_text())
    assert payload["circuit_id"] == circuit_id
    assert len(payload["events"]) == result["events"]


def test_configuration_round_trip():
    """get_configuration/set_configuration round-trip a changed setting."""
    assert tools.get_configuration()["max_statevector_qubits"] == 12
    updated = tools.set_configuration({"max_statevector_qubits": 20})
    assert updated["max_statevector_qubits"] == 20
    assert tools.get_configuration()["max_statevector_qubits"] == 20


def test_set_configuration_rejects_bad_values():
    """set_configuration rejects an out-of-range value via pydantic validation."""
    with pytest.raises(ValidationError):
        tools.set_configuration({"max_statevector_qubits": -1})
    assert tools.get_configuration()["max_statevector_qubits"] == 12


def test_add_list_remove_breakpoint():
    """add_breakpoint/list_breakpoints/remove_breakpoint round-trip a condition."""
    result = tools.add_breakpoint({"kind": "gate", "gate": "cx"})
    handle = result["handle"]
    listed = tools.list_breakpoints()
    assert len(listed) == 1
    assert listed[0]["handle"] == handle
    assert listed[0]["condition"]["gate"] == "cx"
    assert tools.remove_breakpoint(handle) == f"removed breakpoint {handle}"
    assert not tools.list_breakpoints()


def test_add_breakpoint_rejects_invalid_condition():
    """An unknown/malformed condition shape raises instead of silently registering."""
    with pytest.raises(ValidationError):
        tools.add_breakpoint({"kind": "not-a-real-kind"})


def test_background_run_hits_breakpoint_polls_paused_then_resumes(tmp_path):
    """A gate breakpoint pauses a background run; poll_run reports the hit;
    resume_run lets it finish and report the usual timeline/anomaly payload.
    """
    tools.add_breakpoint({"kind": "gate", "gate": "cx"})
    started = tools.run_script(_write_bell_script(tmp_path), background=True)
    run_id = started["run_id"]

    status = _wait_for_status(run_id, "paused")
    assert status is not None
    assert status["hit"]["condition"]["gate"] == "cx"
    assert status["hit"]["event_summary"].startswith("cx")

    tools.resume_run(run_id)
    status = _wait_for_status(run_id, "finished")
    assert status is not None
    bell = next(t for t in status["timelines"] if t["name"] == "bell")
    assert bell["event_count"] > 0


def test_background_run_stop_while_paused_reports_stopped(tmp_path):
    """stop_run on a paused run cancels it cleanly, reported as status "stopped"."""
    tools.add_breakpoint({"kind": "gate", "gate": "cx"})
    started = tools.run_script(_write_bell_script(tmp_path), background=True)
    run_id = started["run_id"]

    assert _wait_for_status(run_id, "paused") is not None
    tools.stop_run(run_id)
    status = _wait_for_status(run_id, "stopped")
    assert status is not None


def test_background_run_rejects_second_concurrent_run(tmp_path):
    """Only one background run may be in flight at a time on the shared session."""
    tools.add_breakpoint({"kind": "gate", "gate": "cx"})
    started = tools.run_script(_write_bell_script(tmp_path), background=True)
    run_id = started["run_id"]
    assert _wait_for_status(run_id, "paused") is not None

    with pytest.raises(ValueError, match="already in flight"):
        tools.run_script(_write_bell_script(tmp_path), background=True)

    tools.resume_run(run_id)
    _wait_while_paused(run_id)


def test_poll_run_unknown_run_id_raises():
    """poll_run/resume_run/stop_run on a never-registered run_id raise, not silently no-op."""
    with pytest.raises(ValueError, match="no background run"):
        tools.poll_run("does-not-exist")
    with pytest.raises(ValueError, match="no background run"):
        tools.resume_run("does-not-exist")
    with pytest.raises(ValueError, match="no background run"):
        tools.stop_run("does-not-exist")
