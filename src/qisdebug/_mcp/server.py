"""MCP stdio server exposing QisDebug telemetry to coding agents.

Wraps every function in :mod:`qisdebug._mcp.tools` as an MCP tool. All the
actual logic -- timeline lookup, rendering, statevector/counts comparison,
anomaly scanning, watch evaluation -- lives in :mod:`qisdebug._mcp.tools` and
the modules it delegates to; this file only registers those functions with
the protocol layer, the same role :mod:`qisdebug._frontends.cli` and
:mod:`qisdebug._frontends.gui` play for human-facing presentation.

Run directly (``python -m qisdebug._mcp.server``, or the ``qisdebug-mcp``
console script installed with the ``mcp`` extra) to serve over stdio, the
transport coding agents expect.
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from . import tools

server = MCPServer(
    "qisdebug",
    instructions=(
        "QisDebug records everything that happens to Qiskit circuits at runtime -- every "
        "gate, transpiler pass, mid-execution simulator statevector, and measurement "
        "result -- so bugs can be diagnosed from what actually happened instead of "
        "from reading code. Typical debugging workflow: "
        "(1) run_script executes a target script under instrumentation and returns each "
        "circuit's id, registers, and an anomaly scan; to focus on a suspect region, pass "
        "auto_activate=false and have the script create its own qisdebug.QisDebug(), call "
        "start()/end() around the region, and export or print what it captured, since that "
        "instance's timelines are separate from this server's own. "
        "(2) add_watch pins qubits/clbits/registers of interest; read_watches reports "
        "their gate history, P(1), and Bloch vector at any sequence point. "
        "(3) get_events, get_state_snapshot, and get_circuit_text extract the runtime "
        "state between any two gates; get_state_snapshot and compare_statevector take "
        "either step (position in the instrumented replay) or seq (a gate_added event's "
        "seq from get_events, for the state right after that gate) -- use seq to avoid "
        "converting between the two numberings by hand. transpile_pass events carry a "
        "property_changes diff (added/removed/changed keys) against the previous pass; "
        "get_property_set(circuit_id, seq) returns the full property_set snapshot for one "
        "pass, for inspecting layout/routing decisions a specific pass made. "
        "(4) You supply what the program should produce: compare_statevector and "
        "compare_counts check your expected state or distribution against the capture "
        "and report exactly where they diverge. compare_statevector's expected argument "
        'is sparse -- {"01": [re, im], ...} keyed by binary label (qubit 0 least '
        "significant), only the basis states you have an expectation about, everything "
        "else assumed zero -- rather than every 2**num_qubits amplitude. find_anomalies "
        "is expectation-free and "
        "only catches impossible states (non-normalized/non-finite statevectors, failed "
        "runs); a clean scan does not mean the program is correct. diff_timelines compares "
        "two points in time directly instead of against a hand-supplied expectation -- "
        "gate-level alignment plus fidelity/TVD/depth deltas. Its optional seq_a/seq_b (from "
        "get_events) bound each side to a specific moment: pass the same circuit_id for both "
        "with different seqs to see what changed between two points in one circuit's own "
        "replay, or two different circuit_ids to compare two circuits -- omit either seq for "
        "that side's final state, useful for confirming a refactor is equivalent or spotting "
        "where two versions of a circuit diverge. "
        "(5) save_timeline persists the evidence to a JSON file. "
        "set_configuration tunes capture (e.g. max_statevector_qubits, "
        "transpile_snapshots='all') before a run. "
        "(6) add_breakpoint pauses a live run the instant a condition matches -- a gate "
        "name/qubit match, a seq threshold, a watched qubit/clbit/register's value crossing "
        "a numeric threshold, or the next event find_anomalies would flag. Live pausing "
        "needs run_script(background=True), which returns a run_id immediately instead of "
        "waiting for the script to finish; poll_run(run_id) reports running/paused/finished "
        "and, when paused, which breakpoint hit and on what event; resume_run continues "
        "execution; stop_run cancels (best-effort: only guaranteed to work while paused, "
        "since a live Python thread can't be forcibly killed while it's actually running)."
    ),
)

for _tool in (
    tools.run_script,
    tools.poll_run,
    tools.resume_run,
    tools.stop_run,
    tools.list_timelines,
    tools.get_timeline_summary,
    tools.get_timeline,
    tools.get_events,
    tools.get_circuit_text,
    tools.get_property_set,
    tools.get_state_snapshot,
    tools.add_watch,
    tools.remove_watch,
    tools.list_watches,
    tools.read_watches,
    tools.add_breakpoint,
    tools.remove_breakpoint,
    tools.list_breakpoints,
    tools.compare_statevector,
    tools.compare_counts,
    tools.diff_timelines,
    tools.find_anomalies,
    tools.save_timeline,
    tools.get_configuration,
    tools.set_configuration,
    tools.clear,
):
    server.tool()(_tool)


def main() -> None:
    """Entry point for the ``qisdebug-mcp`` console script: serve over stdio."""
    server.run()


if __name__ == "__main__":
    main()
