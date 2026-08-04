"""MCP stdio server exposing QDebug telemetry to coding agents.

Wraps every function in :mod:`qdebug._mcp.tools` as an MCP tool. All the
actual logic -- timeline lookup, rendering, statevector/counts comparison,
anomaly scanning, watch evaluation -- lives in :mod:`qdebug._mcp.tools` and
the modules it delegates to; this file only registers those functions with
the protocol layer, the same role :mod:`qdebug._frontends.cli` and
:mod:`qdebug._frontends.gui` play for human-facing presentation.

Run directly (``python -m qdebug._mcp.server``, or the ``qdebug-mcp``
console script installed with the ``mcp`` extra) to serve over stdio, the
transport coding agents expect.
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from . import tools

server = MCPServer(
    "qdebug",
    instructions=(
        "QDebug records everything that happens to Qiskit circuits at runtime -- every "
        "gate, transpiler pass, mid-execution simulator statevector, and measurement "
        "result -- so bugs can be diagnosed from what actually happened instead of "
        "from reading code. Typical debugging workflow: "
        "(1) run_script executes a target script under instrumentation and returns each "
        "circuit's id, registers, and an anomaly scan; to focus on a suspect region, "
        "first edit the script to place qdebug.activate()/qdebug.deactivate() around it "
        "and pass auto_activate=false. "
        "(2) add_watch pins qubits/clbits/registers of interest; read_watches reports "
        "their gate history, P(1), and Bloch vector at any sequence point. "
        "(3) get_events, get_state_snapshot, and get_circuit_text extract the runtime "
        "state between any two gates; get_state_snapshot and compare_statevector take "
        "either step (position in the instrumented replay) or seq (a gate_added event's "
        "seq from get_events, for the state right after that gate) -- use seq to avoid "
        "converting between the two numberings by hand. "
        "(4) You supply what the program should produce: compare_statevector and "
        "compare_counts check your expected state or distribution against the capture "
        "and report exactly where they diverge. compare_statevector's expected argument "
        "is sparse -- {\"01\": [re, im], ...} keyed by binary label (qubit 0 least "
        "significant), only the basis states you have an expectation about, everything "
        "else assumed zero -- rather than every 2**num_qubits amplitude. find_anomalies "
        "is expectation-free and "
        "only catches impossible states (non-normalized/non-finite statevectors, failed "
        "runs); a clean scan does not mean the program is correct. "
        "(5) save_timeline persists the evidence to a JSON file. "
        "set_configuration tunes capture (e.g. max_statevector_qubits, "
        "transpile_snapshots='all') before a run."
    ),
)

for _tool in (
    tools.run_script,
    tools.list_timelines,
    tools.get_timeline_summary,
    tools.get_timeline,
    tools.get_events,
    tools.get_circuit_text,
    tools.get_state_snapshot,
    tools.add_watch,
    tools.remove_watch,
    tools.list_watches,
    tools.read_watches,
    tools.compare_statevector,
    tools.compare_counts,
    tools.find_anomalies,
    tools.save_timeline,
    tools.get_configuration,
    tools.set_configuration,
    tools.clear,
):
    server.tool()(_tool)


def main() -> None:
    """Entry point for the ``qdebug-mcp`` console script: serve over stdio."""
    server.run()


if __name__ == "__main__":
    main()
