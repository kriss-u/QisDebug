"""MCP stdio server exposing QDebug telemetry to coding agents.

Wraps every function in :mod:`qdebug._mcp.tools` as an MCP tool. All the
actual logic -- timeline lookup, rendering, statevector/counts comparison,
anomaly detection -- lives in :mod:`qdebug._mcp.tools` and the modules it
delegates to; this file only registers those functions with the protocol
layer, the same role :mod:`qdebug._frontends.cli` and
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
        "Tools for inspecting QDebug telemetry captured from Qiskit circuits: "
        "construction, transpilation and execution history, state snapshots, and "
        "comparisons against an expected statevector or count distribution. Call "
        "run_script to execute a target script under instrumentation and get an "
        "immediate anomaly scan, then use the other tools to dig into what "
        "happened instead of re-deriving it by reading the script's printed output."
    ),
)

for _tool in (
    tools.run_script,
    tools.list_timelines,
    tools.get_timeline_summary,
    tools.get_timeline,
    tools.get_circuit_text,
    tools.get_state_snapshot,
    tools.compare_statevector,
    tools.compare_counts,
    tools.find_anomalies,
    tools.clear,
):
    server.tool()(_tool)


def main() -> None:
    """Entry point for the ``qdebug-mcp`` console script: serve over stdio."""
    server.run()


if __name__ == "__main__":
    main()
