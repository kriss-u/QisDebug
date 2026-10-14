"""Agent-facing access to QisDebug telemetry, via MCP.

:mod:`qisdebug._mcp.tools` holds the plain, protocol-free query functions;
:mod:`qisdebug._mcp.server` wraps them as MCP tools. Only the latter imports
the ``mcp`` package, so importing this package (or ``tools`` directly, as
the test suite does) never requires the ``mcp`` extra to be installed.
"""

from __future__ import annotations

import sys

from .._compat import install_hint


def main() -> None:
    """Entry point for the ``qisdebug-mcp`` console script.

    The script is installed even without the ``mcp`` extra (pip and conda
    both ship every console script), so the server module is imported here,
    lazily, to turn a missing ``mcp`` package into an install hint and a
    clean exit instead of a traceback.
    """
    try:
        from .server import main as serve
    except ModuleNotFoundError as exc:
        if exc.name is None or exc.name.split(".")[0] != "mcp":
            raise
        print(
            "qisdebug-mcp needs the MCP SDK, which is not installed: "
            + install_hint("mcp", "mcp"),
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    serve()
