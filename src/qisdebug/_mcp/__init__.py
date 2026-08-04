"""Agent-facing access to QisDebug telemetry, via MCP.

:mod:`qisdebug._mcp.tools` holds the plain, protocol-free query functions;
:mod:`qisdebug._mcp.server` wraps them as MCP tools. Only the latter imports
the ``mcp`` package, so importing this package (or ``tools`` directly, as
the test suite does) never requires the ``mcp`` extra to be installed.
"""
