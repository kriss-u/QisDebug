"""The ``qisdebug`` console script: Typer wiring only.

Argument parsing, help text, and ``--version`` live here; everything this
module's commands actually do is delegated to :mod:`.commands`, which has
no Typer/Click import of its own and is usable directly from Python. This
mirrors how :mod:`qisdebug._mcp.server` only registers
:mod:`qisdebug._mcp.tools` functions with the MCP protocol layer rather than
implementing anything itself.
"""

from __future__ import annotations

from typing import List, Optional

import typer

from .. import __version__
from .._events import Timeline
from .._frontends import get_frontend
from . import commands

app = typer.Typer(
    name="qisdebug",
    help="Zero-touch instrumentation and timeline debugging for Qiskit circuits.",
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"qisdebug {__version__}")
        raise typer.Exit()


@app.callback()
def _callback(
    # Unused directly: Typer/Click invokes _version_callback as this option
    # is parsed (is_eager=True), before the command body would run at all;
    # the parameter only needs to exist for Typer to wire the option to it.
    show_version: Optional[bool] = typer.Option(  # pylint: disable=unused-argument
        None,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Print the installed qisdebug version and exit.",
    ),
) -> None:
    """Zero-touch instrumentation and timeline debugging for Qiskit circuits."""


@app.command(context_settings={"ignore_unknown_options": True})
def run(
    script: str = typer.Argument(..., help="Path to the Python script to run."),
    args: Optional[List[str]] = typer.Argument(
        None, help="Arguments for the script; put -- before them if they look like flags."
    ),
    frontend: Optional[str] = typer.Option(
        None,
        "--frontend",
        help='Force a frontend ("gui" or "cli") instead of auto-detecting one.',
    ),
    circuit: Optional[List[str]] = typer.Option(
        None,
        "--circuit",
        help="Show only this circuit's timeline (repeatable). Omit to show everything captured.",
    ),
) -> None:
    """Run SCRIPT under a QisDebug instrumentation session, then show what was captured.

    Brackets the whole script with start()/end() the way a hand-written
    ``qisdebug.QisDebug()`` session would, so every circuit the script
    builds, transpiles, or runs is recorded automatically -- no changes to
    the script itself are needed. A script that raises is reported as a
    failed CLI run (nonzero exit, traceback printed) rather than silently
    losing whatever it captured before the error.
    """
    result = commands.run_script(script, args)
    targets = _select_targets(result.timelines, circuit)
    if targets:
        get_frontend(frontend).show(targets)
    elif not result.timelines:
        typer.echo("no circuits were tracked", err=True)
    if result.error is not None:
        typer.echo(f"script raised: {result.error}", err=True)
        raise typer.Exit(code=1)


def _select_targets(
    timelines: List[Timeline], names: Optional[List[str]]
) -> List[Timeline]:
    """Narrow ``timelines`` to ``names`` (repeated ``--circuit``), warning
    about any name that wasn't tracked; ``names`` empty/None means "show
    everything captured"."""
    if not names:
        return timelines
    by_name = {t.name: t for t in timelines if t.name}
    for name in names:
        if name not in by_name:
            typer.echo(f"warning: no tracked circuit named {name!r}", err=True)
    return [by_name[name] for name in names if name in by_name]


@app.command()
def version() -> None:
    """Print qisdebug/Qiskit versions, which instrumentation layers attach, and installed extras."""
    info = commands.collect_version_info()
    typer.echo(f"qisdebug {info.qisdebug_version}")
    typer.echo(f"qiskit {info.qiskit_version or '(not installed)'}")
    typer.echo("instrumentation layers:")
    for name, attached in info.activation.layers.items():
        typer.echo(f"  {name}: {'yes' if attached else 'no'}")
    typer.echo("optional extras:")
    for name, available in info.extras.items():
        typer.echo(f"  {name}: {'installed' if available else 'not installed'}")
    for warning in info.activation.warnings:
        typer.echo(f"warning: {warning}", err=True)


def main() -> None:
    """Entry point for the ``qisdebug`` console script."""
    app()


if __name__ == "__main__":
    main()
