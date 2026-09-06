"""Logic behind the ``qisdebug`` console script, kept free of Typer/Click.

Plain functions with ordinary Python types, importable and testable without
going through the CLI framework at all -- the same split
:mod:`qisdebug._mcp.tools`/:mod:`qisdebug._mcp.server` draws between agent
logic and the protocol layer that exposes it. :mod:`._app` only parses
arguments and prints the result of calling into this module.
"""

from __future__ import annotations

import runpy
import sys
from dataclasses import dataclass, field

from .._compat import qiskit_version
from .._debug import ActivationReport, QisDebug
from .._events import Timeline


@dataclass
class ScriptRunResult:
    """What :func:`run_script` observed: the timelines captured, and the
    exception a failing script raised, if any (re-raised by the CLI layer
    rather than swallowed -- a script that errors should fail the CLI run
    with a nonzero exit code, not report success)."""

    timelines: list[Timeline]
    error: BaseException | None = None
    active_names: list[str] = field(default_factory=list)


def run_script(path: str, argv: list[str] | None = None) -> ScriptRunResult:
    """Run ``path`` as ``__main__`` under a fresh :class:`QisDebug` session.

    Mirrors :func:`qisdebug._mcp.tools.run_script`'s ``auto_activate=True``,
    synchronous path: this call's own session brackets the whole script
    (``start()`` before, ``end()`` after, even if the script raises), and
    every circuit the script touches is captured under it. ``argv`` becomes
    ``sys.argv[1:]`` for the duration of the run, restored afterward
    regardless of outcome.
    """
    debug = QisDebug()
    report = debug.start()
    saved_argv = sys.argv
    sys.argv = [path, *(argv or [])]
    error: BaseException | None = None
    try:
        runpy.run_path(path, run_name="__main__")
    except BaseException as exc:  # noqa: BLE001 - reported to the caller, not swallowed
        error = exc
    finally:
        sys.argv = saved_argv
        debug.end()
    return ScriptRunResult(
        timelines=debug.timelines(),
        error=error,
        active_names=[name for name, ok in report.layers.items() if ok],
    )


@dataclass
class VersionInfo:
    """A snapshot of what this install of qisdebug can actually do,
    determined by a real ``start()``/``end()`` cycle rather than guessed
    from what's importable -- matching :meth:`QisDebug.start`'s own
    per-layer degrade-with-a-warning behavior, so this report can never
    claim a layer is available when activation would have skipped it."""

    qisdebug_version: str
    qiskit_version: str | None
    activation: ActivationReport
    extras: dict[str, bool]


def _extra_available(module_path: str) -> bool:
    try:
        __import__(module_path)
    except ImportError:
        return False
    return True


def collect_version_info() -> VersionInfo:
    """Gather version/capability info for ``qisdebug version``."""
    from .. import __version__

    debug = QisDebug()
    activation = debug.start()
    debug.end()
    extras = {
        "aer": _extra_available("qiskit_aer"),
        "gui": _extra_available("PySide6"),
        "mcp": _extra_available("mcp"),
    }
    return VersionInfo(
        qisdebug_version=__version__,
        qiskit_version=qiskit_version(),
        activation=activation,
        extras=extras,
    )
