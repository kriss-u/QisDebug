"""Environment detection and the frontend factory.

Picks, in order: a Jupyter widget when running inside a Jupyter kernel, a
native GUI window when a display is available, and a text-based CLI as the
universal fallback. Any of the three can be forced explicitly, either via
the ``frontend`` argument or the ``QDEBUG_FRONTEND`` environment variable;
forcing one that is not actually available raises ``RuntimeError`` instead
of silently falling back or hanging.
"""

from __future__ import annotations

import os
from typing import Literal

from ._display import gui_available
from .base import TimelineFrontend

ENV_VAR = "QDEBUG_FRONTEND"
FRONTEND_NAMES = ("jupyter", "gui", "cli")


def in_jupyter() -> bool:
    """Return whether we're running inside a Jupyter (ZMQ) kernel.

    IPython is an optional, soft dependency (only present when qdebug is
    used from a notebook), so this import is guarded at runtime. Imported
    from ``IPython.core.getipython`` rather than the top-level ``IPython``
    package: the latter re-exports ``get_ipython`` without listing it in
    ``__all__``, which static checkers don't treat as a public re-export.
    """
    try:
        from IPython.core.getipython import get_ipython

        shell = get_ipython()
    except ImportError:
        return False
    return shell is not None and type(shell).__name__ == "ZMQInteractiveShell"


def detect_environment() -> Literal["jupyter", "gui", "cli"]:
    """Auto-detect the best frontend for the current environment."""
    if in_jupyter():
        return "jupyter"
    if gui_available():
        return "gui"
    return "cli"


def _load_frontend(name: str) -> TimelineFrontend:
    """Import and instantiate the frontend for ``name``.

    Each branch imports its module lazily so, for example, forcing ``"cli"``
    never touches ``gui.py`` -- which imports PySide6 unconditionally at
    module scope -- on a machine that doesn't have the ``gui`` extra
    installed.
    """
    if name == "jupyter":
        from .jupyter import JupyterFrontend

        return JupyterFrontend()
    if name == "gui":
        from .gui import QtFrontend

        return QtFrontend()
    from .cli import CLIFrontend

    return CLIFrontend()


def get_frontend(frontend: str | None = None) -> TimelineFrontend:
    """Resolve a :class:`TimelineFrontend` by name, env var, or auto-detection.

    Precedence: the ``frontend`` argument, then the ``QDEBUG_FRONTEND``
    environment variable, then :func:`detect_environment`. A name that
    doesn't match a known frontend raises ``ValueError``; a forced frontend
    that isn't actually available raises ``RuntimeError`` explaining why.
    """
    forced = frontend or os.environ.get(ENV_VAR)
    if forced:
        key = forced.strip().lower()
        if key not in FRONTEND_NAMES:
            raise ValueError(
                f"Unknown qdebug frontend {forced!r}; choose from {sorted(FRONTEND_NAMES)}"
            )
        instance = _load_frontend(key)
        if not instance.is_available():
            raise RuntimeError(
                f"qdebug frontend {key!r} was forced but is not available: "
                f"{instance.unavailable_reason()}"
            )
        return instance
    return _load_frontend(detect_environment())
