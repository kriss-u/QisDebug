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

from .base import TimelineFrontend
from .gui import gui_available

ENV_VAR = "QDEBUG_FRONTEND"


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


def _frontends() -> dict[str, type[TimelineFrontend]]:
    from .cli import CLIFrontend
    from .gui import QtFrontend
    from .jupyter import JupyterFrontend

    return {"jupyter": JupyterFrontend, "gui": QtFrontend, "cli": CLIFrontend}


def get_frontend(frontend: str | None = None) -> TimelineFrontend:
    """Resolve a :class:`TimelineFrontend` by name, env var, or auto-detection.

    Precedence: the ``frontend`` argument, then the ``QDEBUG_FRONTEND``
    environment variable, then :func:`detect_environment`. A name that
    doesn't match a known frontend raises ``ValueError``; a forced frontend
    that isn't actually available raises ``RuntimeError`` explaining why.
    """
    forced = frontend or os.environ.get(ENV_VAR)
    registry = _frontends()
    if forced:
        key = forced.strip().lower()
        if key not in registry:
            raise ValueError(
                f"Unknown qdebug frontend {forced!r}; choose from {sorted(registry)}"
            )
        instance = registry[key]()
        if not instance.is_available():
            raise RuntimeError(
                f"qdebug frontend {key!r} was forced but is not available: "
                f"{instance.unavailable_reason()}"
            )
        return instance
    return registry[detect_environment()]()
