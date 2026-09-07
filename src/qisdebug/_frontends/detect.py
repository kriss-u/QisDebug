"""Environment detection and the frontend factory.

Picks, in order: a native GUI window when a display is available, and a
text-based CLI as the universal fallback (including inside a notebook, where
CLI output renders fine via print). Either can be forced explicitly, via the
``frontend`` argument or the ``QISDEBUG_FRONTEND`` environment variable;
forcing one that is not actually available raises ``RuntimeError`` instead
of silently falling back or hanging.
"""

from __future__ import annotations

import os
from typing import Literal

from ._display import gui_available
from .base import TimelineFrontend

ENV_VAR = "QISDEBUG_FRONTEND"
FRONTEND_NAMES = ("gui", "cli")


def detect_environment() -> Literal["gui", "cli"]:
    """Auto-detect the best frontend for the current environment."""
    if gui_available():
        return "gui"
    return "cli"


def _load_frontend(name: str) -> TimelineFrontend:
    """Import and instantiate the frontend for ``name``.

    Each branch imports its module lazily so, for example, forcing ``"cli"``
    never touches ``gui.py`` on a machine that doesn't have the ``gui``
    extra installed.
    """
    if name == "gui":
        from .gui import WebviewFrontend

        return WebviewFrontend()
    from .cli import CLIFrontend

    return CLIFrontend()


def get_frontend(frontend: str | None = None) -> TimelineFrontend:
    """Resolve a :class:`TimelineFrontend` by name, env var, or auto-detection.

    Precedence: the ``frontend`` argument, then the ``QISDEBUG_FRONTEND``
    environment variable, then :func:`detect_environment`. A name that
    doesn't match a known frontend raises ``ValueError``; a forced frontend
    that isn't actually available raises ``RuntimeError`` explaining why.
    """
    forced = frontend or os.environ.get(ENV_VAR)
    if forced:
        key = forced.strip().lower()
        if key not in FRONTEND_NAMES:
            raise ValueError(
                f"Unknown qisdebug frontend {forced!r}; choose from {sorted(FRONTEND_NAMES)}"
            )
        instance = _load_frontend(key)
        if not instance.is_available():
            raise RuntimeError(
                f"qisdebug frontend {key!r} was forced but is not available: "
                f"{instance.unavailable_reason()}"
            )
        return instance
    return _load_frontend(detect_environment())
