"""Version compatibility helpers.

Every hook resolves its patch targets through :func:`resolve` so that a
missing or renamed Qiskit internal degrades into a logged warning instead of
an ImportError or AttributeError at activation time. Patch bookkeeping itself
lives in :mod:`qisdebug._dispatcher`.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

logger = logging.getLogger("qisdebug")


def qiskit_version() -> str | None:
    """Return the installed Qiskit version string, or None if unavailable."""
    try:
        import qiskit

        return qiskit.__version__
    except Exception:
        return None


def resolve(module_path: str, attribute: str | None = None) -> Any | None:
    """Import ``module_path`` and optionally fetch ``attribute`` from it.

    Returns None instead of raising when the module or attribute is missing,
    so callers can treat absent Qiskit internals as a degraded layer rather
    than a fatal error.
    """
    try:
        module = importlib.import_module(module_path)
    except ImportError:
        return None
    if attribute is None:
        return module
    return getattr(module, attribute, None)
