"""Public entry point for exploring QDebug timelines.

Deliberately decoupled from the instrumentation layers: it only reads the
public fields of :class:`~qdebug._events.Timeline`, so either side can be
replaced independently. Presentation itself is delegated to whichever
:class:`~qdebug._frontends.TimelineFrontend` fits the running environment
(Jupyter widget, native GUI window, or text-based CLI) — see
:func:`qdebug._frontends.get_frontend`.
"""

from __future__ import annotations

from typing import Any, Iterable

from ._events import Timeline
from ._frontends import get_frontend
from ._frontends._render import format_statevector, render_bloch, render_histogram

__all__ = ["show", "format_statevector", "render_histogram", "render_bloch"]


def show(timelines: Timeline | Iterable[Timeline], frontend: str | None = None) -> Any:
    """Display one or more timelines using the best available frontend.

    Accepts a single :class:`Timeline` or several; events from multiple
    timelines are merged into one chronological sequence, with the circuit
    panel showing each circuit's most recently known state side by side.

    Auto-detects a Jupyter widget, a native GUI window, or a text-based CLI
    based on the running environment. Force a specific one with
    ``frontend="jupyter"|"gui"|"cli"``, or set the ``QDEBUG_FRONTEND``
    environment variable to always use the same one. Forcing a frontend that
    isn't actually available (e.g. ``"gui"`` with no display) raises
    ``RuntimeError``.
    """
    normalized = [timelines] if isinstance(timelines, Timeline) else list(timelines)
    return get_frontend(frontend).show(normalized)
