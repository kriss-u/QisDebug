"""Display probing that must not require ``pywebview`` to be installed.

Separate from ``gui.py`` so :func:`gui_available` can be imported by
``detect.py`` -- to decide whether the GUI frontend is even a candidate --
without pulling in ``pywebview``. ``gui.py`` itself imports ``webview``
lazily inside :meth:`~.gui.WebviewFrontend.show`, since it's only ever
needed once a caller has already committed to the GUI frontend.
"""

from __future__ import annotations

import importlib.util
import os
import sys

from .._compat import install_hint


def gui_unavailable_reason() -> str | None:
    """Return why a native GUI window can't be opened here, or None if it can."""
    if importlib.util.find_spec("webview") is None:
        return f"pywebview is not installed ({install_hint('gui', 'pywebview')})"
    if sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        return "no display is available (DISPLAY and WAYLAND_DISPLAY are unset)"
    return None


def gui_available() -> bool:
    """Return whether a native GUI window could actually be opened here."""
    return gui_unavailable_reason() is None
