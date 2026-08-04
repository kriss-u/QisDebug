"""Display probing that must not require PySide6 to be installed.

Separate from ``gui.py`` so :func:`gui_available` can be imported by
``detect.py`` -- to decide whether the GUI frontend is even a candidate --
without pulling in PySide6. ``gui.py`` itself imports PySide6
unconditionally at module scope, since it's only ever imported once a
caller has already committed to the GUI frontend.
"""

from __future__ import annotations

import importlib.util
import os
import sys


def gui_available() -> bool:
    """Return whether a native GUI window could actually be opened here."""
    if importlib.util.find_spec("PySide6") is None:
        return False
    if sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        return False
    try:
        from PySide6.QtGui import QGuiApplication
        from PySide6.QtWidgets import QApplication

        _app = QApplication.instance() or QApplication([])
        if QGuiApplication.primaryScreen() is None:
            return False
    except Exception:
        return False
    return True
