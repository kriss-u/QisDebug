"""Native GUI frontend: a single OS-native window rendering local HTML/CSS.

Built on ``pywebview`` rather than a widget toolkit: pywebview wraps
whichever webview the OS already ships (WKWebView on macOS, WebView2 on
Windows, WebKitGTK on Linux), so ``pip install "qisdebug[gui]"`` is enough
on its own -- no bundled Chromium, no separate system Qt install. The
window renders local HTML/CSS/JS handed to it in-process
(:func:`webview.create_window`'s ``html=`` argument); there is no server
process and no browser tab involved anywhere, matching the project's rule
against a webserver/browser-based viewer while still getting CSS's
typography and layout power for the timeline view. All state and
computation live in :class:`~.bridge.TimelineBridge` (toolkit-free, plain
Python); this module only starts the window and exposes the bridge as the
page's ``js_api``.
"""

from __future__ import annotations

import importlib.resources
from typing import ClassVar

from .._events import Timeline
from . import _render
from ._display import gui_available, gui_unavailable_reason
from .base import TimelineFrontend
from .bridge import TimelineBridge

_ASSETS = importlib.resources.files(__package__) / "_web"


def _read_asset(name: str) -> str:
    return (_ASSETS / name).read_text(encoding="utf-8")


class WebviewFrontend(TimelineFrontend):
    """Opens a native window presenting one merged timeline as HTML/CSS."""

    name: ClassVar[str] = "gui"

    def is_available(self) -> bool:
        return gui_available()

    def unavailable_reason(self) -> str:
        reason = gui_unavailable_reason()
        return f"a native GUI window can't be opened: {reason or 'no display is available'}"

    def show(self, timelines: list[Timeline]) -> None:
        if not self.is_available():
            raise RuntimeError(self.unavailable_reason())

        import webview

        merged, tracks, active_labels = _render.merge_timelines(timelines)
        names = ", ".join(str(t.name) for t in timelines)
        timelines_by_id = {t.circuit_id: t for t in timelines}

        if not merged:
            webview.create_window(
                f"QisDebug timeline — {names}",
                html="<body><p>Timeline(s) have no events.</p></body>",
            )
            webview.start()
            return

        bridge = TimelineBridge(
            title=names,
            merged=merged,
            tracks=tracks,
            active_labels=active_labels,
            timelines_by_id=timelines_by_id,
        )
        html = _read_asset("index.html").replace(
            "/*__STYLE__*/", _read_asset("style.css")
        ).replace("//__SCRIPT__//", _read_asset("app.js"))
        webview.create_window(
            f"QisDebug timeline — {names}",
            html=html,
            js_api=bridge,
            width=1280,
            height=860,
            min_size=(800, 500),
        )
        webview.start()
