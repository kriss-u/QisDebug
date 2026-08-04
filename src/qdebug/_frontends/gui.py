"""Native GUI frontend: a Qt window, used when a display is available.

Built on PySide6 rather than tkinter: PySide6's wheels bundle the Qt
binaries they need, so ``pip install "qdebug[gui]"`` is enough on its own,
no separate system Qt install required, and its layout manager (as opposed
to tkinter's ``pack``) keeps fixed-size widgets like the playback controls
from ever being squeezed out by a large circuit diagram or histogram image.
This is why the GUI frontend is behind the ``gui`` extra rather than always
available like the stdlib-only CLI and Jupyter frontends.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Callable, ClassVar

from .._events import TelemetryEvent, Timeline
from . import _render
from ._playback import PlaybackController
from .base import TimelineFrontend

DEFAULT_AUTOPLAY_INTERVAL_MS = 800
ACTIVE_TRACK_COLOR = "#fff3a0"


def gui_available() -> bool:
    """Return whether a native GUI window could actually be opened here."""
    if importlib.util.find_spec("PySide6") is None:
        return False
    if sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        return False
    try:
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
        if app.primaryScreen() is None:
            return False
    except Exception:
        return False
    return True


class QtFrontend(TimelineFrontend):
    """Opens a Qt window with a slider, play controls, and a diagram panel."""

    name: ClassVar[str] = "gui"

    def is_available(self) -> bool:
        return gui_available()

    def unavailable_reason(self) -> str:
        return (
            "no display is available for a native GUI window "
            "(PySide6 not installed -- pip install \"qdebug[gui]\" -- "
            "or DISPLAY/WAYLAND_DISPLAY unset)"
        )

    def show(self, timelines: list[Timeline]) -> None:
        if not self.is_available():
            raise RuntimeError(self.unavailable_reason())
        from PySide6.QtWidgets import QApplication

        merged, tracks, active_labels = _render.merge_timelines(timelines)
        names = ", ".join(str(t.name) for t in timelines)
        app = QApplication.instance() or QApplication([])
        window = _TimelineWindow(names, merged, tracks, active_labels)
        window.show()
        app.exec()


class _TimelineWindow:  # pylint: disable=too-many-instance-attributes
    """A single Qt window presenting one merged, navigable timeline.

    This class only does two things: lay out Qt widgets and translate Qt
    events into calls on a :class:`~._playback.PlaybackController`. It also
    implements the controller's ``Scheduler`` protocol via ``QTimer``. All
    play/pause/step/seek *logic* lives in the controller so a future toolkit
    could reuse it by writing a new thin window class around the same
    controller instead of re-deriving that state machine.

    The circuit diagram, event detail text, and any extra visual (histogram
    or Bloch sphere image) all live in one scrollable ``QTextEdit`` rather
    than separate widgets, so they share one scroll position and the
    playback controls below never compete with them for space -- Qt's
    layout gives the controls their fixed size first and the text edit
    whatever's left, instead of tkinter's ``pack`` letting oversized content
    push fixed-size widgets out of the window entirely.
    """

    def __init__(
        self,
        title: str,
        merged: list[tuple[Timeline, TelemetryEvent]],
        tracks: list[dict[str, str]],
        active_labels: list[str | None],
    ) -> None:
        from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

        self.tracks = tracks
        self.active_labels = active_labels
        self._syncing_slider = False
        self._timers: list[object] = []

        self.window = QWidget()
        self.window.setWindowTitle(f"QDebug timeline — {title}")
        self.window.resize(900, 650)
        self.window.setMinimumSize(600, 400)

        if not merged:
            layout = QVBoxLayout(self.window)
            layout.addWidget(QLabel("Timeline(s) have no events."))
            self.controller: PlaybackController | None = None
            return

        self.controller = PlaybackController(
            merged=merged,
            scheduler=self,
            get_interval_ms=self._interval_ms,
            on_render=self._render,
            on_play_state_changed=self._on_play_state_changed,
        )
        self._build_widgets()
        self._render()

    def show(self) -> None:
        self.window.show()

    # -- Scheduler protocol, used by PlaybackController for autoplay -----

    def schedule(self, delay_ms: int, callback: Callable[[], None]) -> object:
        from PySide6.QtCore import QTimer

        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(callback)
        timer.start(delay_ms)
        self._timers.append(timer)
        return timer

    def cancel(self, handle: object) -> None:
        handle.stop()  # type: ignore[attr-defined]

    # -- Layout ------------------------------------------------------------

    def _build_widgets(self) -> None:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import (
            QHBoxLayout,
            QLabel,
            QSlider,
            QSpinBox,
            QTextEdit,
            QVBoxLayout,
        )

        assert self.controller is not None
        controller = self.controller

        layout = QVBoxLayout(self.window)

        self.header = QLabel()
        self.header.setWordWrap(True)
        layout.addWidget(self.header)

        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.text.setFont(_monospace_font())
        layout.addWidget(self.text, stretch=1)

        controls = QHBoxLayout()
        controls.addWidget(_button("⏮ First", lambda: self._goto(0)))
        controls.addWidget(_button("◀ Prev", lambda: self._goto(controller.index - 1)))
        self.play_button = _button("Play", self._toggle_play)
        controls.addWidget(self.play_button)
        controls.addWidget(_button("Next ▶", lambda: self._goto(controller.index + 1)))
        controls.addWidget(_button("Last ⏭", lambda: self._goto(controller.last_index)))
        controls.addWidget(QLabel("ms/step"))
        self.interval_input = QSpinBox()
        self.interval_input.setRange(50, 10_000)
        self.interval_input.setSingleStep(50)
        self.interval_input.setValue(DEFAULT_AUTOPLAY_INTERVAL_MS)
        controls.addWidget(self.interval_input)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(1, len(controller.merged))
        self.slider.valueChanged.connect(self._on_slider)
        layout.addWidget(self.slider)

    # -- Widget <-> PlaybackController glue --------------------------------

    def _render(self) -> None:
        """Refresh every widget to reflect the controller's current event."""
        from PySide6.QtGui import QImage, QTextCursor

        assert self.controller is not None
        controller = self.controller

        source, event = controller.current
        index = controller.index
        self.header.setText(
            _render.escape(
                f"{index + 1}/{len(controller.merged)} — {source.name} — {event.type.value} "
                f"at {event.timestamp:.3f} — {event.describe()}"
            )
        )

        tracks_text, active_span = _render.render_tracks_text_with_active(
            self.tracks[index], self.active_labels[index]
        )
        self.text.clear()
        self.text.setPlainText(tracks_text + "\n\n" + _render.render_event(source, event))

        label, png_bytes = _render.render_extra_visual_bytes(event)
        if png_bytes:
            image = QImage.fromData(png_bytes)
            cursor = QTextCursor(self.text.document())
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.insertBlock()
            cursor.insertText(f"-- {label} --\n")
            cursor.insertImage(image)

        self._highlight_and_scroll(active_span)

        self._syncing_slider = True
        try:
            self.slider.setValue(index + 1)
        finally:
            self._syncing_slider = False

    def _highlight_and_scroll(self, active_span: tuple[int, int] | None) -> None:
        """Highlight the track section most relevant to the current event and
        scroll it into view, so the user doesn't have to hunt for what changed.
        """
        from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor

        if active_span is None:
            self.text.setExtraSelections([])
            return

        doc = self.text.document()
        start_line, end_line = active_span
        start_block = doc.findBlockByNumber(start_line - 1)
        end_block = doc.findBlockByNumber(end_line - 1)

        highlight_cursor = QTextCursor(doc)
        highlight_cursor.setPosition(start_block.position())
        highlight_cursor.setPosition(
            end_block.position() + end_block.length() - 1, QTextCursor.MoveMode.KeepAnchor
        )
        fmt = QTextCharFormat()
        fmt.setBackground(QColor(ACTIVE_TRACK_COLOR))
        selection = _extra_selection(highlight_cursor, fmt)
        self.text.setExtraSelections([selection])

        scroll_cursor = QTextCursor(doc)
        scroll_cursor.setPosition(start_block.position())
        self.text.setTextCursor(scroll_cursor)
        self.text.ensureCursorVisible()

    def _on_play_state_changed(self, playing: bool) -> None:
        self.play_button.setText("Pause" if playing else "Play")

    def _goto(self, index: int) -> None:
        assert self.controller is not None
        self.controller.goto(index)

    def _toggle_play(self) -> None:
        assert self.controller is not None
        self.controller.toggle_play()

    def _interval_ms(self) -> int:
        return self.interval_input.value()

    def _on_slider(self, value: int) -> None:
        if self._syncing_slider:
            return
        self._goto(value - 1)


def _button(text: str, on_click: Callable[[], None]) -> object:
    from PySide6.QtWidgets import QPushButton

    button = QPushButton(text)
    button.clicked.connect(on_click)
    return button


def _monospace_font() -> object:
    from PySide6.QtGui import QFontDatabase

    font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
    font.setPointSize(11)
    return font


def _extra_selection(cursor: object, fmt: object) -> object:
    from PySide6.QtWidgets import QTextEdit

    selection = QTextEdit.ExtraSelection()
    selection.cursor = cursor  # type: ignore[attr-defined]
    selection.format = fmt  # type: ignore[attr-defined]
    return selection
