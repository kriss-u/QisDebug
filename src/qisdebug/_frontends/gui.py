"""Native GUI frontend: a Qt window, used when a display is available.

Built on PySide6 rather than tkinter: PySide6's wheels bundle the Qt
binaries they need, so ``pip install "qisdebug[gui]"`` is enough on its own,
no separate system Qt install required, and its layout manager (as opposed
to tkinter's ``pack``) keeps fixed-size widgets like the playback controls
from ever being squeezed out by a large circuit diagram or histogram image.
This is why the GUI frontend is behind the ``gui`` extra rather than always
available like the stdlib-only CLI frontend.
"""

from __future__ import annotations

from typing import Callable, ClassVar

from PySide6.QtCore import QEvent, QObject, QRectF, Qt, QTimer
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QImage,
    QKeyEvent,
    QPainter,
    QPalette,
    QPen,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSlider,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .._events import TelemetryEvent, Timeline
from . import _render
from ._display import gui_available
from ._playback import PlaybackController, PlaybackHooks, Scheduler
from .base import TimelineFrontend

DEFAULT_AUTOPLAY_INTERVAL_MS = 800
ACTIVE_TRACK_BORDER_COLOR_LIGHT = "#1d4ed8"
ACTIVE_TRACK_BORDER_COLOR_DARK = "#60a5fa"
ACTIVE_TRACK_BORDER_WIDTH = 1.5
ACTIVE_TRACK_PADDING_X = 4
ACTIVE_TRACK_PADDING_Y = 3
MAX_IMAGE_WIDTH = 480
MAX_IMAGE_HEIGHT = 360
_PLAY_LABEL = "⏵ Play  (Space)"
_PAUSE_LABEL = "⏸ Pause  (Space)"


class QtFrontend(TimelineFrontend):
    """Opens a Qt window with a slider, play controls, and a diagram panel."""

    name: ClassVar[str] = "gui"

    def is_available(self) -> bool:
        return gui_available()

    def unavailable_reason(self) -> str:
        return (
            "no display is available for a native GUI window "
            "(PySide6 not installed -- pip install \"qisdebug[gui]\" -- "
            "or DISPLAY/WAYLAND_DISPLAY unset)"
        )

    def show(self, timelines: list[Timeline]) -> None:
        if not self.is_available():
            raise RuntimeError(self.unavailable_reason())

        merged, tracks, active_labels = _render.merge_timelines(timelines)
        names = ", ".join(str(t.name) for t in timelines)
        timelines_by_id = {t.circuit_id: t for t in timelines}
        app = QApplication.instance() or QApplication([])
        window = _TimelineWindow(names, merged, tracks, active_labels, timelines_by_id)
        window.show()
        app.exec()


class _TrackTextEdit(QTextEdit):
    """A ``QTextEdit`` that can additionally outline a span of lines with a
    border, rather than the flat background fill ``ExtraSelection`` is
    limited to -- a border reads clearly on both light and dark palettes and
    never tints the circuit characters it surrounds.
    """

    def __init__(self) -> None:
        super().__init__()
        self._active_span: tuple[int, int] | None = None

    def set_active_span(self, span: tuple[int, int] | None) -> None:
        """Outline the 1-indexed (start_line, end_line) span, or clear it if ``None``."""
        self._active_span = span
        self.viewport().update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override name)
        """Paint normally, then outline the active span if one is set."""
        super().paintEvent(event)
        if self._active_span is None:
            return

        doc = self.document()
        start_line, end_line = self._active_span
        start_block = doc.findBlockByNumber(start_line - 1)
        end_block = doc.findBlockByNumber(end_line - 1)

        start_cursor = QTextCursor(doc)
        start_cursor.setPosition(start_block.position())
        end_cursor = QTextCursor(doc)
        end_cursor.setPosition(end_block.position() + max(end_block.length() - 1, 0))

        top = self.cursorRect(start_cursor).top() - ACTIVE_TRACK_PADDING_Y
        bottom = self.cursorRect(end_cursor).bottom() + ACTIVE_TRACK_PADDING_Y
        rect = QRectF(
            ACTIVE_TRACK_PADDING_X,
            top,
            self.viewport().width() - 2 * ACTIVE_TRACK_PADDING_X,
            bottom - top,
        )

        painter = QPainter(self.viewport())
        pen = QPen(QColor(_active_track_border_color(self)))
        pen.setWidthF(ACTIVE_TRACK_BORDER_WIDTH)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(rect, 3, 3)


def _active_track_border_color(widget: QWidget) -> str:
    """Pick a border color with enough contrast for the widget's current
    palette, since a single fixed color cannot read well against both a
    near-white and a near-black background.
    """
    background = widget.palette().color(QPalette.ColorRole.Base)
    is_dark = background.lightness() < 128
    return ACTIVE_TRACK_BORDER_COLOR_DARK if is_dark else ACTIVE_TRACK_BORDER_COLOR_LIGHT


class _TimelineWidget(QWidget):
    """The top-level window widget.

    Only exists to run ``on_close`` from :meth:`closeEvent`, so the window
    can unregister its playback key filter from the shared ``QApplication``
    instead of leaking it onto whatever window a later ``show()`` call in
    the same process opens.
    """

    def __init__(self, on_close: Callable[[], None]) -> None:
        super().__init__()
        self._on_close = on_close

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override name)
        """Run the registered cleanup, then close normally."""
        self._on_close()
        super().closeEvent(event)


class _PlaybackKeyFilter(QObject):
    """Application-wide key filter for the cassette-style playback shortcuts.

    Installed on the ``QApplication`` rather than as a per-widget shortcut,
    so left/right/space navigate the timeline no matter which control (the
    slider, the interval spin box, the text panel) currently has focus --
    those widgets would otherwise consume the same keys themselves before a
    plain per-widget shortcut ever got a chance to fire.
    """

    def __init__(self, handler: Callable[[int], bool]) -> None:
        super().__init__()
        self._handler = handler

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        """Consume Left/Right/Space before the focused widget sees them."""
        if isinstance(event, QKeyEvent) and event.type() == QEvent.Type.KeyPress:
            if self._handler(event.key()):
                return True
        return super().eventFilter(watched, event)


class _TimelineWindow(Scheduler[QTimer]):  # pylint: disable=too-many-instance-attributes
    """A single Qt window presenting one merged, navigable timeline.

    This class only does two things: lay out Qt widgets and translate Qt
    events into calls on a :class:`~._playback.PlaybackController`. It also
    implements the controller's ``Scheduler`` interface via ``QTimer``. All
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
        timelines_by_id: dict[str, Timeline],
    ) -> None:
        self.tracks = tracks
        self.active_labels = active_labels
        self.timelines_by_id = timelines_by_id
        self.watches: list[_render.WatchTarget] = []
        self._syncing_slider = False
        self._timers: list[QTimer] = []
        self._key_filter: _PlaybackKeyFilter | None = None

        self.window = _TimelineWidget(self._cleanup_key_filter)
        self.window.setWindowTitle(f"QisDebug timeline — {title}")
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
            hooks=PlaybackHooks(
                get_interval_ms=self._interval_ms,
                on_render=self._render,
                on_play_state_changed=self._on_play_state_changed,
            ),
        )
        layout = self._build_widgets()
        self._build_watch_widgets(layout)
        self._render()

        app = QApplication.instance()
        if app is not None:
            self._key_filter = _PlaybackKeyFilter(self._handle_shortcut)
            app.installEventFilter(self._key_filter)

    def _cleanup_key_filter(self) -> None:
        """Unregister the playback key filter so it doesn't outlive this window."""
        app = QApplication.instance()
        if app is not None and self._key_filter is not None:
            app.removeEventFilter(self._key_filter)
        self._key_filter = None

    def _handle_shortcut(self, key: int) -> bool:
        """Handle a global playback shortcut key, returning whether it was consumed."""
        if self.controller is None:
            return False
        if key == Qt.Key.Key_Left:
            self._goto(self.controller.index - 1)
            return True
        if key == Qt.Key.Key_Right:
            self._goto(self.controller.index + 1)
            return True
        if key == Qt.Key.Key_Space:
            self._toggle_play()
            return True
        return False

    def show(self) -> None:
        """Show the underlying Qt window."""
        self.window.show()

    # -- Scheduler protocol, used by PlaybackController for autoplay -----

    def schedule(self, delay_ms: int, callback: Callable[[], None]) -> QTimer:
        """Fire ``callback`` once after ``delay_ms``, returning the timer."""
        timer = QTimer()
        timer.setSingleShot(True)
        # pylint can't resolve Signal/SignalInstance attributes on Qt's C
        # extension classes (they're bound dynamically, not visible to
        # static analysis), so every `.connect()` call below false-positives
        # as no-member; this is a known pylint/PySide6 limitation, not a bug.
        timer.timeout.connect(callback)  # pylint: disable=no-member
        timer.start(delay_ms)
        self._timers.append(timer)
        return timer

    def cancel(self, handle: QTimer) -> None:
        """Stop a timer previously returned by :meth:`schedule`."""
        handle.stop()

    # -- Layout ------------------------------------------------------------

    def _build_widgets(self) -> QVBoxLayout:
        assert self.controller is not None
        controller = self.controller

        layout = QVBoxLayout(self.window)

        self.header = QLabel()
        self.header.setTextFormat(Qt.TextFormat.PlainText)
        self.header.setWordWrap(True)
        layout.addWidget(self.header)

        self.text = _TrackTextEdit()
        self.text.setReadOnly(True)
        self.text.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.text.setFont(_monospace_font())
        layout.addWidget(self.text, stretch=1)

        controls = QHBoxLayout()
        self.first_button = _button("⏮ First", lambda: self._goto(0))
        controls.addWidget(self.first_button)
        self.prev_button = _button("◀ Prev  (←)", lambda: self._goto(controller.index - 1))
        controls.addWidget(self.prev_button)
        self.play_button = _button(_PLAY_LABEL, self._toggle_play)
        controls.addWidget(self.play_button)
        self.next_button = _button("Next ▶  (→)", lambda: self._goto(controller.index + 1))
        controls.addWidget(self.next_button)
        self.last_button = _button("Last ⏭", lambda: self._goto(controller.last_index))
        controls.addWidget(self.last_button)
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
        self.slider.valueChanged.connect(self._on_slider)  # pylint: disable=no-member
        layout.addWidget(self.slider)

        return layout

    def _build_watch_widgets(self, layout: QVBoxLayout) -> None:
        """Lay out the watch kind/ref pickers, add/remove buttons, and watch list."""
        watch_row = QHBoxLayout()
        watch_row.addWidget(QLabel("Watch:"))
        self.watch_kind = QComboBox()
        self.watch_kind.addItems(["qubit", "clbit", "register"])
        self.watch_kind.currentTextChanged.connect(  # pylint: disable=no-member
            self._refresh_watch_ref_choices
        )
        watch_row.addWidget(self.watch_kind)
        self.watch_ref = QComboBox()
        watch_row.addWidget(self.watch_ref)
        watch_row.addWidget(_button("+ Add watch", self._add_watch))
        watch_row.addWidget(_button("Remove selected", self._remove_selected_watch))
        watch_row.addStretch(1)
        layout.addLayout(watch_row)

        self.watch_list = QListWidget()
        self.watch_list.setMaximumHeight(70)
        layout.addWidget(self.watch_list)
        self._refresh_watch_ref_choices()

    # -- Widget <-> PlaybackController glue --------------------------------

    def _render(self) -> None:
        """Refresh every widget to reflect the controller's current event."""
        assert self.controller is not None
        controller = self.controller

        source, event = controller.current
        index = controller.index
        self.header.setText(
            f"{index + 1}/{len(controller.merged)} — {source.name} — {event.type.value} "
            f"at {event.timestamp:.3f} — {event.describe()}"
        )

        tracks_text, active_span = _render.render_tracks_text_with_active(
            self.tracks[index], self.active_labels[index]
        )
        self.text.clear()
        self.text.setPlainText(tracks_text + "\n\n" + _render.render_event(source, event))

        label, png_bytes = _render.render_extra_visual_bytes(event)
        if png_bytes:
            image = QImage.fromData(png_bytes)
            if image.width() > MAX_IMAGE_WIDTH or image.height() > MAX_IMAGE_HEIGHT:
                image = image.scaled(
                    MAX_IMAGE_WIDTH,
                    MAX_IMAGE_HEIGHT,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            cursor = QTextCursor(self.text.document())
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.insertBlock()
            cursor.insertText(f"-- {label} --\n")
            cursor.insertImage(image)

        watch_text = _render.render_watches(self.watches, self.timelines_by_id, event.seq)
        if watch_text:
            cursor = QTextCursor(self.text.document())
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.insertBlock()
            cursor.insertText("-- Watches --\n" + watch_text)

        self._highlight_and_scroll(active_span)
        self._update_control_state()
        self._refresh_watch_ref_choices()

        self._syncing_slider = True
        try:
            self.slider.setValue(index + 1)
        finally:
            self._syncing_slider = False

    def _refresh_watch_ref_choices(self, _kind: str | None = None) -> None:
        """Repopulate the watch-ref dropdown for the currently selected circuit."""
        assert self.controller is not None
        source, _event = self.controller.current
        kind = self.watch_kind.currentText()
        previous = self.watch_ref.currentText()
        self.watch_ref.blockSignals(True)
        self.watch_ref.clear()
        if kind == "qubit":
            self.watch_ref.addItems([str(i) for i in range(source.num_qubits)])
        elif kind == "clbit":
            self.watch_ref.addItems([str(i) for i in range(source.num_clbits)])
        else:
            self.watch_ref.addItems([*source.qregs, *source.cregs])
        index = self.watch_ref.findText(previous)
        if index >= 0:
            self.watch_ref.setCurrentIndex(index)
        self.watch_ref.blockSignals(False)

    def _add_watch(self) -> None:
        """Add the currently selected qubit/clbit/register to the watch list."""
        assert self.controller is not None
        ref = self.watch_ref.currentText()
        if not ref:
            return
        source, _event = self.controller.current
        kind = self.watch_kind.currentText()
        assert kind in ("qubit", "clbit", "register")
        target = _render.WatchTarget(circuit_id=source.circuit_id, kind=kind, ref=ref)
        if target not in self.watches:
            self.watches.append(target)
            item = QListWidgetItem(target.label(source))
            item.setData(Qt.ItemDataRole.UserRole, target)
            self.watch_list.addItem(item)
            self._render()

    def _remove_selected_watch(self) -> None:
        """Remove whichever watch-list rows are currently selected."""
        for item in self.watch_list.selectedItems():
            target = item.data(Qt.ItemDataRole.UserRole)
            if target in self.watches:
                self.watches.remove(target)
            self.watch_list.takeItem(self.watch_list.row(item))
        self._render()

    def _update_control_state(self) -> None:
        """Disable controls that can't do anything at the current position."""
        assert self.controller is not None
        controller = self.controller

        at_first = controller.index <= 0
        at_last = controller.index >= controller.last_index

        self.first_button.setEnabled(not at_first)
        self.prev_button.setEnabled(not at_first)
        self.next_button.setEnabled(not at_last)
        self.last_button.setEnabled(not at_last)
        self.play_button.setEnabled(controller.playing or not at_last)

    def _highlight_and_scroll(self, active_span: tuple[int, int] | None) -> None:
        """Highlight the track section most relevant to the current event and
        scroll it into view, so the user doesn't have to hunt for what changed.
        """
        self.text.set_active_span(active_span)
        if active_span is None:
            return

        doc = self.text.document()
        start_line, _ = active_span
        start_block = doc.findBlockByNumber(start_line - 1)

        scroll_cursor = QTextCursor(doc)
        scroll_cursor.setPosition(start_block.position())
        self.text.setTextCursor(scroll_cursor)
        self.text.ensureCursorVisible()

        # ensureCursorVisible() only scrolls the minimum distance needed, so
        # if the span's start is already technically on screen (e.g. right
        # at the bottom edge) nothing moves and the rest of the circuit plus
        # the event details below it stay hidden. Pin the start of the span
        # to the very top of the viewport instead, so the active row and
        # everything beneath it get as much room as the viewport allows.
        scrollbar = self.text.verticalScrollBar()
        offset = self.text.cursorRect(scroll_cursor).top()
        scrollbar.setValue(scrollbar.value() + offset)

    def _on_play_state_changed(self, playing: bool) -> None:
        self.play_button.setText(_PAUSE_LABEL if playing else _PLAY_LABEL)
        self._update_control_state()

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


def _button(text: str, on_click: Callable[[], None]) -> QPushButton:
    button = QPushButton(text)
    button.clicked.connect(on_click)  # pylint: disable=no-member
    return button


def _monospace_font() -> QFont:
    font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
    font.setPointSize(11)
    return font
