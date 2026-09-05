"""Native GUI frontend: a Qt window, used when a display is available.

Built on PySide6 rather than tkinter: PySide6's wheels bundle the Qt
binaries they need, so ``pip install "qisdebug[gui]"`` is enough on its own,
no separate system Qt install required, and its layout manager (as opposed
to tkinter's ``pack``) keeps fixed-size widgets like the playback controls
from ever being squeezed out by a large circuit diagram or histogram image.
This is why the GUI frontend is behind the ``gui`` extra rather than always
available like the stdlib-only CLI frontend.
"""

# pylint: disable=too-many-lines
# One cohesive Qt widget class (_TimelineWindow) implements every timeline
# feature end to end, per the project's own rule that a presentation feature
# must land in both frontends -- splitting the file along feature lines would
# scatter tightly coupled widget/handler pairs across modules for no benefit
# to readability, only to satisfy a line-count heuristic.

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, ClassVar, Literal

from PySide6.QtCore import QEvent, QObject, QRectF, Qt, QStringListModel, QTimer
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QImage,
    QKeyEvent,
    QKeySequence,
    QPainter,
    QPalette,
    QPen,
    QShortcut,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QCompleter,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedLayout,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .._events import TelemetryEvent, Timeline
from . import _breakpoints as _bp
from . import _circuit_diff, _render
from ._display import gui_available
from ._playback import PlaybackController, PlaybackHooks, Scheduler
from .base import TimelineFrontend

DEFAULT_AUTOPLAY_INTERVAL_MS = 800
ACTIVE_TRACK_BORDER_COLOR_LIGHT = "#1d4ed8"
ACTIVE_TRACK_BORDER_COLOR_DARK = "#60a5fa"
ACTIVE_TRACK_BORDER_WIDTH = 1.5
ACTIVE_TRACK_PADDING_X = 4
ACTIVE_TRACK_PADDING_Y = 3
DIFF_REMOVED_COLOR_LIGHT = "#fecaca"
DIFF_REMOVED_COLOR_DARK = "#7f1d1d"
DIFF_ADDED_COLOR_LIGHT = "#bbf7d0"
DIFF_ADDED_COLOR_DARK = "#14532d"
DIFF_REMOVED_TEXT_COLOR_LIGHT = "#dc2626"
DIFF_REMOVED_TEXT_COLOR_DARK = "#f87171"
DIFF_ADDED_TEXT_COLOR_LIGHT = "#16a34a"
DIFF_ADDED_TEXT_COLOR_DARK = "#4ade80"
MAX_IMAGE_WIDTH = 480
MAX_IMAGE_HEIGHT = 360
DEFAULT_CIRCUIT_FONT_POINT_SIZE = 10
MIN_CIRCUIT_FONT_POINT_SIZE = 6
MAX_CIRCUIT_FONT_POINT_SIZE = 24
CIRCUIT_FONT_ZOOM_STEP = 1
HELP_TEXT = """\
Keyboard shortcuts

Ctrl+← / Ctrl+→      previous / next event
Ctrl+Space          play / pause
Ctrl+= / Ctrl+-     zoom the circuit drawing in / out
Ctrl+0              reset circuit drawing zoom
Ctrl+scroll wheel   zoom the circuit drawing (over the drawing pane)
?                   this help
"""
_PLAY_LABEL = "⏵ Play  (Ctrl+Space)"
_PAUSE_LABEL = "⏸ Pause  (Ctrl+Space)"


@dataclass(frozen=True)
class _TimelineView:
    """Everything :class:`_TimelineWindow` needs to render one merged timeline.

    Bundles :func:`~._render.merge_timelines`'s three parallel per-event
    arrays into one object, so the window constructor takes one view
    argument instead of growing a parameter for each new piece of render
    state.
    """

    merged: list[tuple[Timeline, TelemetryEvent]]
    tracks: list[dict[str, str]]
    active_labels: list[str | None]
    timelines_by_id: dict[str, Timeline]


class QtFrontend(TimelineFrontend):
    """Opens a Qt window with a slider, play controls, and a diagram panel."""

    name: ClassVar[str] = "gui"

    def is_available(self) -> bool:
        return gui_available()

    def unavailable_reason(self) -> str:
        return (
            "no display is available for a native GUI window "
            '(PySide6 not installed -- pip install "qisdebug[gui]" -- '
            "or DISPLAY/WAYLAND_DISPLAY unset)"
        )

    def show(self, timelines: list[Timeline]) -> None:
        if not self.is_available():
            raise RuntimeError(self.unavailable_reason())

        merged, tracks, active_labels = _render.merge_timelines(timelines)
        names = ", ".join(str(t.name) for t in timelines)
        timelines_by_id = {t.circuit_id: t for t in timelines}

        view = _TimelineView(
            merged=merged,
            tracks=tracks,
            active_labels=active_labels,
            timelines_by_id=timelines_by_id,
        )
        app = QApplication.instance() or QApplication([])
        window = _TimelineWindow(names, view)
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
        self.on_ctrl_wheel: Callable[[int], None] | None = None

    def set_active_span(self, span: tuple[int, int] | None) -> None:
        """Outline the 1-indexed (start_line, end_line) span, or clear it if ``None``."""
        self._active_span = span
        self.viewport().update()

    def wheelEvent(self, event) -> None:  # noqa: N802 (Qt override name)
        """Ctrl+scroll zooms the circuit drawing instead of scrolling it,
        mirroring the desktop-wide convention (browsers, editors); plain
        scrolling is left to the base class for everything else.
        """
        if (
            self.on_ctrl_wheel is not None
            and event.modifiers() == Qt.KeyboardModifier.ControlModifier
        ):
            self.on_ctrl_wheel(1 if event.angleDelta().y() > 0 else -1)
            event.accept()
            return
        super().wheelEvent(event)

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
    return (
        ACTIVE_TRACK_BORDER_COLOR_DARK if is_dark else ACTIVE_TRACK_BORDER_COLOR_LIGHT
    )


def _diff_highlight_color(widget: QWidget, kind: Literal["removed", "added"]) -> str:
    """Pick a red ("removed") or green ("added") background fill with
    enough contrast for the widget's current palette, same approach as
    :func:`_active_track_border_color`.
    """
    background = widget.palette().color(QPalette.ColorRole.Base)
    is_dark = background.lightness() < 128
    if kind == "removed":
        return DIFF_REMOVED_COLOR_DARK if is_dark else DIFF_REMOVED_COLOR_LIGHT
    return DIFF_ADDED_COLOR_DARK if is_dark else DIFF_ADDED_COLOR_LIGHT


def _diff_text_color(widget: QWidget, kind: Literal["removed", "added"]) -> str:
    """Pick a red ("removed") or green ("added") *text* color with enough
    contrast for the widget's current palette -- used for the gate diff
    summary's ``  - ``/``  + `` lines, which read as plain colored text
    rather than the background fill :func:`_diff_highlight_color` gives the
    circuit drawing panes.
    """
    background = widget.palette().color(QPalette.ColorRole.Base)
    is_dark = background.lightness() < 128
    if kind == "removed":
        return (
            DIFF_REMOVED_TEXT_COLOR_DARK if is_dark else DIFF_REMOVED_TEXT_COLOR_LIGHT
        )
    return DIFF_ADDED_TEXT_COLOR_DARK if is_dark else DIFF_ADDED_TEXT_COLOR_LIGHT


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
    so Ctrl+Left/Ctrl+Right/Ctrl+Space navigate the timeline no matter which
    control (the slider, the interval spin box, the text panel) currently
    has focus. The shortcuts require the Ctrl modifier specifically so they
    never swallow plain Left/Right/Space while the user is typing in a text
    box or watch/breakpoint field, and Ctrl+arrow/Ctrl+space are not bound to
    anything at the OS level on any of the supported platforms, unlike
    Alt+Left/Right (browser/window-manager back-forward) or plain Space.
    """

    def __init__(self, handler: Callable[[int], bool]) -> None:
        super().__init__()
        self._handler = handler

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        """Consume Ctrl+Left/Ctrl+Right/Ctrl+Space before the focused widget sees them."""
        if isinstance(event, QKeyEvent) and event.type() == QEvent.Type.KeyPress:
            if (
                event.modifiers() == Qt.KeyboardModifier.ControlModifier
                and self._handler(event.key())
            ):
                return True
        return super().eventFilter(watched, event)


class _TimelineWindow(
    Scheduler[QTimer]
):  # pylint: disable=too-many-instance-attributes
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

    def __init__(self, title: str, view: _TimelineView) -> None:
        self.tracks = view.tracks
        self.active_labels = view.active_labels
        self.timelines_by_id = view.timelines_by_id
        self.watches: list[_render.WatchTarget] = []
        self.marks: dict[str, _render.StageMark | None] = {"a": None, "b": None}
        self.breakpoints = _bp.FrontendBreakpoints()
        self._syncing_slider = False
        self._timers: list[QTimer] = []
        self._key_filter: _PlaybackKeyFilter | None = None
        self._show_all_circuits = False
        self._circuit_font_point_size = DEFAULT_CIRCUIT_FONT_POINT_SIZE

        self.window = _TimelineWidget(self._cleanup_key_filter)
        self.window.setWindowTitle(f"QisDebug timeline — {title}")
        self.window.resize(1280, 860)
        self.window.setMinimumSize(800, 500)

        if not view.merged:
            layout = QVBoxLayout(self.window)
            layout.addWidget(QLabel("Timeline(s) have no events."))
            self.controller: PlaybackController | None = None
            return

        self.controller = PlaybackController(
            merged=view.merged,
            scheduler=self,
            hooks=PlaybackHooks(
                get_interval_ms=self._interval_ms,
                on_render=self._render,
                on_play_state_changed=self._on_play_state_changed,
            ),
        )
        container = QWidget()
        layout = self._build_widgets(container)
        self._build_watch_widgets(layout)
        self._build_mark_widgets(layout)
        self._build_breakpoint_pages()
        self._build_breakpoint_widgets(layout)
        self._diff_panel = self._build_diff_panel()

        self.tabs = QTabWidget()
        self.tabs.addTab(container, "Timeline")
        window_layout = QHBoxLayout(self.window)
        window_layout.addWidget(self.tabs)
        self._refresh_diff_panel()

        self._render()

        app = QApplication.instance()
        if app is not None:
            self._key_filter = _PlaybackKeyFilter(self._handle_shortcut)
            app.installEventFilter(self._key_filter)

        self._help_shortcut = QShortcut(QKeySequence("?"), self.window)
        self._help_shortcut.activated.connect(self._show_help)  # pylint: disable=no-member

    def _cleanup_key_filter(self) -> None:
        """Unregister the playback key filter so it doesn't outlive this window."""
        app = QApplication.instance()
        if app is not None and self._key_filter is not None:
            app.removeEventFilter(self._key_filter)
        self._key_filter = None

    def _handle_shortcut(self, key: int) -> bool:
        """Handle a global playback/zoom shortcut key, returning whether it was consumed."""
        if self.controller is None:
            return False
        controller = self.controller
        actions: dict[int, Callable[[], None]] = {
            Qt.Key.Key_Left: lambda: self._goto(controller.index - 1),
            Qt.Key.Key_Right: lambda: self._goto(controller.index + 1),
            Qt.Key.Key_Space: self._toggle_play,
            Qt.Key.Key_Plus: self._zoom_in,
            Qt.Key.Key_Equal: self._zoom_in,
            Qt.Key.Key_Minus: self._zoom_out,
            Qt.Key.Key_0: self._zoom_reset,
        }
        action = actions.get(key)
        if action is None:
            return False
        action()
        return True

    def _zoom_in(self) -> None:
        self._set_circuit_font_point_size(
            self._circuit_font_point_size + CIRCUIT_FONT_ZOOM_STEP
        )

    def _zoom_out(self) -> None:
        self._set_circuit_font_point_size(
            self._circuit_font_point_size - CIRCUIT_FONT_ZOOM_STEP
        )

    def _zoom_reset(self) -> None:
        self._set_circuit_font_point_size(DEFAULT_CIRCUIT_FONT_POINT_SIZE)

    def _set_circuit_font_point_size(self, point_size: int) -> None:
        """Apply a clamped font point size to the circuit drawing pane --
        the GUI's "zoom", a native desktop interaction the terminal frontend
        can't offer, which instead zooms by redrawing at a different fold
        width (see :meth:`~._cli_session.CliSession.zoom_in`).
        """
        self._circuit_font_point_size = max(
            MIN_CIRCUIT_FONT_POINT_SIZE,
            min(MAX_CIRCUIT_FONT_POINT_SIZE, point_size),
        )
        font = self.text.font()
        font.setPointSize(self._circuit_font_point_size)
        self.text.setFont(font)

    def _set_show_all_circuits(self, checked: bool) -> None:
        """Switch the drawing pane between just the active circuit and every
        tracked circuit's every track, per the "Show all circuits" checkbox."""
        self._show_all_circuits = checked
        self._render()

    def _show_help(self) -> None:
        """Open a minimal dialog listing the window's keyboard shortcuts."""
        QMessageBox.information(self.window, "QisDebug — Help", HELP_TEXT)

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

    def _build_widgets(self, container: QWidget) -> QVBoxLayout:
        if self.controller is None:
            raise RuntimeError(
                "internal error: controller accessed before it was built"
            )
        controller = self.controller

        layout = QVBoxLayout(container)

        header_row = QHBoxLayout()
        self.header = QLabel()
        self.header.setTextFormat(Qt.TextFormat.PlainText)
        self.header.setWordWrap(True)
        header_row.addWidget(self.header, stretch=1)
        self.show_all_circuits_checkbox = QCheckBox("Show all circuits")
        self.show_all_circuits_checkbox.setToolTip(
            "Only the circuit relevant to the current event is drawn by "
            "default; check this to draw every tracked circuit's every "
            "track at once, the way earlier versions always did."
        )
        self.show_all_circuits_checkbox.toggled.connect(  # pylint: disable=no-member
            self._set_show_all_circuits
        )
        header_row.addWidget(self.show_all_circuits_checkbox)
        layout.addLayout(header_row)

        # Its own row, outside the circuit-drawing pane itself -- session
        # metadata (what else is tracked but not drawn), not part of the
        # diagram, so it shouldn't live inside the same text that zooms.
        self.other_circuits_label = QLabel()
        self.other_circuits_label.setTextFormat(Qt.TextFormat.PlainText)
        self.other_circuits_label.setWordWrap(True)
        self.other_circuits_label.setVisible(False)
        layout.addWidget(self.other_circuits_label)

        detail_split, self.text, self.info_text = self._build_detail_split()
        self._set_circuit_font_point_size(DEFAULT_CIRCUIT_FONT_POINT_SIZE)
        layout.addWidget(detail_split, stretch=1)

        controls = QHBoxLayout()
        self.first_button = _button("⏮ First", lambda: self._goto(0))
        controls.addWidget(self.first_button)
        self.prev_button = _button(
            "◀ Prev  (Ctrl+←)", lambda: self._goto(controller.index - 1)
        )
        controls.addWidget(self.prev_button)
        self.play_button = _button(_PLAY_LABEL, self._toggle_play)
        controls.addWidget(self.play_button)
        self.next_button = _button(
            "Next ▶  (Ctrl+→)", lambda: self._goto(controller.index + 1)
        )
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

    def _build_detail_split(self) -> tuple[QSplitter, _TrackTextEdit, QTextEdit]:
        """Build the circuit-drawing and event-info panes, returning the
        splitter plus each pane's text widget for the caller to keep.

        Two panes, not one blob of text, so both stay visible together
        instead of one pushing the other down a shared scroll region; a
        ``QSplitter`` so the user can still resize the split by hand.
        Stacked (circuit above info), not side by side: the circuit drawing
        is usually wide, and a side-by-side info column would either cramp
        it or itself be too narrow to read.
        """
        circuit_text = _TrackTextEdit()
        circuit_text.setReadOnly(True)
        circuit_text.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        circuit_text.setFont(_monospace_font())
        circuit_text.on_ctrl_wheel = lambda direction: (
            self._zoom_in() if direction > 0 else self._zoom_out()
        )

        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_text.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        info_text.setFont(_monospace_font())

        detail_split = QSplitter(Qt.Orientation.Vertical)
        detail_split.addWidget(self._labeled_pane("Circuit", circuit_text))
        detail_split.addWidget(self._labeled_pane("Info", info_text))
        detail_split.setStretchFactor(0, 2)
        detail_split.setStretchFactor(1, 1)
        return detail_split, circuit_text, info_text

    @staticmethod
    def _labeled_pane(title: str, widget: QWidget) -> QWidget:
        """Wrap ``widget`` with a small caption above it, for a splitter pane
        whose contents alone wouldn't say what it's showing. Styled bold
        with a bottom rule (via Qt's theme-relative ``palette()`` stylesheet
        function, so it stays correct in both light and dark palettes
        without hardcoding a color) so it actually reads as a title rather
        than blending into the pane's own content.
        """
        pane = QWidget()
        pane_layout = QVBoxLayout(pane)
        pane_layout.setContentsMargins(0, 0, 0, 0)
        label = QLabel(title)
        label.setTextFormat(Qt.TextFormat.PlainText)
        label.setStyleSheet(
            "font-weight: bold; padding: 2px 2px; border-bottom: 1px solid palette(mid);"
        )
        pane_layout.addWidget(label)
        pane_layout.addWidget(widget, stretch=1)
        return pane

    def _build_watch_widgets(self, layout: QVBoxLayout) -> None:
        """Lay out the watch kind/ref pickers and add/remove buttons on the
        left, and the formatted watch value text on the right, side by side
        on one line -- the input that adds a watch and the panel showing
        what's currently watched belong together, not stacked below the
        selection list.
        """
        watch_section = QHBoxLayout()

        controls_column = QVBoxLayout()
        watch_row = QHBoxLayout()
        watch_row.addWidget(QLabel("Watch:"))
        self.watch_kind = QComboBox()
        self.watch_kind.addItems(["qubit", "clbit", "register", "track"])
        self.watch_kind.currentTextChanged.connect(  # pylint: disable=no-member
            self._refresh_watch_ref_choices
        )
        watch_row.addWidget(self.watch_kind)
        self.watch_ref = QComboBox()
        self.watch_ref.setToolTip(
            "For qubit/clbit/register, gates-touched is reported for every "
            "circuit lineage (construction, transpiled, backend-compiled) "
            'automatically. Pick "track" instead to watch one lineage\'s '
            "transpile pass stats directly."
        )
        watch_row.addWidget(self.watch_ref)
        watch_row.addWidget(_button("+ Add watch", self._add_watch))
        watch_row.addWidget(_button("Remove selected", self._remove_selected_watch))
        watch_row.addStretch(1)
        controls_column.addLayout(watch_row)

        self.watch_list = QListWidget()
        self.watch_list.setMaximumHeight(70)
        controls_column.addWidget(self.watch_list)
        watch_section.addLayout(controls_column, stretch=1)

        self.watch_text = QTextEdit()
        self.watch_text.setReadOnly(True)
        self.watch_text.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.watch_text.setFont(_monospace_font())
        self.watch_text.setMaximumHeight(120)
        watch_section.addWidget(self.watch_text, stretch=1)

        layout.addLayout(watch_section)

        self._refresh_watch_ref_choices()

    def _build_mark_widgets(self, layout: QVBoxLayout) -> None:
        """Lay out the mark A/B/clear buttons and status label, on the
        Timeline tab -- marking is something you do while stepping through
        the timeline, not on a separate tab.
        """
        mark_row = QHBoxLayout()
        mark_row.addWidget(QLabel("Diff:"))
        mark_a_button = _button("Mark A", self._mark_a)
        mark_a_button.setToolTip(
            "Pin the event currently shown below as point A of a diff"
        )
        mark_row.addWidget(mark_a_button)
        mark_b_button = _button("Mark B", self._mark_b)
        mark_b_button.setToolTip(
            "Pin the event currently shown below as point B of a diff"
        )
        mark_row.addWidget(mark_b_button)
        clear_marks_button = _button("Clear marks", self._clear_marks)
        clear_marks_button.setToolTip("Remove both A and B marks")
        mark_row.addWidget(clear_marks_button)
        self.mark_status = QLabel()
        self.mark_status.setTextFormat(Qt.TextFormat.PlainText)
        mark_row.addWidget(self.mark_status)
        self.show_diff_button = _button("Show diff →", self._show_diff)
        self.show_diff_button.setToolTip(
            "Mark A and B pinned -- open the Diff tab to compare them"
        )
        self.show_diff_button.setVisible(False)
        mark_row.addWidget(self.show_diff_button)
        mark_row.addStretch(1)
        layout.addLayout(mark_row)

    def _build_breakpoint_pages(self) -> None:
        """Build the per-kind breakpoint field groups (comparator dropdown
        with symbols, qubit/ref dropdowns populated from the live circuit,
        autocomplete on free-text fields) into ``self.breakpoint_pages``,
        the same cascading-dropdown idea already used by the watch row's
        kind->ref pickers (:meth:`_refresh_watch_ref_choices`), just one
        page per kind instead of one dropdown swapping a single ref list.

        Split out from :meth:`_build_breakpoint_widgets` (called directly
        from ``__init__`` alongside it, both self-contained) purely to keep
        each under the project's statement-count lint threshold -- there's
        no ordering dependency between the two besides both running before
        :meth:`_refresh_breakpoint_fields`.
        """
        self.breakpoint_pages = QStackedLayout()

        self.bp_gate_name, self.bp_gate_name_choices = _completer_field(
            "gate name, e.g. cx"
        )
        self.bp_gate_qubits = QLineEdit()
        self.bp_gate_qubits.setPlaceholderText("qubits, e.g. 0 1 (optional)")
        self.breakpoint_pages.addWidget(
            self._page(self.bp_gate_name, self.bp_gate_qubits)
        )

        self.bp_seq_cmp = _comparator_combo()
        self.bp_seq_value = QSpinBox()
        self.bp_seq_value.setRange(0, 1_000_000)
        self.breakpoint_pages.addWidget(self._page(self.bp_seq_cmp, self.bp_seq_value))

        self.bp_watch_kind = QComboBox()
        self.bp_watch_kind.addItems(list(_bp.WATCH_KINDS))
        self.bp_watch_kind.currentTextChanged.connect(  # pylint: disable=no-member
            self._refresh_breakpoint_fields
        )
        self.bp_watch_ref = QComboBox()
        self.bp_watch_cmp = _comparator_combo()
        self.bp_watch_threshold = _threshold_spinbox()
        self.breakpoint_pages.addWidget(
            self._page(
                self.bp_watch_kind,
                self.bp_watch_ref,
                self.bp_watch_cmp,
                self.bp_watch_threshold,
            )
        )

        self.bp_expect_pauli = QComboBox()
        self.bp_expect_pauli.addItems(list(_bp.PAULIS))
        self.bp_expect_qubit = QComboBox()
        self.bp_expect_cmp = _comparator_combo()
        self.bp_expect_threshold = _threshold_spinbox()
        self.breakpoint_pages.addWidget(
            self._page(
                self.bp_expect_pauli,
                self.bp_expect_qubit,
                self.bp_expect_cmp,
                self.bp_expect_threshold,
            )
        )

        self.bp_prob_outcome, self.bp_prob_outcome_choices = _completer_field(
            "outcome, e.g. 01"
        )
        self.bp_prob_cmp = _comparator_combo()
        self.bp_prob_threshold = _threshold_spinbox()
        self.breakpoint_pages.addWidget(
            self._page(self.bp_prob_outcome, self.bp_prob_cmp, self.bp_prob_threshold)
        )

        self.bp_tvd_expected = QLineEdit()
        self.bp_tvd_expected.setPlaceholderText("expected, e.g. 00=0.5 11=0.5")
        self.bp_tvd_cmp = _comparator_combo()
        self.bp_tvd_threshold = _threshold_spinbox()
        self.breakpoint_pages.addWidget(
            self._page(self.bp_tvd_expected, self.bp_tvd_cmp, self.bp_tvd_threshold)
        )

        self.bp_anomaly_severity = QComboBox()
        self.bp_anomaly_severity.addItems(list(_bp.SEVERITIES))
        self.breakpoint_pages.addWidget(self._page(self.bp_anomaly_severity))

    def _build_breakpoint_widgets(self, layout: QVBoxLayout) -> None:
        """Lay out the breakpoint condition builder, active-breakpoint list,
        and hit navigation, mirroring the watch row above -- adding/removing
        entries on the left, what's active on the right, one line each.

        Assumes :meth:`_build_breakpoint_pages` (called first, directly from
        ``__init__``) already populated ``self.breakpoint_pages``.
        """
        breakpoint_row = QHBoxLayout()
        breakpoint_row.addWidget(QLabel("Break:"))
        self.breakpoint_kind = QComboBox()
        self.breakpoint_kind.addItems(list(_bp.KINDS))
        self.breakpoint_kind.currentTextChanged.connect(  # pylint: disable=no-member
            self._on_breakpoint_kind_changed
        )
        breakpoint_row.addWidget(self.breakpoint_kind)

        pages_widget = QWidget()
        pages_widget.setLayout(self.breakpoint_pages)
        breakpoint_row.addWidget(pages_widget, stretch=1)

        breakpoint_row.addWidget(QLabel("on"))
        self.breakpoint_scope = QComboBox()
        self.breakpoint_scope.setToolTip(
            "Which circuit this breakpoint watches -- defaults to the one "
            "currently on screen; '(all)' means any circuit (gate/seq/anomaly only)"
        )
        breakpoint_row.addWidget(self.breakpoint_scope)

        breakpoint_row.addWidget(_button("+ Add breakpoint", self._add_breakpoint))
        breakpoint_row.addWidget(
            _button("Remove selected", self._remove_selected_breakpoint)
        )
        layout.addLayout(breakpoint_row)

        self.breakpoint_list = QListWidget()
        self.breakpoint_list.setMaximumHeight(60)
        layout.addWidget(self.breakpoint_list)

        hit_row = QHBoxLayout()
        hit_row.addWidget(QLabel("Breakpoint hits:"))
        hit_row.addWidget(_button("⏮ First hit", self._goto_first_hit))
        hit_row.addWidget(_button("◀ Prev hit", self._goto_prev_hit))
        hit_row.addWidget(_button("Next hit ▶", self._goto_next_hit))
        hit_row.addWidget(_button("Last hit ⏭", self._goto_last_hit))
        hit_row.addStretch(1)
        layout.addLayout(hit_row)

        self.hit_status_label = QLabel("")
        self.hit_status_label.setWordWrap(True)
        layout.addWidget(self.hit_status_label)

        self._refresh_breakpoint_fields()

    @staticmethod
    def _page(*widgets: QWidget) -> QWidget:
        page = QWidget()
        row = QHBoxLayout(page)
        row.setContentsMargins(0, 0, 0, 0)
        for widget in widgets:
            row.addWidget(widget)
        return page

    def _on_breakpoint_kind_changed(self, kind: str) -> None:
        self.breakpoint_pages.setCurrentIndex(_bp.KINDS.index(kind))
        self._refresh_breakpoint_fields()

    def _build_diff_panel(self) -> QWidget:
        """Build the "Diff" tab's contents, shown only once both marks are set.

        A mark pins a *stage* (a point in time), which can hold more than
        one circuit, and each circuit more than one track -- its
        construction, plus one per transpile run -- since transpilation
        doesn't emit its own construction-style events. The A/B pickers
        therefore choose an *element* (circuit + track, e.g.
        "bell [transpiled]") to diff in detail, not just a circuit, so a
        transpiled circuit is never silently diffed as its pre-transpile
        construction gates.
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)

        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel("Diff element A:"))
        self.diff_circuit_a = QComboBox()
        self.diff_circuit_a.setToolTip(
            "Which circuit and track (construction or a specific "
            "transpile run) to diff on side A"
        )
        self.diff_circuit_a.currentIndexChanged.connect(  # pylint: disable=no-member
            self._update_diff_text
        )
        selector_row.addWidget(self.diff_circuit_a)
        selector_row.addWidget(QLabel("Diff element B:"))
        self.diff_circuit_b = QComboBox()
        self.diff_circuit_b.setToolTip(
            "Which circuit and track (construction or a specific "
            "transpile run) to diff on side B"
        )
        self.diff_circuit_b.currentIndexChanged.connect(  # pylint: disable=no-member
            self._update_diff_text
        )
        selector_row.addWidget(self.diff_circuit_b)
        selector_row.addStretch(1)
        layout.addLayout(selector_row)

        split = QSplitter(Qt.Orientation.Horizontal)
        self.diff_label_a, self.diff_text_a = self._build_diff_pane(split)
        self.diff_label_b, self.diff_text_b = self._build_diff_pane(split)
        layout.addWidget(split, stretch=2)

        bottom_split = QSplitter(Qt.Orientation.Horizontal)

        watch_diff_pane = QWidget()
        watch_diff_layout = QVBoxLayout(watch_diff_pane)
        watch_diff_layout.setContentsMargins(0, 0, 0, 0)
        watch_diff_layout.addWidget(QLabel("Watch diff"))
        self.watch_diff_text = QTextEdit()
        self.watch_diff_text.setReadOnly(True)
        self.watch_diff_text.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.watch_diff_text.setFont(_monospace_font())
        watch_diff_layout.addWidget(self.watch_diff_text, stretch=1)
        bottom_split.addWidget(watch_diff_pane)

        self.diff_text = QTextEdit()
        self.diff_text.setReadOnly(True)
        self.diff_text.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.diff_text.setFont(_monospace_font())
        bottom_split.addWidget(self.diff_text)
        bottom_split.setStretchFactor(0, 1)
        bottom_split.setStretchFactor(1, 2)

        layout.addWidget(bottom_split, stretch=1)

        return panel

    @staticmethod
    def _build_diff_pane(split: QSplitter) -> tuple[QLabel, QTextEdit]:
        """Build one side (label + read-only monospace text) of the split-view
        circuit diagram diff and add it to ``split``.
        """
        pane = QWidget()
        pane_layout = QVBoxLayout(pane)
        pane_layout.setContentsMargins(0, 0, 0, 0)
        label = QLabel()
        label.setTextFormat(Qt.TextFormat.PlainText)
        pane_layout.addWidget(label)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        text.setFont(_monospace_font())
        pane_layout.addWidget(text, stretch=1)
        split.addWidget(pane)
        return label, text

    def _mark_a(self) -> None:
        self._set_mark("a")

    def _mark_b(self) -> None:
        self._set_mark("b")

    def _set_mark(self, slot: str) -> None:
        """Pin the current playback position's stage as ``slot`` of the diff
        panel. Once both slots become set, jump straight to the Diff tab so
        the per-stage element pickers (populated by
        :meth:`_refresh_diff_panel`) are the next thing the user sees --
        the guided "which element do you want" question a mark implies,
        without an extra click to go find it.
        """
        if self.controller is None:
            raise RuntimeError(
                "internal error: controller accessed before it was built"
            )
        was_both_marked = self.marks["a"] is not None and self.marks["b"] is not None
        _source, event = self.controller.current
        self.marks[slot] = _render.StageMark(seq=event.seq)
        self._refresh_diff_panel()
        now_both_marked = self.marks["a"] is not None and self.marks["b"] is not None
        if now_both_marked and not was_both_marked:
            self._show_diff()

    def _clear_marks(self) -> None:
        self.marks = {"a": None, "b": None}
        self._refresh_diff_panel()

    def _show_diff(self) -> None:
        """Switch straight to the Diff tab, rather than making the user
        notice and click it themselves once it appears.
        """
        self.tabs.setCurrentWidget(self._diff_panel)

    def _refresh_diff_panel(self) -> None:
        """Update the mark status label, the per-stage element pickers, and
        show or hide the Diff tab and its "Show diff" shortcut depending on
        whether both marks are now set.
        """
        self.mark_status.setText(
            f"A: {self._mark_label('a')}   B: {self._mark_label('b')}"
        )
        diff_index = self.tabs.indexOf(self._diff_panel)
        both_marked = self.marks["a"] is not None and self.marks["b"] is not None
        self.show_diff_button.setVisible(both_marked)
        self._populate_diff_element_choices()
        if both_marked:
            self._update_diff_text()
            if diff_index < 0:
                self.tabs.addTab(self._diff_panel, "Diff")
        elif diff_index >= 0:
            self.tabs.removeTab(diff_index)

    def _populate_diff_element_choices(self) -> None:
        """Refill the circuit/track A/B pickers from each stage's available
        elements -- a circuit's construction track plus one per transpile
        run (see :meth:`~._render.StageMark.elements_at`) -- keeping the
        previous selection when it is still one of the options. Each combo
        entry shows the friendly track label (:func:`_render.track_label`)
        but carries the raw ``"circuit:track"`` element as its data, which
        is what actually gets passed to :func:`_render.render_stage_diff`.
        """
        for slot, combo in (("a", self.diff_circuit_a), ("b", self.diff_circuit_b)):
            mark = self.marks[slot]
            previous = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            if mark is not None:
                elements = mark.elements_at(self.timelines_by_id)
                for element in elements:
                    combo.addItem(_render.track_label(element), element)
                if previous in elements:
                    combo.setCurrentIndex(combo.findData(previous))
            combo.blockSignals(False)

    def _update_diff_text(self) -> None:
        """Recompute the diff panel's text from the current marks/picks, without
        touching the element picker contents (see :meth:`_populate_diff_element_choices`).
        """
        mark_a, mark_b = self.marks["a"], self.marks["b"]
        if mark_a is None or mark_b is None:
            self.diff_text.setPlainText("")
            self.diff_text.setExtraSelections([])
            self.watch_diff_text.setPlainText("")
            self.watch_diff_text.setExtraSelections([])
            self._set_diff_pane(self.diff_label_a, self.diff_text_a, "", "")
            self._set_diff_pane(self.diff_label_b, self.diff_text_b, "", "")
            return
        element_a = self.diff_circuit_a.currentData()
        element_b = self.diff_circuit_b.currentData()
        resolved = _circuit_diff.resolve_split_and_stage_diff(
            mark_a,
            mark_b,
            self.timelines_by_id,
            element_a=element_a,
            element_b=element_b,
        )
        if isinstance(resolved, str):
            # Ambiguous/unresolved pick.
            self.diff_text.setPlainText(resolved)
            self.diff_text.setExtraSelections([])
            self.watch_diff_text.setPlainText("")
            self.watch_diff_text.setExtraSelections([])
            self._set_diff_pane(self.diff_label_a, self.diff_text_a, "", "")
            self._set_diff_pane(self.diff_label_b, self.diff_text_b, "", "")
            return
        split_diff, stage_diff_text = resolved
        self.diff_text.setPlainText(stage_diff_text)
        self._highlight_diff_lines(self.diff_text, stage_diff_text)
        watch_diff_text = _render.render_watch_diff(
            self.watches, self.timelines_by_id, mark_a.seq, mark_b.seq
        )
        self.watch_diff_text.setPlainText(watch_diff_text)
        self._highlight_diff_lines(self.watch_diff_text, watch_diff_text)
        self._set_diff_pane(
            self.diff_label_a, self.diff_text_a, split_diff.a.label, split_diff.a.text
        )
        self._set_diff_pane(
            self.diff_label_b, self.diff_text_b, split_diff.b.label, split_diff.b.text
        )
        self._highlight_diff_spans(self.diff_text_a, split_diff.a.spans, "removed")
        self._highlight_diff_spans(self.diff_text_b, split_diff.b.spans, "added")

    @staticmethod
    def _set_diff_pane(
        label: QLabel, text: QTextEdit, label_text: str, body: str
    ) -> None:
        label.setText(label_text)
        text.setExtraSelections([])
        text.setPlainText(body)

    @staticmethod
    def _highlight_diff_spans(
        text: QTextEdit, spans: list[tuple[int, int]], kind: Literal["removed", "added"]
    ) -> None:
        """Background-highlight the character ranges where the two circuit
        drawings differ -- red for ``"removed"`` (side A's own content, not
        matched on side B), green for ``"added"`` (side B's own content,
        not matched on side A). Unlike the active-event outline
        (:class:`_TrackTextEdit`), these are many short, scattered ranges
        rather than one contiguous block, which ``ExtraSelection`` handles
        directly without a custom ``paintEvent``.
        """
        color = QColor(_diff_highlight_color(text, kind))
        doc = text.document()
        selections = []
        for start, end in spans:
            if end <= start:
                continue
            cursor = QTextCursor(doc)
            cursor.setPosition(start)
            cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
            selection = QTextEdit.ExtraSelection()
            selection.cursor = cursor
            # pylint sees `.format` as the builtin function rather than the
            # QTextCharFormat property PySide6 actually returns here.
            selection.format.setBackground(color)  # pylint: disable=no-member
            selections.append(selection)
        text.setExtraSelections(selections)

    @staticmethod
    def _highlight_diff_lines(text_edit: QTextEdit, body: str) -> None:
        """Color the unified-diff ``  - ``/``  + `` gate lines from
        :func:`~._render.render_state_diff` red/green text (not the
        background fill :meth:`_highlight_diff_spans` uses for the circuit
        drawing panes -- these lines are already just one gate's
        description, so plain colored text reads better than a background
        block).
        """
        doc = text_edit.document()
        selections = []
        offset = 0
        for line in body.split("\n"):
            kind: Literal["removed", "added"] | None = None
            if line.startswith("  - "):
                kind = "removed"
            elif line.startswith("  + "):
                kind = "added"
            if kind is not None:
                cursor = QTextCursor(doc)
                cursor.setPosition(offset)
                cursor.setPosition(offset + len(line), QTextCursor.MoveMode.KeepAnchor)
                selection = QTextEdit.ExtraSelection()
                selection.cursor = cursor
                # pylint sees `.format` as the builtin function rather than
                # the QTextCharFormat property PySide6 actually returns here.
                selection.format.setForeground(  # pylint: disable=no-member
                    QColor(_diff_text_color(text_edit, kind))
                )
                selections.append(selection)
            offset += len(line) + 1
        text_edit.setExtraSelections(selections)

    def _mark_label(self, slot: str) -> str:
        mark = self.marks[slot]
        if mark is None:
            return "(unmarked)"
        return mark.label(self.timelines_by_id)

    # -- Widget <-> PlaybackController glue --------------------------------

    def _render(self) -> None:
        """Refresh every widget to reflect the controller's current event."""
        if self.controller is None:
            raise RuntimeError(
                "internal error: controller accessed before it was built"
            )
        controller = self.controller

        source, event = controller.current
        index = controller.index
        header_text = (
            f"{index + 1}/{len(controller.merged)} — {source.name} — {event.type.value} "
            f"at {event.timestamp:.3f} — {event.describe()}"
        )
        self.header.setText(header_text)

        other_circuits: list[str] = []
        if self._show_all_circuits:
            tracks_text, active_span = _render.render_tracks_text_with_active(
                self.tracks[index], self.active_labels[index]
            )
        else:
            tracks_text, other_circuits = _render.split_active_track(
                self.tracks[index], self.active_labels[index]
            )
            active_span = None
        self.text.clear()
        self.text.setPlainText(tracks_text)

        if other_circuits:
            names_not_shown = ", ".join(other_circuits)
            self.other_circuits_label.setText(
                f"+{len(other_circuits)} more not shown: {names_not_shown}"
            )
            self.other_circuits_label.setVisible(True)
        else:
            self.other_circuits_label.setVisible(False)

        self.info_text.clear()
        self.info_text.setPlainText(_render.render_event(source, event))

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
            cursor = QTextCursor(self.info_text.document())
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.insertBlock()
            cursor.insertText(f"-- {label} --\n")
            cursor.insertImage(image)

        self._refresh_watch_text(event.seq)

        self._highlight_and_scroll(active_span)
        self._update_control_state()
        self._refresh_watch_ref_choices()
        self._refresh_breakpoint_fields()

        self._syncing_slider = True
        try:
            self.slider.setValue(index + 1)
        finally:
            self._syncing_slider = False

    def _refresh_watch_text(self, upto_seq: int) -> None:
        """Repopulate the formatted watch value text for the event at ``upto_seq``."""
        self.watch_text.setPlainText(
            _render.render_watches(self.watches, self.timelines_by_id, upto_seq)
        )

    def _refresh_watch_ref_choices(self, _kind: str | None = None) -> None:
        """Repopulate the watch-ref dropdown for the currently selected circuit."""
        if self.controller is None:
            raise RuntimeError(
                "internal error: controller accessed before it was built"
            )
        source, event = self.controller.current
        kind = self.watch_kind.currentText()
        previous = self.watch_ref.currentText()
        self.watch_ref.blockSignals(True)
        self.watch_ref.clear()
        if kind == "qubit":
            self.watch_ref.addItems([str(i) for i in range(source.num_qubits)])
        elif kind == "clbit":
            self.watch_ref.addItems([str(i) for i in range(source.num_clbits)])
        elif kind == "register":
            self.watch_ref.addItems([*source.qregs, *source.cregs])
        else:
            self.watch_ref.addItems(_render.track_keys_upto(source, event.seq))
        index = self.watch_ref.findText(previous)
        if index >= 0:
            self.watch_ref.setCurrentIndex(index)
        self.watch_ref.blockSignals(False)

    def _add_watch(self) -> None:
        """Add the currently selected qubit/clbit/register/track to the watch list."""
        if self.controller is None:
            raise RuntimeError(
                "internal error: controller accessed before it was built"
            )
        ref = self.watch_ref.currentText()
        if not ref:
            return
        source, _event = self.controller.current
        kind = self.watch_kind.currentText()
        if kind not in ("qubit", "clbit", "register", "track"):
            raise ValueError(f"unknown watch kind {kind!r}")
        target = _render.WatchTarget(circuit_id=source.circuit_id, kind=kind, ref=ref)
        if target not in self.watches:
            self.watches.append(target)
            item = QListWidgetItem(target.label(source))
            item.setData(Qt.ItemDataRole.UserRole, target)
            self.watch_list.addItem(item)
            self._render()
            self._update_diff_text()

    def _remove_selected_watch(self) -> None:
        """Remove whichever watch-list rows are currently selected."""
        for item in self.watch_list.selectedItems():
            target = item.data(Qt.ItemDataRole.UserRole)
            if target in self.watches:
                self.watches.remove(target)
            self.watch_list.takeItem(self.watch_list.row(item))
        self._render()
        self._update_diff_text()

    _BREAKPOINT_SCOPE_CURRENT = "(current)"
    _BREAKPOINT_SCOPE_ALL = "(all)"

    def _refresh_breakpoint_fields(self, _value: object = None) -> None:
        """Repopulate every dynamic breakpoint field (watch ref, expect
        qubit, gate-name/outcome autocomplete, circuit scope) for the
        circuit currently on screen, mirroring :meth:`_refresh_watch_ref_choices`.
        """
        if self.controller is None:
            raise RuntimeError(
                "internal error: controller accessed before it was built"
            )
        source, _event = self.controller.current

        self.bp_gate_name_choices.setStringList(_bp.known_gate_names(source))
        self.bp_prob_outcome_choices.setStringList(_bp.small_outcome_space(source))

        previous_ref = self.bp_watch_ref.currentText()
        self.bp_watch_ref.blockSignals(True)
        self.bp_watch_ref.clear()
        watch_kind = self.bp_watch_kind.currentText()
        if watch_kind == "qubit":
            self.bp_watch_ref.addItems(_bp.qubit_range(source))
        elif watch_kind == "clbit":
            self.bp_watch_ref.addItems(_bp.clbit_range(source))
        else:
            self.bp_watch_ref.addItems(_bp.register_names(source))
        index = self.bp_watch_ref.findText(previous_ref)
        if index >= 0:
            self.bp_watch_ref.setCurrentIndex(index)
        self.bp_watch_ref.blockSignals(False)

        previous_qubit = self.bp_expect_qubit.currentText()
        self.bp_expect_qubit.clear()
        self.bp_expect_qubit.addItems(_bp.qubit_range(source))
        index = self.bp_expect_qubit.findText(previous_qubit)
        if index >= 0:
            self.bp_expect_qubit.setCurrentIndex(index)

        previous_scope_data = (
            self.breakpoint_scope.currentData()
            if self.breakpoint_scope.count()
            else None
        )
        self.breakpoint_scope.blockSignals(True)
        self.breakpoint_scope.clear()
        self.breakpoint_scope.addItem(self._BREAKPOINT_SCOPE_CURRENT, None)
        self.breakpoint_scope.addItem(self._BREAKPOINT_SCOPE_ALL, "*")
        for circuit_id in _bp.circuit_choices(self.timelines_by_id):
            timeline = self.timelines_by_id.get(circuit_id)
            name = timeline.name if timeline is not None else None
            label = f"{name} ({circuit_id})" if name else circuit_id
            self.breakpoint_scope.addItem(label, circuit_id)
        new_index = 0
        for i in range(self.breakpoint_scope.count()):
            if self.breakpoint_scope.itemData(i) == previous_scope_data:
                new_index = i
                break
        self.breakpoint_scope.setCurrentIndex(new_index)
        self.breakpoint_scope.blockSignals(False)

    def _breakpoint_tokens(self, kind: str) -> list[str]:
        """Read the currently visible field group's widgets into the token
        list :func:`~._breakpoints.parse_condition_args` expects, so parsing
        and validation still live in exactly one place.
        """
        if kind == "gate":
            tokens = [self.bp_gate_name.text(), *self.bp_gate_qubits.text().split()]
        elif kind == "seq":
            tokens = [self.bp_seq_cmp.currentText(), str(self.bp_seq_value.value())]
        elif kind == "watch":
            tokens = [
                self.bp_watch_kind.currentText(),
                self.bp_watch_ref.currentText(),
                self.bp_watch_cmp.currentText(),
                str(self.bp_watch_threshold.value()),
            ]
        elif kind == "expect":
            tokens = [
                self.bp_expect_pauli.currentText(),
                self.bp_expect_qubit.currentText(),
                self.bp_expect_cmp.currentText(),
                str(self.bp_expect_threshold.value()),
            ]
        elif kind == "prob":
            tokens = [
                self.bp_prob_outcome.text(),
                self.bp_prob_cmp.currentText(),
                str(self.bp_prob_threshold.value()),
            ]
        elif kind == "tvd":
            tokens = [
                *self.bp_tvd_expected.text().split(),
                self.bp_tvd_cmp.currentText(),
                str(self.bp_tvd_threshold.value()),
            ]
        else:
            severity = self.bp_anomaly_severity.currentText()
            tokens = [severity] if severity else []
        scope = self.breakpoint_scope.currentData()
        if scope is not None:
            tokens += ["on", scope]
        return tokens

    def _add_breakpoint(self) -> None:
        """Parse the active condition-builder page and register a
        breakpoint, scoped to the circuit currently on screen unless the
        "on" dropdown overrides it.
        """
        if self.controller is None:
            raise RuntimeError(
                "internal error: controller accessed before it was built"
            )
        source, _event = self.controller.current
        kind = self.breakpoint_kind.currentText()
        args = self._breakpoint_tokens(kind)
        try:
            condition = _bp.parse_condition_args(kind, args, source.circuit_id)
        except ValueError as exc:
            self.breakpoint_kind.setToolTip(str(exc))
            return
        self.breakpoint_kind.setToolTip("")
        handle = self.breakpoints.add(condition)
        item = QListWidgetItem(f"[{handle}] {_bp.describe_condition(condition)}")
        item.setData(Qt.ItemDataRole.UserRole, handle)
        self.breakpoint_list.addItem(item)

    def _remove_selected_breakpoint(self) -> None:
        """Remove whichever breakpoint-list rows are currently selected."""
        for item in self.breakpoint_list.selectedItems():
            handle = item.data(Qt.ItemDataRole.UserRole)
            self.breakpoints.remove(handle)
            self.breakpoint_list.takeItem(self.breakpoint_list.row(item))

    def _goto_hit(self, hit: tuple[int, str] | None, *, none_message: str) -> None:
        """Jump to a breakpoint hit and report which condition matched (or
        why there wasn't one), mirroring the CLI's ``_jump_to_hit`` so both
        frontends give the same feedback when several breakpoints are active.
        """
        if hit is None:
            self.hit_status_label.setText(none_message)
            return
        index, handle = hit
        condition = self.breakpoints.get(handle)
        assert condition is not None
        self.hit_status_label.setText(
            f"hit breakpoint [{handle}] {_bp.describe_condition(condition)}"
        )
        self._goto(index)

    def _goto_first_hit(self) -> None:
        """Jump to the earliest event matching any active breakpoint."""
        if self.controller is None:
            return
        if not self.breakpoints:
            self.hit_status_label.setText("no active breakpoints")
            return
        self._goto_hit(
            _bp.find_first_hit(self.controller.merged, self.breakpoints.as_dict()),
            none_message="no event matches any breakpoint",
        )

    def _goto_prev_hit(self) -> None:
        """Jump to the previous event matching any active breakpoint."""
        if self.controller is None:
            return
        if not self.breakpoints:
            self.hit_status_label.setText("no active breakpoints")
            return
        self._goto_hit(
            _bp.find_prev_hit(
                self.controller.merged,
                self.breakpoints.as_dict(),
                self.controller.index,
            ),
            none_message="no earlier event matches any breakpoint",
        )

    def _goto_next_hit(self) -> None:
        """Jump forward to the next event matching any active breakpoint."""
        if self.controller is None:
            return
        if not self.breakpoints:
            self.hit_status_label.setText("no active breakpoints")
            return
        self._goto_hit(
            _bp.find_next_hit(
                self.controller.merged,
                self.breakpoints.as_dict(),
                self.controller.index,
            ),
            none_message="no later event matches any breakpoint",
        )

    def _goto_last_hit(self) -> None:
        """Jump to the latest event matching any active breakpoint."""
        if self.controller is None:
            return
        if not self.breakpoints:
            self.hit_status_label.setText("no active breakpoints")
            return
        self._goto_hit(
            _bp.find_last_hit(self.controller.merged, self.breakpoints.as_dict()),
            none_message="no event matches any breakpoint",
        )

    def _update_control_state(self) -> None:
        """Disable controls that can't do anything at the current position."""
        if self.controller is None:
            raise RuntimeError(
                "internal error: controller accessed before it was built"
            )
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
        if self.controller is None:
            raise RuntimeError(
                "internal error: controller accessed before it was built"
            )
        self.controller.goto(index)

    def _toggle_play(self) -> None:
        if self.controller is None:
            raise RuntimeError(
                "internal error: controller accessed before it was built"
            )
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


def _comparator_combo() -> QComboBox:
    """A dropdown of the six symbol comparators (``>=``, ``<=``, ...),
    shared by every numeric breakpoint field group."""
    combo = QComboBox()
    combo.addItems(list(_bp.COMPARATOR_SYMBOLS))
    return combo


def _threshold_spinbox() -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(-1_000_000.0, 1_000_000.0)
    box.setDecimals(6)
    box.setValue(0.5)
    return box


def _completer_field(placeholder: str) -> tuple[QLineEdit, QStringListModel]:
    """A free-text field with an initially-empty, updatable autocomplete
    list -- the model is returned separately so callers can repopulate it
    later (:meth:`_TimelineWindow._refresh_breakpoint_fields`) without
    reaching back through ``QCompleter``'s generic model type.
    """
    field = QLineEdit()
    field.setPlaceholderText(placeholder)
    model = QStringListModel([])
    field.setCompleter(QCompleter(model))
    return field, model


def _monospace_font() -> QFont:
    font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
    font.setPointSize(11)
    return font
