"""Full-screen TUI frontend: layout, column widths, and pure ``_render``
helpers that don't need a running ``TimelineApp``.

Split out of ``test_tui.py`` to keep each module under pylint's
too-many-lines threshold.
"""

from _helpers import traced_bell
from _tui_helpers import _make_app, _plain

from qisdebug._events import ExecutionFinishedEvent, StateSnapshotEvent, Timeline
from qisdebug._frontends._cli_session import CliSession
from qisdebug._frontends._tui import TimelineApp


async def test_tui_timeline_row_name_column_is_narrow(debug):
    """The circuit-name column is trimmed short (names are usually short);
    the event-type column keeps its own, wider fixed width untouched."""
    app, session, _timeline = _make_app(debug)
    async with app.run_test():
        row = session.timeline_rows()[0]
        # "  <name col>  <type col>  <description...>", index-column first.
        _index, name_col, _type_col, _rest = row.split(maxsplit=3)
        assert len(name_col) <= 10
        assert row.count(" ") >= 2  # sanity: still column-separated, not squashed


async def test_tui_circuit_panel_stacked_above_info_panel(debug):
    """The circuit drawing and the event's other info are stacked (circuit
    on top), not placed side by side."""
    app, _session, _timeline = _make_app(debug)
    async with app.run_test():
        circuit_region = app.query_one("#circuit-pane").region
        info_region = app.query_one("#info-pane").region
        assert circuit_region.y < info_region.y
        assert circuit_region.x == info_region.x


def test_current_frame_includes_bloch_text_for_single_qubit_snapshot():
    """A single-qubit state snapshot's Bloch sphere is available for a
    presentation layer to show alongside the circuit -- CLI's counterpart to
    the GUI's image-based Bloch sphere."""
    timeline = Timeline(circuit_id="c1", name="single", num_qubits=1)
    timeline.events.append(
        StateSnapshotEvent(seq=1, step=0, num_qubits=1, statevector=[1.0, 0.0])
    )
    session = CliSession([timeline])
    frame = session.current_frame()
    assert frame.bloch_text is not None
    assert "bloch" in frame.bloch_text


def test_current_frame_histogram_respects_info_width():
    """Current frame histogram respects info width."""
    timeline = Timeline(circuit_id="c1", name="bell", num_qubits=2, num_clbits=2)
    timeline.events.append(ExecutionFinishedEvent(seq=1, counts={"00": 4, "11": 996}))
    session = CliSession([timeline])
    frame = session.current_frame(info_width=24)
    assert frame.histogram_text is not None
    for line in frame.histogram_text.split("\n"):
        assert len(line) <= 24


def test_timeline_rows_truncate_with_ellipsis_when_too_narrow():
    """A row that doesn't fit max_width is visibly marked as cut off (a
    trailing "…"), not silently clipped by the widget with no indication."""
    timeline = Timeline(circuit_id="c1", name="viz", num_qubits=1)
    timeline.events.append(
        ExecutionFinishedEvent(
            seq=1, backend="a_very_long_backend_name_that_will_not_fit", counts={"0": 1}
        )
    )
    session = CliSession([timeline])
    rows = session.timeline_rows(max_width=25)
    assert len(rows[0]) == 25
    assert rows[0].endswith("…")


def test_timeline_rows_unbounded_without_max_width():
    """Timeline rows unbounded without max width."""
    timeline = Timeline(circuit_id="c1", name="viz", num_qubits=1)
    timeline.events.append(
        ExecutionFinishedEvent(
            seq=1, backend="a_very_long_backend_name_that_will_not_fit", counts={"0": 1}
        )
    )
    session = CliSession([timeline])
    rows = session.timeline_rows()
    assert not rows[0].endswith("…")
    assert "a_very_long_backend_name_that_will_not_fit" in rows[0]


def test_format_statevector_uses_unicode_ket_bracket(debug):
    """Ket notation uses the proper Unicode closing bracket (⟩), matching
    the rest of the app's existing Unicode use (circuit box-drawing,
    shaded bars) rather than a plain ASCII ">"."""
    from qisdebug._frontends._render import format_statevector

    timeline = traced_bell(debug)
    text = format_statevector(timeline.snapshots[-1])
    assert "⟩" in text


async def test_tui_focus_pane_class_on_every_focusable_panel(debug):
    """Tab-focus is visually distinguishable (see .focus-pane:focus in
    TimelineApp.CSS) -- every pane a user can Tab into carries the class
    that CSS rule targets, so focus never lands somewhere invisible."""
    app, _session, _timeline = _make_app(debug)
    async with app.run_test():
        for widget_id in (
            "timeline-list", "circuit-pane", "info-pane", "watches-pane", "breakpoints-pane",
        ):  # fmt: skip
            assert app.query_one(f"#{widget_id}").has_class("focus-pane")


async def test_tui_tab_moves_focus_between_panes(debug):
    """Tui tab moves focus between panes."""
    app, _session, _timeline = _make_app(debug)
    async with app.run_test() as pilot:
        first_focused = app.focused
        await pilot.press("tab")
        assert app.focused is not first_focused
        assert app.focused is not None
        assert app.focused.has_class("focus-pane")


async def test_tui_histogram_pane_hidden_without_counts_shown_with(debug):
    """Tui histogram pane hidden without counts shown with."""
    app, _session, _timeline = _make_app(debug)
    async with app.run_test() as pilot:
        histogram = app.query_one("#info-histogram")
        assert histogram.has_class("hidden")

        await pilot.press("]")  # last event: execution finished, has counts
        await pilot.pause()
        assert not histogram.has_class("hidden")
        assert "Counts histogram" in _plain(histogram)


def test_render_tracks_text_heading_has_underline_rule(debug):
    """Section headings render as a title with an underline rule beneath it."""
    from qisdebug._frontends._render import render_tracks_text

    timeline = traced_bell(debug)
    session = CliSession([timeline])
    session.toggle_show_all_circuits()
    text = render_tracks_text(session.tracks[-1])
    assert "─" in text


def _traced_two_circuits(debug):
    """Two tracked circuits, so the collapsed view always has one other
    circuit not shown -- for exercising other_circuits/#circuit-note."""
    timeline_a = traced_bell(debug, name="viz_a")
    timeline_b = traced_bell(debug, name="viz_b")
    return timeline_a, timeline_b


def test_current_frame_other_circuits_kept_separate_from_tracks_text(debug):
    """The "+N more not shown" note is its own field, not appended into the
    circuit drawing text -- so a presentation layer can put it somewhere
    that doesn't scroll/zoom with the diagram."""
    timeline_a, timeline_b = _traced_two_circuits(debug)
    session = CliSession([timeline_a, timeline_b])
    session.go_last()
    frame = session.current_frame()
    assert frame.other_circuits
    assert "not shown" not in frame.tracks_text
    assert all(name not in frame.tracks_text for name in frame.other_circuits)


async def test_tui_circuit_note_shown_separately_from_circuit_drawing(debug):
    """Tui circuit note shown separately from circuit drawing."""
    timeline_a, timeline_b = _traced_two_circuits(debug)
    session = CliSession([timeline_a, timeline_b])
    app = TimelineApp(session, "t")
    async with app.run_test() as pilot:
        await pilot.press("]")
        await pilot.pause()
        note = app.query_one("#circuit-note")
        assert not note.has_class("hidden")
        assert "not shown" in _plain(note)
        circuit_text = _plain(app.query_one("#circuit"))
        assert "not shown" not in circuit_text


def test_default_zoom_is_unlimited_width(debug):
    """The circuit draws at its natural width by default -- the TUI's
    circuit panel scrolls horizontally to it (overflow-x: auto) instead of
    wrapping up front; "zoom out" is what steps down to a narrower fold."""
    timeline = traced_bell(debug)
    session = CliSession([timeline])
    assert session.fold_width == -1


async def test_tui_circuit_pane_scrolls_horizontally_at_default_zoom(debug):
    """A circuit wider than the panel is genuinely horizontally scrollable
    at the default zoom, not clipped with no way to see the rest."""
    debug.start()
    from qiskit.circuit import QuantumCircuit

    qc = QuantumCircuit(2, name="long")
    for _ in range(40):
        qc.h(0)
        qc.cx(0, 1)
    debug.end()
    timeline = debug.get(qc)
    session = CliSession([timeline])
    app = TimelineApp(session, "t")
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.press("]")
        await pilot.pause()
        pane = app.query_one("#circuit-pane")
        assert pane.virtual_size.width > pane.size.width

        pane.focus()
        await pilot.press("right")
        await pilot.pause()
        assert pane.scroll_x > 0


async def test_tui_quit_exits_cleanly(debug):
    """Tui quit exits cleanly."""
    app, _session, _timeline = _make_app(debug)
    async with app.run_test() as pilot:
        await pilot.press("q")
        await pilot.pause()
        assert app.is_running is False
