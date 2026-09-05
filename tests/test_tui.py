"""Full-screen TUI frontend: persistent panels driven by ``CliSession``."""

from qiskit import transpile
from textual.widgets import Input
from _helpers import bell_circuit
from _tui_helpers import _make_app, _plain

from qisdebug._frontends._cli_session import CliSession
from qisdebug._frontends._guided import GuidedWizardScreen
from qisdebug._frontends._tui import CommandScreen, DiffScreen, ElementPickerScreen, TimelineApp


def _make_transpiled_app(debug):
    """A timeline with two elements at its final stage (construction plus one
    transpile run), for exercising :class:`ElementPickerScreen`."""
    debug.start()
    qc = bell_circuit()
    transpile(qc, basis_gates=["cx", "u"], optimization_level=1)
    timeline = debug.get(qc)
    session = CliSession([timeline])
    app = TimelineApp(session, "viz (test)")
    return app, session, timeline


async def _mark(pilot, slot):
    """Press the ``m`` then ``a``/``b`` chord and let its worker settle."""
    await pilot.press("m")
    await pilot.press(slot)
    await pilot.pause()


async def test_tui_launch_shows_first_event(debug):
    """Tui launch shows first event."""
    app, session, _timeline = _make_app(debug)
    async with app.run_test():
        status = app.query_one("#status")
        assert f"[1/{len(session.merged)}]" in _plain(status)


async def test_tui_next_prev_navigate_and_update_panels(debug):
    """Tui next prev navigate and update panels."""
    app, session, _timeline = _make_app(debug)
    async with app.run_test() as pilot:
        await pilot.press("n")
        assert session.index == 1
        status = app.query_one("#status")
        assert f"[2/{len(session.merged)}]" in _plain(status)

        await pilot.press("p")
        assert session.index == 0
        status = app.query_one("#status")
        assert f"[1/{len(session.merged)}]" in _plain(status)


async def test_tui_arrow_key_highlight_updates_panels_without_enter(debug):
    """Tui arrow key highlight updates panels without enter."""
    app, session, _timeline = _make_app(debug)
    async with app.run_test() as pilot:
        await pilot.press("down")
        await pilot.pause()

        assert session.index == 1
        status = app.query_one("#status")
        assert f"[2/{len(session.merged)}]" in _plain(status)


async def test_tui_circuit_and_info_are_separate_panels(debug):
    """The circuit drawing and the event's other info (description, state,
    histogram, ...) are two distinct widgets, not one blob of text -- both
    stay visible together instead of one pushing the other off-screen."""
    app, _session, _timeline = _make_app(debug)
    async with app.run_test():
        circuit_text = _plain(app.query_one("#circuit"))
        info_text = _plain(app.query_one("#info-text"))
        assert circuit_text != info_text
        assert "q_0" in circuit_text or "q0" in circuit_text
        assert "gate" in info_text.lower() or "h" in info_text.lower()


async def test_tui_mark_chord_pins_slots_and_opens_diff_screen(debug):
    """"m" then "a"/"b" -- not a single keystroke -- pins a diff slot; once
    both are set, "D" opens the diff as its own full-screen view."""
    app, session, _timeline = _make_app(debug)
    async with app.run_test() as pilot:
        await _mark(pilot, "a")
        await pilot.press("n")
        await pilot.press("n")
        await _mark(pilot, "b")

        assert session.marks["a"] is not None
        assert session.marks["b"] is not None

        await pilot.press("D")
        await pilot.pause()
        assert isinstance(app.screen, DiffScreen)
        status_text = _plain(app.screen.query_one("#diff-status"))
        assert "diff ready" in status_text

        await pilot.press("escape")
        assert not isinstance(app.screen, DiffScreen)


async def test_tui_mark_chord_other_key_cancels_and_still_runs(debug):
    """Any key other than a/b after "m" cancels the chord (no mark) and that
    key still does its own normal thing -- the chord never eats it."""
    app, session, _timeline = _make_app(debug)
    async with app.run_test() as pilot:
        await pilot.press("m")
        await pilot.press("x")  # "x" toggles show-all-circuits, unrelated to marking
        await pilot.pause()

        assert session.marks["a"] is None
        assert session.marks["b"] is None
        assert session.show_all_circuits is True


async def test_tui_chord_stays_armed_indefinitely(debug):
    """No auto-cancel timeout -- the chord stays armed for as long as it
    takes until "a"/"b" resolves it."""
    app, session, _timeline = _make_app(debug)
    async with app.run_test() as pilot:
        await pilot.press("m")
        await pilot.pause(2.0)  # longer than the old (now-removed) timeout
        assert app.query_one("#chord-hint").display is True

        await pilot.press("a")
        await pilot.pause()
        assert session.marks["a"] is not None


async def test_tui_chord_escape_cancels_explicitly(debug):
    """Tui chord escape cancels explicitly."""
    app, session, _timeline = _make_app(debug)
    async with app.run_test() as pilot:
        hint = app.query_one("#chord-hint")
        await pilot.press("m")
        assert hint.display is True

        await pilot.press("escape")
        assert hint.display is False
        assert session.marks["a"] is None


async def test_tui_chord_hint_appears_while_armed_and_clears_on_resolve(debug):
    """Pressing "m" alone used to give zero on-screen feedback -- a hint now
    shows while the chord is armed, and clears once "a" completes it."""
    app, _session, _timeline = _make_app(debug)
    async with app.run_test() as pilot:
        hint = app.query_one("#chord-hint")
        assert hint.display is False

        await pilot.press("m")
        assert hint.display is True
        assert "mark" in _plain(hint)

        await pilot.press("a")
        await pilot.pause()
        assert hint.display is False


async def test_tui_chord_hint_clears_on_cancel(debug):
    """Tui chord hint clears on cancel."""
    app, _session, _timeline = _make_app(debug)
    async with app.run_test() as pilot:
        hint = app.query_one("#chord-hint")
        await pilot.press("u")
        assert hint.display is True
        assert "unmark" in _plain(hint)

        await pilot.press("x")  # cancels the chord, still toggles circuits
        assert hint.display is False


async def test_tui_unmark_chord_clears_a_specific_slot(debug):
    """"u" then "a"/"b" mirrors "m" then "a"/"b", clearing that one slot."""
    app, session, _timeline = _make_app(debug)
    async with app.run_test() as pilot:
        await _mark(pilot, "a")
        await pilot.press("n")
        await _mark(pilot, "b")
        assert session.marks["a"] is not None
        assert session.marks["b"] is not None

        await pilot.press("u")
        await pilot.press("a")
        await pilot.pause()

        assert session.marks["a"] is None
        assert session.marks["b"] is not None


async def test_tui_unmark_chord_uu_clears_both_slots(debug):
    """"u" then "u" again (mirroring vim's "dd"/"yy" doubled-key-for-both
    mnemonic) unmarks both slots in one chord, instead of requiring two
    separate "u a" / "u b" chords."""
    app, session, _timeline = _make_app(debug)
    async with app.run_test() as pilot:
        await _mark(pilot, "a")
        await pilot.press("n")
        await _mark(pilot, "b")
        assert session.marks["a"] is not None
        assert session.marks["b"] is not None

        await pilot.press("u")
        await pilot.press("u")
        await pilot.pause()

        assert session.marks["a"] is None
        assert session.marks["b"] is None


async def test_tui_mark_unmark_actions_disabled_off_main_screen(debug):
    """The Footer must stop advertising "Mark (a/b)"/"Unmark (a/b/u)" while
    a non-main screen (help, diff, ...) is on top -- those chords are
    inert there (see test_tui_mark_chord_does_not_fire_while_another_screen_is_open),
    so the Footer shouldn't claim otherwise."""
    app, _session, _timeline = _make_app(debug)
    async with app.run_test() as pilot:
        assert app.check_action("arm_mark", ()) is True
        assert app.check_action("arm_unmark", ()) is True

        await pilot.press("question_mark")
        assert app.screen.__class__.__name__ == "HelpScreen"
        assert app.check_action("arm_mark", ()) is False
        assert app.check_action("arm_unmark", ()) is False


async def test_tui_diff_command_line_opens_diff_screen(debug):
    """Typing "diff" through the ":" command line opens the same full-screen
    diff view as the "D" shortcut, not a toast dump of its (potentially
    long) content."""
    app, session, _timeline = _make_app(debug)
    async with app.run_test() as pilot:
        await _mark(pilot, "a")
        await pilot.press("n")
        await _mark(pilot, "b")

        await pilot.press("colon")
        for char in "diff":
            await pilot.press(char)
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, DiffScreen)
        assert session.diff_expanded is True


async def test_tui_element_picker_shown_when_stage_has_multiple_elements(debug):
    """Marking a stage that has both a construction and a transpiled track
    asks which element to diff, instead of silently picking one."""
    app, session, _timeline = _make_transpiled_app(debug)
    async with app.run_test() as pilot:
        await pilot.press("]")  # the transpiled event, where both elements exist
        await pilot.press("m")
        await pilot.press("a")
        await pilot.pause()

        assert isinstance(app.screen, ElementPickerScreen)
        await pilot.press("enter")
        await pilot.pause()

        assert not isinstance(app.screen, ElementPickerScreen)
        assert session.marks["a"] is not None
        assert session.diff_selection.element_a is not None


async def test_tui_element_picker_previews_highlighted_element_live(debug):
    """Arrowing through the picker's candidates updates its own embedded
    preview pane live, to whichever one is currently highlighted -- not
    just whichever one ends up chosen. This preview lives on the picker's
    own full screen (#picker-preview), not the main screen's #circuit --
    an earlier version wrote into #circuit behind a small floating modal,
    which needed careful size-capping so the modal didn't grow to cover
    the very panel it was meant to sit over; a full screen with its own
    preview pane sidesteps that class of bug entirely."""
    app, _session, _timeline = _make_transpiled_app(debug)
    async with app.run_test() as pilot:
        await pilot.press("]")
        await pilot.press("m")
        await pilot.press("a")
        await pilot.pause()
        assert isinstance(app.screen, ElementPickerScreen)

        before = _plain(app.screen.query_one("#picker-preview"))
        await pilot.press("down")
        await pilot.pause()
        after = _plain(app.screen.query_one("#picker-preview"))

        assert before != after
        await pilot.press("up")
        await pilot.pause()
        assert _plain(app.screen.query_one("#picker-preview")) == before


async def test_tui_element_picker_is_full_screen(debug):
    """The picker is its own full screen (matching DiffScreen), not a
    small centered modal floating over the main screen."""
    app, _session, _timeline = _make_transpiled_app(debug)
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.press("]")
        await pilot.press("m")
        await pilot.press("a")
        await pilot.pause()
        assert isinstance(app.screen, ElementPickerScreen)

        body = app.screen.query_one("#picker-body")
        assert body.size.width == 140


async def test_tui_element_picker_skipped_for_single_element_stage(debug):
    """A stage with only one candidate element marks straight away, no prompt."""
    app, session, _timeline = _make_app(debug)
    async with app.run_test() as pilot:
        await _mark(pilot, "a")

        assert not isinstance(app.screen, ElementPickerScreen)
        assert session.marks["a"] is not None
        assert session.diff_selection.element_a is None


async def test_tui_guided_watch_add_via_w_shortcut(debug):
    """"w" opens the guided step wizard -- typing the exact ``watch``
    syntax with embedded spaces and pressing Enter commits both tokens at
    once and closes automatically, since a watch's ``kind ref`` shape has
    nothing left to add after that."""
    app, session, _timeline = _make_app(debug)
    async with app.run_test() as pilot:
        await pilot.press("w")
        await pilot.pause()
        assert isinstance(app.screen, GuidedWizardScreen)

        for char in "qubit 0":
            await pilot.press(char if char != " " else "space")
        await pilot.press("enter")
        await pilot.pause()

        assert not isinstance(app.screen, GuidedWizardScreen)
        assert len(session.watches) == 1
        assert session.watches[0].kind == "qubit"
        assert session.watches[0].ref == "0"


async def test_tui_watch_wizard_tab_completes_kind_then_ref(debug):
    """Tab completes the kind from its unique prefix, then completes the
    ref from whatever the chosen kind offers -- committing the kind token
    with Enter leaves the wizard open on the next step (a ref is still
    needed) rather than closing early."""
    app, session, _timeline = _make_app(debug)
    async with app.run_test() as pilot:
        await pilot.press("w")
        await pilot.pause()
        for char in "r":  # "r" uniquely prefixes "register" among the kinds
            await pilot.press(char)
        await pilot.press("tab")
        await pilot.pause()

        input_widget = app.screen.query_one("#cmd-input", Input)
        assert input_widget.value == "register"

        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, GuidedWizardScreen)

        for char in "q":  # this bell circuit's quantum register is named "q"
            await pilot.press(char)
        await pilot.press("tab")
        await pilot.pause()
        assert input_widget.value == "q"

        await pilot.press("enter")
        await pilot.pause()
        assert not isinstance(app.screen, GuidedWizardScreen)
        assert len(session.watches) == 1
        assert session.watches[0].kind == "register"
        assert session.watches[0].ref == "q"


async def test_tui_watch_wizard_cancel_via_escape(debug):
    """Escape closes the wizard without adding anything when nothing has
    been committed yet, mirroring every other modal screen in this app."""
    app, session, _timeline = _make_app(debug)
    async with app.run_test() as pilot:
        await pilot.press("w")
        await pilot.pause()
        assert isinstance(app.screen, GuidedWizardScreen)

        await pilot.press("escape")
        await pilot.pause()

        assert not isinstance(app.screen, GuidedWizardScreen)
        assert len(session.watches) == 0


async def test_tui_watch_wizard_escape_steps_back_one_token(debug):
    """Escape after a token is already committed backs up one step instead
    of closing outright, so a wrong pick doesn't mean starting over."""
    app, _session, _timeline = _make_app(debug)
    async with app.run_test() as pilot:
        await pilot.press("w")
        await pilot.pause()
        for char in "qubit":
            await pilot.press(char)
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, GuidedWizardScreen)

        await pilot.press("escape")
        await pilot.pause()

        assert isinstance(app.screen, GuidedWizardScreen)
        breadcrumb = app.screen.query_one("#cmd-breadcrumb")
        assert str(breadcrumb.render()) == "(nothing chosen yet)"


async def test_tui_guided_breakpoint_add_enumerable_kind(debug):
    """A fully-enumerable breakpoint kind (anomaly) needs no free text --
    type it, Enter, then pick the pinned "finish" entry once the only
    thing left on offer is the optional 'on' clause."""
    app, session, _timeline = _make_app(debug)
    async with app.run_test() as pilot:
        await pilot.press("b")
        await pilot.pause()
        assert isinstance(app.screen, GuidedWizardScreen)

        for char in "anomaly warning":
            await pilot.press(char if char != " " else "space")
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, GuidedWizardScreen)  # 'on' is still optional

        await pilot.press("down")  # move off 'on' onto the pinned finish entry
        await pilot.press("enter")
        await pilot.pause()

        assert not isinstance(app.screen, GuidedWizardScreen)
        rows = session.breakpoint_rows()
        assert len(rows) == 1
        assert "anomaly" in rows[0][1]


async def test_tui_guided_breakpoint_add_needs_free_text(debug):
    """A "seq" breakpoint mixes an enumerable comparator with a free-text
    event number, typed straight into the wizard like any other command,
    then finished explicitly since 'on' remains optional afterward."""
    app, session, _timeline = _make_app(debug)
    async with app.run_test() as pilot:
        await pilot.press("b")
        await pilot.pause()

        for char in "seq >= 3":
            await pilot.press(char if char != " " else "space")
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("down")
        await pilot.press("enter")
        await pilot.pause()

        rows = session.breakpoint_rows()
        assert len(rows) == 1
        assert "seq" in rows[0][1]
        assert "3" in rows[0][1]


async def test_tui_breakpoint_wizard_stays_open_after_comparator_for_threshold(debug):
    """Regression: choosing a comparator one step at a time (rather than
    typing "seq >= 3" in one go) must not auto-close the wizard before the
    threshold is typed -- the threshold step has nothing to suggest, but
    "nothing to suggest" isn't "nothing left to type"."""
    app, session, _timeline = _make_app(debug)
    async with app.run_test() as pilot:
        await pilot.press("b")
        await pilot.pause()
        for char in "seq":
            await pilot.press(char)
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, GuidedWizardScreen)

        for char in ">=":
            await pilot.press(char)
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, GuidedWizardScreen)  # must still be open
        assert len(session.breakpoint_rows()) == 0

        for char in "3":
            await pilot.press(char)
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("down")
        await pilot.press("enter")
        await pilot.pause()

        rows = session.breakpoint_rows()
        assert len(rows) == 1
        assert "seq" in rows[0][1]
        assert "3" in rows[0][1]


async def test_tui_breakpoint_wizard_tab_completes_kind(debug):
    """Tab completes the breakpoint kind from a unique prefix, mirroring
    the typed ``:`` command line's own completion contract."""
    app, _session, _timeline = _make_app(debug)
    async with app.run_test() as pilot:
        await pilot.press("b")
        await pilot.pause()
        for char in "an":
            await pilot.press(char)
        await pilot.press("tab")
        await pilot.pause()

        input_widget = app.screen.query_one("#cmd-input", Input)
        assert input_widget.value == "anomaly"


async def test_tui_breakpoint_wizard_blank_enter_picks_top_of_list(debug):
    """With nothing typed, Enter accepts whatever is highlighted at the
    top of the always-visible option list -- the wizard's default choice
    -- and advances a step, rather than erroring on a blank line."""
    app, _session, _timeline = _make_app(debug)
    async with app.run_test() as pilot:
        await pilot.press("b")
        await pilot.pause()

        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, GuidedWizardScreen)
        breadcrumb = app.screen.query_one("#cmd-breadcrumb")
        assert str(breadcrumb.render()) == "gate"


async def test_tui_breakpoint_wizard_cancel_via_escape(debug):
    """Escape closes the breakpoint wizard without adding anything when
    nothing has been committed yet."""
    app, session, _timeline = _make_app(debug)
    async with app.run_test() as pilot:
        await pilot.press("b")
        await pilot.pause()
        assert isinstance(app.screen, GuidedWizardScreen)

        await pilot.press("escape")
        await pilot.pause()

        assert not isinstance(app.screen, GuidedWizardScreen)
        assert len(session.breakpoint_rows()) == 0


async def test_tui_remove_breakpoint_via_shift_b(debug):
    """"B" opens the same wizard as "b", scoped to removing a breakpoint
    by handle -- the CLI counterpart to the GUI's "Remove selected"
    button."""
    app, session, _timeline = _make_app(debug)
    async with app.run_test() as pilot:
        await pilot.press("b")
        await pilot.pause()
        for char in "anomaly":
            await pilot.press(char)
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("down", "down", "down", "down")
        await pilot.press("enter")
        await pilot.pause()
        assert len(session.breakpoint_rows()) == 1

        await pilot.press("B")
        await pilot.pause()
        assert isinstance(app.screen, GuidedWizardScreen)
        for char in "1":
            await pilot.press(char)
        await pilot.press("enter")
        await pilot.pause()

        assert len(session.breakpoint_rows()) == 0


async def test_tui_remove_breakpoint_shortcut_noop_with_none_active(debug):
    """"B" with no active breakpoints toasts instead of opening an empty,
    useless wizard."""
    app, _session, _timeline = _make_app(debug)
    async with app.run_test() as pilot:
        await pilot.press("B")
        await pilot.pause()
        assert not isinstance(app.screen, GuidedWizardScreen)


async def test_tui_remove_watch_via_shift_w(debug):
    """"W" opens the same kind of wizard as "w", scoped to removing a
    watch by kind/ref -- suggestions only ever include an actually active
    watch, never any valid-but-unset one."""
    app, session, _timeline = _make_app(debug)
    async with app.run_test() as pilot:
        await pilot.press("w")
        await pilot.pause()
        for char in "qubit 0":
            await pilot.press(char if char != " " else "space")
        await pilot.press("enter")
        await pilot.pause()
        assert len(session.watches) == 1

        await pilot.press("W")
        await pilot.pause()
        assert isinstance(app.screen, GuidedWizardScreen)
        for char in "qubit 0":
            await pilot.press(char if char != " " else "space")
        await pilot.press("enter")
        await pilot.pause()

        assert len(session.watches) == 0


async def test_tui_remove_watch_shortcut_noop_with_none_active(debug):
    """"W" with no active watches toasts instead of opening an empty,
    useless wizard."""
    app, _session, _timeline = _make_app(debug)
    async with app.run_test() as pilot:
        await pilot.press("W")
        await pilot.pause()
        assert not isinstance(app.screen, GuidedWizardScreen)


async def test_tui_command_palette_runs_watch_and_dismisses(debug):
    """":" opens the transient command line, Enter runs it and closes the
    overlay -- there is no permanently docked command box any more."""
    app, session, _timeline = _make_app(debug)
    async with app.run_test() as pilot:
        await pilot.press("colon")
        assert isinstance(app.screen, CommandScreen)
        for char in "watch qubit 0":
            await pilot.press(char if char != " " else "space")
        await pilot.press("enter")

        assert not isinstance(app.screen, CommandScreen)
        assert len(session.watches) == 1
        watches_text = _plain(app.query_one("#watches"))
        assert "qubit 0" in watches_text


async def test_tui_command_palette_escape_cancels(debug):
    """Tui command palette escape cancels."""
    app, session, _timeline = _make_app(debug)
    async with app.run_test() as pilot:
        await pilot.press("colon")
        await pilot.press("n")  # types "n" into the command line, not "next"
        await pilot.press("escape")

        assert not isinstance(app.screen, CommandScreen)
        assert session.index == 0
        assert not session.watches


async def test_tui_question_mark_opens_help_from_normal_navigation(debug):
    """Regression test: "?" used to be swallowed by whichever widget had
    focus and never reach the app-level help binding."""
    app, _session, _timeline = _make_app(debug)
    async with app.run_test() as pilot:
        await pilot.press("question_mark")
        assert app.screen.__class__.__name__ == "HelpScreen"


async def test_tui_question_mark_closes_help_when_already_open(debug):
    """"?" toggles: pressed again while help is open, it closes it instead
    of stacking a second help screen or doing nothing."""
    app, _session, _timeline = _make_app(debug)
    async with app.run_test() as pilot:
        await pilot.press("question_mark")
        assert app.screen.__class__.__name__ == "HelpScreen"
        await pilot.press("question_mark")
        assert app.screen.__class__.__name__ != "HelpScreen"


async def test_tui_toggle_circuits_expands_and_collapses_circuit_panel(debug):
    """Tui toggle circuits expands and collapses circuit panel."""
    app, session, _timeline = _make_app(debug)
    async with app.run_test() as pilot:
        assert session.show_all_circuits is False
        await pilot.press("x")
        assert session.show_all_circuits is True
        await pilot.press("x")
        assert session.show_all_circuits is False


async def test_tui_zoom_keys_change_fold_width(debug):
    """Tui zoom keys change fold width."""
    app, session, _timeline = _make_app(debug)
    async with app.run_test() as pilot:
        default_width = session.fold_width
        await pilot.press("minus")
        assert session.fold_width != default_width
        await pilot.press("0")
        assert session.fold_width == default_width


async def test_tui_bracket_keys_jump_to_first_and_last(debug):
    """"[" / "]" replace Home/End -- plenty of laptop keyboards lack those."""
    app, session, _timeline = _make_app(debug)
    async with app.run_test() as pilot:
        await pilot.press("]")
        assert session.index == len(session.merged) - 1
        await pilot.press("[")
        assert session.index == 0


async def test_tui_main_screen_shortcuts_do_not_leak_into_help_screen(debug):
    """While help is on top, "n" must drive nothing underneath it -- keys
    are scoped to whichever screen is actually being looked at."""
    app, session, _timeline = _make_app(debug)
    async with app.run_test() as pilot:
        await pilot.press("question_mark")
        assert app.screen.__class__.__name__ == "HelpScreen"
        await pilot.press("n")
        await pilot.press("]")
        assert session.index == 0
        assert app.screen.__class__.__name__ == "HelpScreen"


async def test_tui_mark_chord_does_not_fire_while_another_screen_is_open(debug):
    """"m" then "a" while a non-main screen is on top must not mark
    anything -- the chord is scoped to the main timeline screen."""
    app, session, _timeline = _make_app(debug)
    async with app.run_test() as pilot:
        await pilot.press("question_mark")
        assert app.screen.__class__.__name__ == "HelpScreen"
        await pilot.press("m")
        await pilot.press("a")
        await pilot.pause()
        assert session.marks["a"] is None


async def test_tui_main_screen_shortcuts_do_not_leak_into_diff_screen(debug):
    """Same scoping, exercised against the diff screen: "x" (toggle all
    circuits) must not touch the underlying session while diff is shown."""
    app, session, _timeline = _make_app(debug)
    async with app.run_test() as pilot:
        await _mark(pilot, "a")
        await pilot.press("n")
        await _mark(pilot, "b")
        await pilot.press("D")
        await pilot.pause()
        assert isinstance(app.screen, DiffScreen)

        await pilot.press("x")
        assert session.show_all_circuits is False


async def test_tui_diff_indicator_appears_once_both_marks_set(debug):
    """The one-line "diff ready" indicator replaces the old inline diff pane."""
    app, session, _timeline = _make_app(debug)
    async with app.run_test() as pilot:
        indicator = app.query_one("#diff-indicator")
        assert indicator.display is False

        await _mark(pilot, "a")
        await pilot.press("n")
        await _mark(pilot, "b")

        assert indicator.display is True
        assert "diff ready" in _plain(indicator)
        assert session.diff_ready() is True
