"""Toolkit-free timeline state + view-model builder, shared by the web GUI.

Everything :class:`TimelineBridge` does is plain Python operating on plain
data (dicts/lists/strings) -- no ``pywebview``, no DOM, no widgets -- so it
is unit-testable exactly like :mod:`._render` and usable from a script
without ever opening a window. ``gui.py`` wraps one instance as a
``pywebview`` ``js_api`` object: every public method here is called
directly from ``app.js`` and returns a JSON-serializable dict, mirroring
how the old ``_TimelineWindow`` (Qt) translated the same underlying state
into widget updates. Autoplay is intentionally *not* implemented here --
the browser side runs its own ``setInterval`` calling :meth:`goto` each
tick and stops when the returned state says ``at_last``, since a webview
has no equivalent of Qt's ``QTimer``/event-filter machinery to abstract
over and reinventing one would add a layer for no benefit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from .._events import Timeline
from . import _breakpoints as _bp
from . import _circuit_diff, _render

MIN_CIRCUIT_ZOOM = 6
MAX_CIRCUIT_ZOOM = 24
DEFAULT_CIRCUIT_ZOOM = 13
CIRCUIT_ZOOM_STEP = 1

HELP_ENTRIES: list[tuple[str, str]] = [
    ("← / →", "previous / next event"),
    ("Space", "play / pause"),
    ("+ / -", "zoom the circuit drawing in / out"),
    ("0", "reset circuit drawing zoom"),
    ("Ctrl/⌘ + scroll", "zoom the circuit drawing (over the drawing pane)"),
    ("?", "this help"),
]

_BREAKPOINT_SCOPE_CURRENT = "(current)"
_BREAKPOINT_SCOPE_ALL = "(all)"


@dataclass
class TimelineBridge:  # pylint: disable=too-many-instance-attributes
    """Owns one merged timeline's interactive state (playhead, watches,
    breakpoints, marks) and turns it into JSON view-models for the web GUI.

    One instance per window, exactly like ``_TimelineWindow`` owned one Qt
    window's worth of state; ``gui.py`` constructs it once per
    :meth:`~.base.TimelineFrontend.show` call.
    """

    title: str
    merged: list[tuple[Timeline, Any]]
    tracks: list[dict[str, str]]
    active_labels: list[str | None]
    timelines_by_id: dict[str, Timeline]

    index: int = field(default=0, init=False)
    show_all_circuits: bool = field(default=False, init=False)
    circuit_zoom: int = field(default=DEFAULT_CIRCUIT_ZOOM, init=False)
    watches: list[_render.WatchTarget] = field(default_factory=list, init=False)
    marks: dict[str, _render.StageMark | None] = field(
        default_factory=lambda: {"a": None, "b": None}, init=False
    )
    breakpoints: _bp.FrontendBreakpoints = field(
        default_factory=_bp.FrontendBreakpoints, init=False
    )
    diff_elements: dict[str, str | None] = field(
        default_factory=lambda: {"a": None, "b": None}, init=False
    )
    hit_status: str = field(default="", init=False)

    # -- Navigation ----------------------------------------------------

    @property
    def last_index(self) -> int:
        """The highest valid index into :attr:`merged`."""
        return len(self.merged) - 1

    @property
    def current(self) -> tuple[Timeline, Any]:
        """The (source timeline, event) pair at the current playhead."""
        return self.merged[self.index]

    def goto(self, index: int) -> dict[str, Any]:
        """Move the playhead to ``index`` (clamped) and return the new state."""
        self.index = min(max(int(index), 0), self.last_index)
        return self.state()

    def set_show_all_circuits(self, value: bool) -> dict[str, Any]:
        """Toggle between showing only the active track and every tracked
        circuit's every track."""
        self.show_all_circuits = bool(value)
        return self.state()

    def zoom_in(self) -> dict[str, Any]:
        """Increase the circuit drawing's font size by one step."""
        return self._set_zoom(self.circuit_zoom + CIRCUIT_ZOOM_STEP)

    def zoom_out(self) -> dict[str, Any]:
        """Decrease the circuit drawing's font size by one step."""
        return self._set_zoom(self.circuit_zoom - CIRCUIT_ZOOM_STEP)

    def zoom_reset(self) -> dict[str, Any]:
        """Reset the circuit drawing's font size to the default."""
        return self._set_zoom(DEFAULT_CIRCUIT_ZOOM)

    def _set_zoom(self, size: int) -> dict[str, Any]:
        self.circuit_zoom = max(MIN_CIRCUIT_ZOOM, min(MAX_CIRCUIT_ZOOM, size))
        return self.state()

    # -- Main view-model -------------------------------------------------

    def state(self) -> dict[str, Any]:
        """The full view-model for the current playhead position.

        Returned wholesale after every mutating call (goto, add/remove
        watch or breakpoint, mark, zoom, ...) rather than as separate
        incremental patches -- the timeline is small enough per event that
        recomputing the whole thing is simpler than tracking which fields
        actually changed, mirroring how ``_TimelineWindow._render``
        refreshed every widget unconditionally on each step.
        """
        source, event = self.current
        header = (
            f"{self.index + 1}/{len(self.merged)} — {source.name} — "
            f"{event.type.value} at {event.timestamp:.3f} — {event.describe()}"
        )

        other_circuits: list[str] = []
        if self.show_all_circuits:
            tracks_text, active_span = _render.render_tracks_text_with_active(
                self.tracks[self.index], self.active_labels[self.index]
            )
        else:
            tracks_text, other_circuits = _render.split_active_track(
                self.tracks[self.index], self.active_labels[self.index]
            )
            active_span = None

        label, png_bytes = _render.render_extra_visual_bytes(event)
        extra_visual = None
        if png_bytes:
            import base64

            extra_visual = {
                "label": label,
                "data_uri": "data:image/png;base64,"
                + base64.b64encode(png_bytes).decode("ascii"),
            }

        return {
            "title": self.title,
            "header": header,
            "index": self.index,
            "count": len(self.merged),
            "at_first": self.index <= 0,
            "at_last": self.index >= self.last_index,
            "circuit_text": tracks_text,
            "circuit_zoom": self.circuit_zoom,
            "active_span": list(active_span) if active_span else None,
            "show_all_circuits": self.show_all_circuits,
            "other_circuits": other_circuits,
            "info_text": _render.render_event(source, event),
            "extra_visual": extra_visual,
            "watches": self._watch_state(),
            "marks": self._mark_state(),
            "breakpoints": self._breakpoint_state(),
            "hit_status": self.hit_status,
        }

    def help_entries(self) -> list[tuple[str, str]]:
        """Keyboard shortcut table for the help overlay."""
        return HELP_ENTRIES

    # -- Watches -----------------------------------------------------------

    def watch_ref_choices(self, kind: str) -> list[str]:
        """Fill-in choices for the watch-ref dropdown, given a selected kind."""
        source, event = self.current
        if kind == "qubit":
            return [str(i) for i in range(source.num_qubits)]
        if kind == "clbit":
            return [str(i) for i in range(source.num_clbits)]
        if kind == "register":
            return [*source.qregs, *source.cregs]
        return _render.track_keys_upto(source, event.seq)

    def add_watch(self, kind: str, ref: str) -> dict[str, Any]:
        """Add a qubit/clbit/register/track watch, if not already present."""
        if not ref or kind not in ("qubit", "clbit", "register", "track"):
            return self.state()
        source, _event = self.current
        target = _render.WatchTarget(circuit_id=source.circuit_id, kind=kind, ref=ref)
        if target not in self.watches:
            self.watches.append(target)
        return self.state()

    def remove_watch(self, watch_index: int) -> dict[str, Any]:
        """Remove the watch at ``watch_index`` in :attr:`watches`."""
        if 0 <= watch_index < len(self.watches):
            del self.watches[watch_index]
        return self.state()

    def _watch_state(self) -> dict[str, Any]:
        """Per-watch structured cards, not one preformatted text blob --
        each watch gets its own label, kind (for a badge), and detail
        lines, so the web GUI can render an actual card per watch instead
        of a single ``<pre>`` transcript (the CLI/TUI still get the plain
        text form via :func:`~._render.render_watches` directly, unaffected
        by this).
        """
        _source, event = self.current
        cards = []
        for i, target in enumerate(self.watches):
            timeline = self.timelines_by_id.get(target.circuit_id)
            rendered = _render.render_watch(target, timeline, event.seq) if timeline else ""
            # render_watch's first line is always "* <label>" (see its
            # docstring); the label is shown as the card title separately,
            # so only the indented detail lines become the card body.
            detail_lines = rendered.split("\n")[1:]
            cards.append(
                {
                    "index": i,
                    "kind": target.kind,
                    "label": target.label(timeline),
                    "lines": [line.strip() for line in detail_lines],
                }
            )
        return {
            "list": cards,
            "kind_choices": ["qubit", "clbit", "register", "track"],
            "ref_choices": self.watch_ref_choices("qubit"),
        }

    # -- Marks / diff --------------------------------------------------

    def mark(self, slot: Literal["a", "b"]) -> dict[str, Any]:
        """Pin the current playhead's stage as mark ``slot``."""
        _source, event = self.current
        self.marks[slot] = _render.StageMark(seq=event.seq)
        self._reset_diff_choice(slot)
        return self.state()

    def clear_marks(self) -> dict[str, Any]:
        """Unpin both marks."""
        self.marks = {"a": None, "b": None}
        self.diff_elements = {"a": None, "b": None}
        return self.state()

    def _reset_diff_choice(self, slot: str) -> None:
        mark = self.marks[slot]
        elements = mark.elements_at(self.timelines_by_id) if mark else []
        if self.diff_elements[slot] not in elements:
            self.diff_elements[slot] = elements[-1] if elements else None

    def set_diff_element(self, slot: Literal["a", "b"], element: str) -> dict[str, Any]:
        """Change which ``"<circuit>:<track>"`` element diff slot ``slot`` compares."""
        self.diff_elements[slot] = element or None
        return self.state()

    def _mark_state(self) -> dict[str, Any]:
        mark_a, mark_b = self.marks["a"], self.marks["b"]
        both = mark_a is not None and mark_b is not None
        result: dict[str, Any] = {
            "label_a": mark_a.label(self.timelines_by_id) if mark_a else "(unmarked)",
            "label_b": mark_b.label(self.timelines_by_id) if mark_b else "(unmarked)",
            "both_marked": both,
            "choices_a": mark_a.elements_at(self.timelines_by_id) if mark_a else [],
            "choices_b": mark_b.elements_at(self.timelines_by_id) if mark_b else [],
            "selected_a": self.diff_elements["a"],
            "selected_b": self.diff_elements["b"],
            "track_label": {},
        }
        for element in {*result["choices_a"], *result["choices_b"]}:
            result["track_label"][element] = _render.track_label(element)
        if mark_a is None or mark_b is None:
            result["diff"] = None
            return result

        resolved = _circuit_diff.resolve_split_and_stage_diff(
            mark_a,
            mark_b,
            self.timelines_by_id,
            element_a=self.diff_elements["a"],
            element_b=self.diff_elements["b"],
        )
        if isinstance(resolved, str):
            result["diff"] = {"message": resolved}
            return result
        split_diff, stage_diff_text = resolved
        watch_diff_text = _render.render_watch_diff(
            self.watches, self.timelines_by_id, mark_a.seq, mark_b.seq
        )
        result["diff"] = {
            "message": None,
            "stage_diff": stage_diff_text,
            "watch_diff": watch_diff_text,
            "side_a": {
                "label": split_diff.a.label,
                "text": split_diff.a.text,
                "spans": split_diff.a.spans,
            },
            "side_b": {
                "label": split_diff.b.label,
                "text": split_diff.b.text,
                "spans": split_diff.b.spans,
            },
        }
        return result

    # -- Breakpoints ---------------------------------------------------

    def breakpoint_field_choices(self) -> dict[str, Any]:
        """Dynamic field choices (autocomplete lists, dropdown contents) for
        the breakpoint condition builder, for whichever circuit is currently
        on screen -- mirrors ``_TimelineWindow._refresh_breakpoint_fields``.
        """
        source, _event = self.current
        scope_choices = [
            {"value": "", "label": _BREAKPOINT_SCOPE_CURRENT},
            {"value": "*", "label": _BREAKPOINT_SCOPE_ALL},
        ]
        for circuit_id in sorted(self.timelines_by_id):
            name = self.timelines_by_id[circuit_id].name
            label = f"{name} ({circuit_id})" if name else circuit_id
            scope_choices.append({"value": circuit_id, "label": label})
        return {
            "kinds": list(_bp.KINDS),
            "comparators": list(_bp.COMPARATOR_SYMBOLS),
            "watch_kinds": list(_bp.WATCH_KINDS),
            "paulis": list(_bp.PAULIS),
            "severities": list(_bp.SEVERITIES),
            "gate_names": _bp.known_gate_names(source),
            "outcomes": _bp.small_outcome_space(source),
            "qubits": _bp.qubit_range(source),
            "clbits": _bp.clbit_range(source),
            "registers": _bp.register_names(source),
            "scopes": scope_choices,
        }

    def add_breakpoint(self, kind: str, tokens: list[str], scope: str) -> dict[str, Any]:
        """Parse and register a breakpoint from the condition-builder fields.

        ``tokens`` is the flat argument list for ``kind`` exactly as
        :func:`~._breakpoints.parse_condition_args` expects (the browser
        assembles it client-side per kind, mirroring
        ``_TimelineWindow._breakpoint_tokens``); ``scope`` is ``""`` for
        "current circuit", ``"*"`` for all, or a circuit id.
        """
        source, _event = self.current
        args = list(tokens)
        if scope:
            args += ["on", scope]
        try:
            condition = _bp.parse_condition_args(
                kind, args, source.circuit_id, self.timelines_by_id
            )
        except ValueError as exc:
            state = self.state()
            state["breakpoint_error"] = str(exc)
            return state
        self.breakpoints.add(condition)
        return self.state()

    def remove_breakpoint(self, handle: str) -> dict[str, Any]:
        """Remove a previously registered breakpoint by its handle."""
        self.breakpoints.remove(handle)
        return self.state()

    def _breakpoint_state(self) -> dict[str, Any]:
        return {
            "list": [
                {
                    "handle": handle,
                    "label": _bp.describe_condition(condition, self.timelines_by_id),
                }
                for handle, condition in self.breakpoints.items()
            ],
        }

    def _goto_hit(
        self, hit: tuple[int, str] | None, *, none_message: str
    ) -> dict[str, Any]:
        if hit is None:
            self.hit_status = none_message
            return self.state()
        index, handle = hit
        condition = self.breakpoints.get(handle)
        assert condition is not None
        self.hit_status = (
            f"hit breakpoint [{handle}] "
            f"{_bp.describe_condition(condition, self.timelines_by_id)}"
        )
        return self.goto(index)

    _HIT_FINDERS: ClassVar[dict[str, Any]] = {
        "first": lambda bridge: _bp.find_first_hit(bridge.merged, bridge.breakpoints.as_dict()),
        "prev": lambda bridge: _bp.find_prev_hit(
            bridge.merged, bridge.breakpoints.as_dict(), bridge.index
        ),
        "next": lambda bridge: _bp.find_next_hit(
            bridge.merged, bridge.breakpoints.as_dict(), bridge.index
        ),
        "last": lambda bridge: _bp.find_last_hit(bridge.merged, bridge.breakpoints.as_dict()),
    }
    _HIT_NONE_MESSAGES: ClassVar[dict[str, str]] = {
        "first": "no event matches any breakpoint",
        "prev": "no earlier event matches any breakpoint",
        "next": "no later event matches any breakpoint",
        "last": "no event matches any breakpoint",
    }

    def goto_hit(self, direction: Literal["first", "prev", "next", "last"]) -> dict[str, Any]:
        """Jump to the first/previous/next/last event matching any active breakpoint."""
        if not self.breakpoints:
            self.hit_status = "no active breakpoints"
            return self.state()
        return self._goto_hit(
            self._HIT_FINDERS[direction](self),
            none_message=self._HIT_NONE_MESSAGES[direction],
        )
