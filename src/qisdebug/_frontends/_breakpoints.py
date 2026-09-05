"""Breakpoints as a navigation aid over an already-captured timeline, plus
the frontend-facing syntax both ``cli.py`` and ``gui.py`` build conditions
from.

"Jump to next/prev/first/last hit" reuses the exact same
:func:`~qisdebug._breakpoints.matches` the live pause path in
:mod:`qisdebug._store` checks against -- the only difference is this walks a
merged, already-recorded event list instead of blocking anything live, so a
breakpoint set while replaying a captured run means the same thing it would
have meant live.

The condition syntax is a small, symbol-comparator grammar (``expect Z 0 >=
0.5``, not ``watch qubit 0 gt 0.9``) plus a suggestion layer
(:func:`suggest_tokens` and the enumeration helpers it's built from) so both
frontends can offer tab-completion/autocomplete without maintaining two
independent notions of what's valid to type next.
"""

from __future__ import annotations

import re
from typing import Callable, Literal, cast

from pydantic import ValidationError

from .._breakpoints import (
    AnomalyCondition,
    BreakpointCondition,
    Comparator,
    DistributionDistanceCondition,
    ExpectationCondition,
    GateMatchCondition,
    ProbabilityCondition,
    Severity,
    SeqThresholdCondition,
    WatchThresholdCondition,
    matches,
)
from .._events import TelemetryEvent, Timeline

Merged = list[tuple[Timeline, TelemetryEvent]]

KINDS: tuple[str, ...] = (
    "gate",
    "seq",
    "watch",
    "expect",
    "prob",
    "tvd",
    "anomaly",
)
WATCH_KINDS: tuple[str, ...] = ("qubit", "clbit", "register")
PAULIS: tuple[str, ...] = ("X", "Y", "Z")
SEVERITIES: tuple[Severity, ...] = ("info", "warning", "error")

COMPARATOR_SYMBOLS: dict[str, Comparator] = {
    ">=": "ge",
    "<=": "le",
    "==": "eq",
    "!=": "ne",
    ">": "gt",
    "<": "lt",
}
_SYMBOL_FOR_COMPARATOR: dict[Comparator, str] = {
    v: k for k, v in COMPARATOR_SYMBOLS.items()
}
_OUTCOME_PAIR_RE = re.compile(r"^[01]+=[0-9.eE+-]+$")

CONDITION_USAGE = (
    "gate NAME [Q1 Q2 ...] | seq CMP N | "
    "watch qubit|clbit|register REF CMP THRESHOLD | expect X|Y|Z QUBIT CMP THRESHOLD | "
    "prob OUTCOME CMP THRESHOLD | tvd OUTCOME=PROB [OUTCOME=PROB ...] CMP THRESHOLD | "
    "anomaly [info|warning|error] -- each may end with 'on CIRCUIT' or 'on *' "
    "(CMP is one of >= <= == != > <)"
)


class FrontendBreakpoints:
    """A frontend session's set of breakpoint conditions plus handle
    assignment -- purely for "jump to hit" navigation, no live pause (see
    :class:`qisdebug._store.BreakpointRegistry` for that). Bundling the
    condition dict and handle counter into one object, shared by ``cli.py``
    and ``gui.py``, keeps each frontend's session class down to one
    attribute for this feature instead of two, and means the two frontends
    can't drift on how handles are assigned.
    """

    def __init__(self) -> None:
        self._conditions: dict[str, BreakpointCondition] = {}
        self._next_id = 0

    def add(self, condition: BreakpointCondition) -> str:
        """Register a condition, returning a handle for :meth:`remove`."""
        self._next_id += 1
        handle = str(self._next_id)
        self._conditions[handle] = condition
        return handle

    def remove(self, handle: str) -> None:
        """Unregister a breakpoint by its handle. No-op if already removed."""
        self._conditions.pop(handle, None)

    def clear(self) -> None:
        """Remove every registered breakpoint."""
        self._conditions.clear()

    def items(self) -> list[tuple[str, BreakpointCondition]]:
        """Return every active ``(handle, condition)`` pair."""
        return list(self._conditions.items())

    def as_dict(self) -> dict[str, BreakpointCondition]:
        """Return a plain ``{handle: condition}`` copy, for the ``find_*_hit`` functions."""
        return dict(self._conditions)

    def get(self, handle: str) -> BreakpointCondition | None:
        """Return the condition for ``handle``, or ``None`` if not registered."""
        return self._conditions.get(handle)

    def __bool__(self) -> bool:
        return bool(self._conditions)

    def __len__(self) -> int:
        return len(self._conditions)

    def __contains__(self, handle: str) -> bool:
        return handle in self._conditions


# -- Enumeration helpers, shared by CLI tab-completion and GUI dropdowns ---


def known_gate_names(timeline: Timeline) -> list[str]:
    """Return every distinct gate name seen on ``timeline``, sorted."""
    return sorted({event.gate for event in timeline.gate_events})


def qubit_range(timeline: Timeline) -> list[str]:
    """Return every valid qubit index on ``timeline``, as strings."""
    return [str(i) for i in range(timeline.num_qubits)]


def clbit_range(timeline: Timeline) -> list[str]:
    """Return every valid clbit index on ``timeline``, as strings."""
    return [str(i) for i in range(timeline.num_clbits)]


def register_names(timeline: Timeline) -> list[str]:
    """Return every qreg/creg name on ``timeline``."""
    return [*timeline.qregs.keys(), *timeline.cregs.keys()]


def circuit_choices(timelines_by_id: dict[str, Timeline]) -> list[str]:
    """Return every circuit's display label currently known, for the
    ``on CIRCUIT`` clause -- each circuit's name where it has one, falling
    back to its otherwise-hidden id only when it doesn't, never the raw id
    when a name is available (per the project's rule that ``circuit_id`` is
    internal plumbing, not part of the public surface)."""
    return sorted({timeline.name or circuit_id for circuit_id, timeline in timelines_by_id.items()})


def _resolve_circuit_by_label(
    label: str, timelines_by_id: dict[str, Timeline]
) -> Timeline | None:
    """Resolve a typed ``on CIRCUIT`` label back to its timeline, checking
    the (hidden) id first and falling back to a name match -- mirroring
    :meth:`~qisdebug._store.TelemetryStore.get_by_name`'s "most recently
    touched wins" rule for the (rare) case of two circuits sharing a name.
    """
    if label in timelines_by_id:
        return timelines_by_id[label]
    for timeline in reversed(list(timelines_by_id.values())):
        if timeline.name == label:
            return timeline
    return None


# -- Comparator/threshold/scope parsing, shared by every numeric kind -----


def _split_on_clause(args: list[str]) -> tuple[list[str], str | None]:
    """Strip a trailing ``on CIRCUIT``/``on *`` clause off ``args``.

    Returns the remaining args and the circuit override (``None`` if no
    clause was present; ``"*"`` means "any circuit", handled by the caller).
    """
    if len(args) >= 2 and args[-2] == "on":
        return args[:-2], args[-1]
    return args, None


def _parse_trailing_comparator(
    args: list[str], usage: str
) -> tuple[list[str], Comparator, float]:
    """Split ``[...leading, CMP, THRESHOLD]`` into ``(leading, comparator, threshold)``."""
    if len(args) < 2 or args[-2] not in COMPARATOR_SYMBOLS:
        raise ValueError(usage)
    *leading, symbol, threshold_arg = args
    try:
        threshold = float(threshold_arg)
    except ValueError as exc:
        raise ValueError(usage) from exc
    return leading, COMPARATOR_SYMBOLS[symbol], threshold


def _resolve_circuit_id(
    on_clause: str | None,
    default_circuit_id: str,
    timelines_by_id: dict[str, Timeline],
    *,
    allow_any: bool,
) -> str | None:
    if on_clause is None:
        return default_circuit_id
    if on_clause == "*":
        if not allow_any:
            raise ValueError("this condition needs a specific circuit, use 'on NAME'")
        return None
    resolved = _resolve_circuit_by_label(on_clause, timelines_by_id)
    return resolved.circuit_id if resolved is not None else on_clause


def _parse_gate(
    args: list[str], circuit_id: str, timelines_by_id: dict[str, Timeline]
) -> GateMatchCondition:
    args, on_clause = _split_on_clause(args)
    if not args:
        raise ValueError(
            f"usage: gate NAME [Q1 Q2 ...] [on CIRCUIT|*]  ({CONDITION_USAGE})"
        )
    gate, qubit_args = args[0], args[1:]
    qubits = tuple(int(q) for q in qubit_args) if qubit_args else None
    return GateMatchCondition(
        circuit_id=_resolve_circuit_id(on_clause, circuit_id, timelines_by_id, allow_any=True),
        gate=gate,
        qubits=qubits,
    )


def _parse_seq(
    args: list[str], circuit_id: str, timelines_by_id: dict[str, Timeline]
) -> SeqThresholdCondition:
    usage = f"usage: seq CMP N [on CIRCUIT|*]  ({CONDITION_USAGE})"
    args, on_clause = _split_on_clause(args)
    leading, comparator, threshold = _parse_trailing_comparator(args, usage)
    if leading:
        raise ValueError(usage)
    return SeqThresholdCondition(
        circuit_id=_resolve_circuit_id(on_clause, circuit_id, timelines_by_id, allow_any=True),
        seq=int(threshold),
        comparator=comparator,
    )


def _parse_watch(
    args: list[str], circuit_id: str, timelines_by_id: dict[str, Timeline]
) -> WatchThresholdCondition:
    usage = (
        "usage: watch qubit|clbit|register REF CMP THRESHOLD [on CIRCUIT]"
        f"  ({CONDITION_USAGE})"
    )
    args, on_clause = _split_on_clause(args)
    leading, comparator, threshold = _parse_trailing_comparator(args, usage)
    if len(leading) != 2 or leading[0] not in WATCH_KINDS:
        raise ValueError(usage)
    watch_kind_arg, ref = leading
    watch_kind = cast(Literal["qubit", "clbit", "register"], watch_kind_arg)
    return WatchThresholdCondition(
        circuit_id=cast(
            str,
            _resolve_circuit_id(on_clause, circuit_id, timelines_by_id, allow_any=False),
        ),
        watch_kind=watch_kind,
        ref=ref,
        comparator=comparator,
        threshold=threshold,
    )


def _parse_expect(
    args: list[str], circuit_id: str, timelines_by_id: dict[str, Timeline]
) -> ExpectationCondition:
    usage = f"usage: expect X|Y|Z QUBIT CMP THRESHOLD [on CIRCUIT]  ({CONDITION_USAGE})"
    args, on_clause = _split_on_clause(args)
    leading, comparator, threshold = _parse_trailing_comparator(args, usage)
    if len(leading) != 2 or leading[0] not in PAULIS:
        raise ValueError(usage)
    pauli_arg, qubit_arg = leading
    try:
        qubit = int(qubit_arg)
    except ValueError as exc:
        raise ValueError(usage) from exc
    return ExpectationCondition(
        circuit_id=cast(
            str,
            _resolve_circuit_id(on_clause, circuit_id, timelines_by_id, allow_any=False),
        ),
        pauli=cast(Literal["X", "Y", "Z"], pauli_arg),
        qubit=qubit,
        comparator=comparator,
        threshold=threshold,
    )


def _parse_prob(
    args: list[str], circuit_id: str, timelines_by_id: dict[str, Timeline]
) -> ProbabilityCondition:
    usage = f"usage: prob OUTCOME CMP THRESHOLD [on CIRCUIT]  ({CONDITION_USAGE})"
    args, on_clause = _split_on_clause(args)
    leading, comparator, threshold = _parse_trailing_comparator(args, usage)
    if len(leading) != 1:
        raise ValueError(usage)
    return ProbabilityCondition(
        circuit_id=cast(
            str,
            _resolve_circuit_id(on_clause, circuit_id, timelines_by_id, allow_any=False),
        ),
        outcome=leading[0],
        comparator=comparator,
        threshold=threshold,
    )


def _parse_tvd(
    args: list[str], circuit_id: str, timelines_by_id: dict[str, Timeline]
) -> DistributionDistanceCondition:
    usage = (
        "usage: tvd OUTCOME=PROB [OUTCOME=PROB ...] CMP THRESHOLD [on CIRCUIT]"
        f"  ({CONDITION_USAGE})"
    )
    args, on_clause = _split_on_clause(args)
    leading, comparator, threshold = _parse_trailing_comparator(args, usage)
    if not leading or not all(_OUTCOME_PAIR_RE.match(tok) for tok in leading):
        raise ValueError(usage)
    expected: dict[str, float] = {}
    for token in leading:
        outcome, prob_arg = token.split("=", 1)
        try:
            expected[outcome] = float(prob_arg)
        except ValueError as exc:
            raise ValueError(usage) from exc
    return DistributionDistanceCondition(
        circuit_id=cast(
            str,
            _resolve_circuit_id(on_clause, circuit_id, timelines_by_id, allow_any=False),
        ),
        expected=expected,
        comparator=comparator,
        threshold=threshold,
    )


def _parse_anomaly(
    args: list[str], circuit_id: str, timelines_by_id: dict[str, Timeline]
) -> AnomalyCondition:
    usage = f"usage: anomaly [info|warning|error] [on CIRCUIT|*]  ({CONDITION_USAGE})"
    args, on_clause = _split_on_clause(args)
    if len(args) > 1 or (args and args[0] not in SEVERITIES):
        raise ValueError(usage)
    min_severity = cast(Severity, args[0]) if args else "warning"
    return AnomalyCondition(
        circuit_id=_resolve_circuit_id(on_clause, circuit_id, timelines_by_id, allow_any=True),
        min_severity=min_severity,
    )


_PARSERS = {
    "gate": _parse_gate,
    "seq": _parse_seq,
    "watch": _parse_watch,
    "expect": _parse_expect,
    "prob": _parse_prob,
    "tvd": _parse_tvd,
    "anomaly": _parse_anomaly,
}


def parse_condition_args(
    kind: str,
    args: list[str],
    circuit_id: str,
    timelines_by_id: dict[str, Timeline] | None = None,
) -> BreakpointCondition:
    """Parse one condition's kind and remaining tokens into a
    :class:`~qisdebug._breakpoints.BreakpointCondition`, defaulting to
    ``circuit_id`` unless the tokens end with an ``on CIRCUIT``/``on *``
    clause.

    A small fixed per-kind tokenizer, not an expression parser -- breakpoint
    conditions are a closed set (see :mod:`qisdebug._breakpoints`), never
    arbitrary code. Shared by both frontends (CLI's ``break`` command line,
    GUI's condition-builder row) so the two accept exactly the same syntax
    and never drift apart on what's valid. Raises ``ValueError`` (including
    on a wrapped :class:`pydantic.ValidationError`) with a message safe to
    show the user directly.

    ``timelines_by_id`` resolves an ``on CIRCUIT`` clause's typed label
    (a circuit's name, per :func:`circuit_choices`) back to the real,
    otherwise-hidden ``circuit_id`` the stored condition needs; omitted (or
    a label that matches nothing tracked), the clause is used verbatim as
    the id, matching this function's original id-only behavior.
    """
    parser = _PARSERS.get(kind)
    if parser is None:
        raise ValueError(f"unknown breakpoint kind {kind!r}  ({CONDITION_USAGE})")
    try:
        return parser(args, circuit_id, timelines_by_id or {})
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc


# -- Autocomplete/suggestion -------------------------------------------------


def _suggest_gate(tokens: list[str], timeline: Timeline | None) -> list[str]:
    if not tokens:
        return known_gate_names(timeline) if timeline else []
    return [*(qubit_range(timeline) if timeline else []), "on"]


def _suggest_seq(tokens: list[str], comparators: list[str]) -> list[str]:
    if not tokens:
        return comparators
    if len(tokens) == 1:
        return []
    return ["on"]


def _suggest_watch(
    tokens: list[str], timeline: Timeline | None, comparators: list[str]
) -> list[str]:
    position = len(tokens)
    if position == 0:
        return list(WATCH_KINDS)
    if position == 1 and timeline:
        refs_by_kind = {
            "qubit": qubit_range,
            "clbit": clbit_range,
            "register": register_names,
        }
        return refs_by_kind.get(tokens[0], lambda _t: [])(timeline)
    if position == 2:
        return comparators
    if position == 3:
        return []
    return ["on"]


def _suggest_expect(
    tokens: list[str], timeline: Timeline | None, comparators: list[str]
) -> list[str]:
    position = len(tokens)
    if position == 0:
        return list(PAULIS)
    if position == 1:
        return qubit_range(timeline) if timeline else []
    if position == 2:
        return comparators
    if position == 3:
        return []
    return ["on"]


def _suggest_prob(
    tokens: list[str], timeline: Timeline | None, comparators: list[str]
) -> list[str]:
    position = len(tokens)
    if position == 0:
        return small_outcome_space(timeline)
    if position == 1:
        return comparators
    if position == 2:
        return []
    return ["on"]


def _suggest_anomaly(tokens: list[str]) -> list[str]:
    if not tokens:
        return [*SEVERITIES, "on"]
    return ["on"]


# Every per-kind suggester normalized to `(tokens, timeline, comparators) ->
# list[str]`, even the ones that ignore an argument, so `suggest_tokens` can
# dispatch through one dict instead of branching on every kind itself.
_SUGGESTERS: dict[str, Callable[..., list[str]]] = {
    "gate": lambda tokens, timeline, _c: _suggest_gate(tokens, timeline),
    "seq": lambda tokens, _t, comparators: _suggest_seq(tokens, comparators),
    "watch": _suggest_watch,
    "expect": _suggest_expect,
    "prob": _suggest_prob,
    "tvd": lambda tokens, _t, comparators: _suggest_tvd_tail(tokens, comparators),
    "anomaly": lambda tokens, _t, _c: _suggest_anomaly(tokens),
}


def suggest_tokens(
    kind: str,
    tokens: list[str],
    timeline: Timeline | None,
    circuits: list[str],
) -> list[str]:
    """Return valid completions for the next token, given ``kind`` and the
    tokens already typed after it (excluding the kind itself).

    Pure and read-only -- no parsing side effects -- so it's safe to call on
    every keystroke. ``timeline`` is the circuit currently in scope (for
    gate names/qubit-clbit/register ranges); pass ``None`` if unknown, which
    just means fewer suggestions rather than an error. Built from the same
    small enumeration helpers (:func:`known_gate_names` etc.) the GUI uses
    directly for its own dropdowns, so CLI completion and GUI choices never
    diverge on what's valid. Dispatches to one small per-kind helper so no
    single function has to branch on every kind and every position at once.

    ``on CIRCUIT``/``on *`` is a clause every kind accepts at most once, so
    once ``"on"`` already appears anywhere in ``tokens`` (not just as the
    very last one, which instead means "still typing the circuit") nothing
    further is offered -- a condition already has its scope decided by
    then, and a per-kind suggester further down would otherwise keep
    re-offering ``"on"`` forever with no way to tell "done" from "still
    going" (see :meth:`~._cli_session.CliSession.break_is_complete`, which
    is what actually closes the guided wizard once this returns empty).
    """
    if tokens and tokens[-1] == "on":
        return circuits
    if "on" in tokens:
        return []
    suggester = _SUGGESTERS.get(kind)
    if suggester is None:
        return []
    return suggester(tokens, timeline, list(COMPARATOR_SYMBOLS))


def _hint_gate(rest: list[str]) -> str:
    if not rest:
        return "Which gate name should trigger it?"
    return (
        "Optionally add the exact qubit indices it must touch, add 'on' to "
        "scope to one circuit, or finish now to match any qubits/circuit."
    )


def _hint_seq(rest: list[str]) -> str:
    if not rest:
        return "Choose a comparator for the sequence number."
    if len(rest) == 1:
        return "Type the sequence-number threshold."
    return "Optionally scope to one circuit with 'on', or finish now."


def _hint_watch(rest: list[str]) -> str:
    position = len(rest)
    if position == 0:
        return "Watch a qubit, a clbit, or a register?"
    if position == 1:
        return "Which one? (index or name)"
    if position == 2:
        return "Choose a comparator."
    if position == 3:
        return "Type the threshold value (0-1 probability)."
    return "Optionally scope to one circuit with 'on', or finish now."


def _hint_expect(rest: list[str]) -> str:
    position = len(rest)
    if position == 0:
        return "Which Pauli operator (X, Y, or Z)?"
    if position == 1:
        return "Which qubit index?"
    if position == 2:
        return "Choose a comparator."
    if position == 3:
        return "Type the expectation-value threshold."
    return "Optionally scope to one circuit with 'on', or finish now."


def _hint_prob(rest: list[str]) -> str:
    position = len(rest)
    if position == 0:
        return "Which exact outcome bitstring?"
    if position == 1:
        return "Choose a comparator."
    if position == 2:
        return "Type the probability threshold (0-1)."
    return "Optionally scope to one circuit with 'on', or finish now."


def _hint_tvd(rest: list[str]) -> str:
    if not rest:
        return "Type an OUTCOME=PROB pair, e.g. 00=0.5 (repeatable)."
    if rest[-1] in COMPARATOR_SYMBOLS:
        return "Type the total-variation-distance threshold."
    if len(rest) >= 2 and rest[-2] in COMPARATOR_SYMBOLS:
        return "Optionally scope to one circuit with 'on', or finish now."
    if _OUTCOME_PAIR_RE.match(rest[-1]):
        return "Add another OUTCOME=PROB pair, or choose a comparator to set the threshold."
    return "Type an OUTCOME=PROB pair, e.g. 00=0.5."


def _hint_anomaly(rest: list[str]) -> str:
    if not rest:
        return "Minimum severity to break on, or finish now for the default (warning)."
    return "Optionally scope to one circuit with 'on', or finish now."


_STEP_HINTS: dict[str, Callable[[list[str]], str]] = {
    "gate": _hint_gate,
    "seq": _hint_seq,
    "watch": _hint_watch,
    "expect": _hint_expect,
    "prob": _hint_prob,
    "tvd": _hint_tvd,
    "anomaly": _hint_anomaly,
}


def condition_step_hint(tokens: list[str]) -> str:
    """Return a one-line, plain-language explanation of what the next
    breakpoint token means, given everything typed so far (kind included).

    Mirrors :func:`suggest_tokens`'s ``(kind, position)`` dispatch exactly,
    but describes the step in words instead of enumerating candidates --
    the two are shown side by side by a guided input screen so the visible
    option list is never the only explanation of what it's for.
    """
    if not tokens or tokens[0] not in KINDS:
        return "What should trigger this breakpoint?"
    if tokens[-1] == "on":
        return "Which circuit (or * for any)?"
    if "on" in tokens[1:]:
        return "Ready -- finish to register this breakpoint."
    return _STEP_HINTS[tokens[0]](tokens[1:])


def small_outcome_space(timeline: Timeline | None) -> list[str]:
    """Enumerate every bitstring outcome, when the qubit count is small
    enough that doing so is actually useful as a suggestion list."""
    if timeline is None or not 0 < timeline.num_qubits <= 4:
        return []
    return [
        format(i, f"0{timeline.num_qubits}b") for i in range(2**timeline.num_qubits)
    ]


def _suggest_tvd_tail(tokens: list[str], comparators: list[str]) -> list[str]:
    """Suggestions for ``tvd``'s variable-length ``OUTCOME=PROB ...`` prefix:
    once the last token typed is a comparator symbol, a threshold is
    expected next (not enumerable); once a threshold follows a comparator,
    only ``on`` is left; while still reading ``OUTCOME=PROB`` pairs, offer
    the comparators as the way to end that list (further pairs are valid
    too, but free-text and not worth enumerating).
    """
    if not tokens:
        return []
    if tokens[-1] in comparators:
        return []
    if len(tokens) >= 2 and tokens[-2] in comparators:
        return ["on"]
    if _OUTCOME_PAIR_RE.match(tokens[-1]):
        return comparators
    return []


# -- Navigation over an already-captured, merged timeline -------------------


def _replay_prev_depth(merged: Merged, upto_index: int) -> dict[str, int | None]:
    """Seed per-circuit transpile depth from the start of ``merged`` through
    ``upto_index`` inclusive, matching how :class:`~qisdebug._store.TelemetryStore`
    tracks it incrementally live -- needed so an :class:`~qisdebug._breakpoints.AnomalyCondition`
    checking transpile depth sees the same previous-depth context regardless
    of where in the timeline a search starts.
    """
    prev_depth: dict[str, int | None] = {}
    for i in range(min(upto_index + 1, len(merged))):
        timeline, event = merged[i]
        depth = getattr(event, "depth", None)
        if depth is not None:
            prev_depth[timeline.circuit_id] = depth
    return prev_depth


def find_next_hit(
    merged: Merged,
    conditions: dict[str, BreakpointCondition],
    start_index: int,
) -> tuple[int, str] | None:
    """Return ``(index, handle)`` of the first event after ``start_index``
    that satisfies any registered condition, or ``None`` if none do.
    """
    prev_depth = _replay_prev_depth(merged, start_index)
    for i in range(start_index + 1, len(merged)):
        timeline, event = merged[i]
        for handle, condition in conditions.items():
            if matches(
                condition,
                event,
                timeline,
                prev_depth=prev_depth.get(timeline.circuit_id),
            ):
                return i, handle
        depth = getattr(event, "depth", None)
        if depth is not None:
            prev_depth[timeline.circuit_id] = depth
    return None


def find_prev_hit(
    merged: Merged,
    conditions: dict[str, BreakpointCondition],
    start_index: int,
) -> tuple[int, str] | None:
    """Return ``(index, handle)`` of the last event before ``start_index``
    that satisfies any registered condition, or ``None`` if none do.

    Depth context for an anomaly condition is replayed from the very start
    of ``merged`` rather than assumed, since walking backwards can't reuse a
    forward-accumulated ``prev_depth`` map the way :func:`find_next_hit` can.
    """
    if start_index <= 0:
        return None
    depth_before: dict[int, int | None] = {}
    running: dict[str, int | None] = {}
    for i in range(start_index):
        timeline, event = merged[i]
        depth_before[i] = running.get(timeline.circuit_id)
        depth = getattr(event, "depth", None)
        if depth is not None:
            running[timeline.circuit_id] = depth
    for i in range(start_index - 1, -1, -1):
        timeline, event = merged[i]
        for handle, condition in conditions.items():
            if matches(condition, event, timeline, prev_depth=depth_before.get(i)):
                return i, handle
    return None


def find_first_hit(
    merged: Merged, conditions: dict[str, BreakpointCondition]
) -> tuple[int, str] | None:
    """Return ``(index, handle)`` of the earliest event in ``merged`` that
    satisfies any registered condition, or ``None`` if none do."""
    return find_next_hit(merged, conditions, -1)


def find_last_hit(
    merged: Merged, conditions: dict[str, BreakpointCondition]
) -> tuple[int, str] | None:
    """Return ``(index, handle)`` of the latest event in ``merged`` that
    satisfies any registered condition, or ``None`` if none do."""
    return find_prev_hit(merged, conditions, len(merged))


def _circuit_label(circuit_id: str, timelines_by_id: dict[str, Timeline]) -> str:
    """Human-facing label for a condition's stored ``circuit_id``: the
    circuit's name if it's still tracked and named, else the id itself as a
    last resort (e.g. a circuit evicted since the breakpoint was set) --
    never omitted outright, but never the raw id when a name is available.
    """
    timeline = timelines_by_id.get(circuit_id)
    return timeline.name if timeline is not None and timeline.name else circuit_id


def _optional_scope(circuit_id: str | None, timelines_by_id: dict[str, Timeline]) -> str:
    """`` on <name>`` for a scoped condition, or ``""`` for one that isn't."""
    if not circuit_id:
        return ""
    return f" on {_circuit_label(circuit_id, timelines_by_id)}"


def _describe_gate(condition: GateMatchCondition, timelines_by_id: dict[str, Timeline]) -> str:
    qubits = f" q{list(condition.qubits)}" if condition.qubits is not None else ""
    scope = _optional_scope(condition.circuit_id, timelines_by_id)
    return f"gate {condition.gate}{qubits}{scope}"


def _describe_seq(condition: SeqThresholdCondition, timelines_by_id: dict[str, Timeline]) -> str:
    scope = _optional_scope(condition.circuit_id, timelines_by_id)
    return f"seq {_SYMBOL_FOR_COMPARATOR[condition.comparator]} {condition.seq}{scope}"


def _describe_watch(
    condition: WatchThresholdCondition, timelines_by_id: dict[str, Timeline]
) -> str:
    label = _circuit_label(condition.circuit_id, timelines_by_id)
    return (
        f"{label} {condition.watch_kind} {condition.ref} "
        f"{_SYMBOL_FOR_COMPARATOR[condition.comparator]} {condition.threshold}"
    )


def _describe_expectation(
    condition: ExpectationCondition, timelines_by_id: dict[str, Timeline]
) -> str:
    return (
        f"{_circuit_label(condition.circuit_id, timelines_by_id)} "
        f"expect {condition.pauli}{condition.qubit} "
        f"{_SYMBOL_FOR_COMPARATOR[condition.comparator]} {condition.threshold}"
    )


def _describe_probability(
    condition: ProbabilityCondition, timelines_by_id: dict[str, Timeline]
) -> str:
    return (
        f"{_circuit_label(condition.circuit_id, timelines_by_id)} prob {condition.outcome} "
        f"{_SYMBOL_FOR_COMPARATOR[condition.comparator]} {condition.threshold}"
    )


def _describe_distribution(
    condition: DistributionDistanceCondition, timelines_by_id: dict[str, Timeline]
) -> str:
    expected = " ".join(f"{k}={v}" for k, v in condition.expected.items())
    return (
        f"{_circuit_label(condition.circuit_id, timelines_by_id)} tvd({expected}) "
        f"{_SYMBOL_FOR_COMPARATOR[condition.comparator]} {condition.threshold}"
    )


def _describe_anomaly(condition: AnomalyCondition, timelines_by_id: dict[str, Timeline]) -> str:
    scope = _optional_scope(condition.circuit_id, timelines_by_id)
    return f"anomaly >= {condition.min_severity}{scope}"


_DESCRIBERS: dict[str, Callable[..., str]] = {
    "gate": _describe_gate,
    "seq": _describe_seq,
    "watch": _describe_watch,
    "expectation": _describe_expectation,
    "probability": _describe_probability,
    "distribution": _describe_distribution,
    "anomaly": _describe_anomaly,
}


def describe_condition(
    condition: BreakpointCondition, timelines_by_id: dict[str, Timeline] | None = None
) -> str:
    """Return a compact one-line label for a breakpoint condition, for
    listing widgets in both frontends. ``timelines_by_id`` renders any
    scoped circuit by name rather than its hidden id; omitted, an id that
    matches nothing tracked is shown as-is."""
    return _DESCRIBERS[condition.kind](condition, timelines_by_id or {})
