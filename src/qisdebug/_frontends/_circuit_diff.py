"""Gate-aligned split view of two *real* Qiskit circuit drawings.

Split out from :mod:`_render` (which stays generic drawing/event
presentation) because this diff needs Qiskit-specific machinery of its
own: inserting layout barriers into a circuit copy so its drawn text can
be split back into one column per instruction (see
:func:`_circuit_with_layer_barriers`), aligned against
:attr:`~qisdebug._analysis.TimelineDiff.gate_diff`. Diffing the two
drawings as raw text or characters can't tell a moved gate from a changed
one -- a single inserted gate shifts every character after it, so most of
the diagram would read as "changed" even when only one gate actually is.
Building the split view directly from the gate alignment instead gives
every aligned pair (or unmatched gate) its own column by construction, so
only the gates that actually differ are marked, on both sides, regardless
of how the rest of the circuit shifted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._analysis import GateDiffEntry, resolve_track_circuit
from .._events import Timeline
from ._render import StageMark, diff_at_marks, resolve_stage_element, track_label


@dataclass
class CircuitDiffSide:
    """One side of a gate-aligned circuit split view (see
    :class:`CircuitSplitDiff`): its label, its assembled drawing text, and
    where the content unique to this side is -- ``spans`` as
    character-offset ranges into ``text`` (for a renderer that addresses
    the text as one block, e.g. Qt's ``QTextCursor``), ``row_ranges`` as
    the same ranges expressed as (row-relative start, end) column pairs
    (for a renderer that works one printed line at a time, e.g. the CLI).
    """

    label: str
    text: str
    spans: list[tuple[int, int]]
    row_ranges: list[tuple[int, int]]


@dataclass
class CircuitSplitDiff:
    """A gate-aligned split view of two *real* Qiskit circuit drawings,
    ready to render side by side. ``a``'s ranges mark content that only
    side A has (removed, conventionally red); ``b``'s mark content that
    only side B has (added, conventionally green). Both frontends consume
    the same ranges, computed once here, so they highlight exactly the
    same regions (see :func:`render_circuit_split_diff`).
    """

    a: CircuitDiffSide
    b: CircuitDiffSide


def _circuit_with_layer_barriers(circuit: Any) -> Any:
    """Copy ``circuit`` with a full-width barrier before and after every
    instruction, so each instruction renders in its own drawing column.
    :func:`_split_barrier_columns` uses those barrier columns to recover
    exactly which characters of the *real* Qiskit drawing belong to which
    instruction, rather than diffing the raw drawn text (which can't tell
    a moved gate from a changed one -- inserting one gate shifts every
    character after it) or hand-rolling a synthetic diagram.
    """
    barriered = circuit.copy_empty_like()
    barriered.barrier()
    for instruction in circuit.data:
        barriered.append(instruction.operation, instruction.qubits, instruction.clbits)
        barriered.barrier()
    return barriered


def _split_barrier_columns(text: str) -> list[list[str]]:
    """Split a drawing produced from :func:`_circuit_with_layer_barriers`
    into column chunks (one string per line), dropping the barrier columns
    themselves -- ``░`` is the character Qiskit's text drawer reserves for
    barriers, so any column containing it anywhere is a separator, never
    real gate content. The first chunk is the row-label gutter (before the
    leading barrier); each chunk after that is exactly one instruction, in
    ``circuit.data`` order, plus an empty trailing chunk after the last
    barrier that callers should drop.
    """
    lines = text.split("\n")
    width = max((len(line) for line in lines), default=0)
    lines = [line.ljust(width) for line in lines]
    is_separator = [any(line[i] == "░" for line in lines) for i in range(width)]

    chunks: list[list[str]] = []
    col = 0
    while col < width:
        if is_separator[col]:
            col += 1
            continue
        start = col
        while col < width and not is_separator[col]:
            col += 1
        chunks.append([line[start:col] for line in lines])
    return chunks


def _circuit_diagram_chunks(circuit: Any) -> tuple[list[str], list[list[str]]]:
    """Draw ``circuit`` and split it into its row-label gutter plus one
    column chunk per instruction, in ``circuit.data`` order -- the same
    order :func:`~.._analysis.diff_timeline_points` aligns gates in, so
    entry-by-entry these line up with ``gate_diff``.
    """
    text = str(_circuit_with_layer_barriers(circuit).draw(output="text", fold=-1))
    chunks = _split_barrier_columns(text)
    prefix = chunks[0] if chunks else []
    gate_chunks = chunks[1 : 1 + len(circuit.data)]
    return prefix, gate_chunks


def _paired_gate_chunks(
    gate_diff: list[GateDiffEntry],
    gate_chunks_a: list[list[str]],
    gate_chunks_b: list[list[str]],
) -> list[tuple[list[str] | None, list[str] | None]]:
    """Pair each ``gate_diff`` entry with the column chunk(s) it consumes
    from each side, in order -- ``None`` on a side with no entry there.
    """
    iter_a = iter(gate_chunks_a)
    iter_b = iter(gate_chunks_b)
    return [
        (
            next(iter_a) if entry.a is not None else None,
            next(iter_b) if entry.b is not None else None,
        )
        for entry in gate_diff
    ]


def _entry_widths(pairs: list[tuple[list[str] | None, list[str] | None]]) -> list[int]:
    """The shared column width each aligned gate needs: wide enough for
    whichever side's chunk is wider (the narrower side, or a missing side,
    gets padded up to match -- see :func:`_fill_side_blocks`).
    """
    return [max(len(a[0]) if a else 0, len(b[0]) if b else 0) for a, b in pairs]


def _fill_side_blocks(
    gate_diff: list[GateDiffEntry],
    chunks: list[list[str] | None],
    widths: list[int],
    rows: int,
    changed_ops: tuple[str, ...],
) -> tuple[list[list[str]], list[tuple[int, int]]]:
    """Build one side's per-row gate content blocks (each padded to its
    entry's shared column width -- a missing chunk becomes a blank gap of
    that width) plus the row-relative ranges where ``changed_ops`` says
    this side differs from the other.
    """
    blocks: list[list[str]] = [[] for _ in range(rows)]
    row_ranges: list[tuple[int, int]] = []
    offset = 0
    for entry, chunk, width in zip(gate_diff, chunks, widths):
        for row in range(rows):
            blocks[row].append((chunk[row] if chunk else "").ljust(width))
        if entry.op in changed_ops:
            row_ranges.append((offset, offset + width))
        offset += width
    return blocks, row_ranges


def _align_gate_chunks(
    gate_diff: list[GateDiffEntry],
    gate_chunks_a: list[list[str]],
    gate_chunks_b: list[list[str]],
    rows_a: int,
    rows_b: int,
) -> tuple[list[list[str]], list[list[str]], list[tuple[int, int]], list[tuple[int, int]]]:
    """Walk a gate-aligned diff, pairing each entry with the column
    chunk(s) it consumes from each side (see :func:`_paired_gate_chunks`),
    and return each side's per-row gate content blocks plus the
    row-relative ranges -- measured from the start of the gate content, not
    counting a row-label prefix -- where the sides differ. An
    ``"insert"``/``"delete"`` leaves a blank gap of the same width on the
    other side instead of everything past it silently drifting out of sync.
    """
    pairs = _paired_gate_chunks(gate_diff, gate_chunks_a, gate_chunks_b)
    widths = _entry_widths(pairs)
    blocks_a, row_ranges_a = _fill_side_blocks(
        gate_diff, [a for a, _ in pairs], widths, rows_a, ("delete", "replace")
    )
    blocks_b, row_ranges_b = _fill_side_blocks(
        gate_diff, [b for _, b in pairs], widths, rows_b, ("insert", "replace")
    )
    return blocks_a, blocks_b, row_ranges_a, row_ranges_b


def _assemble_diff_side(
    label: str,
    prefix: list[str],
    blocks: list[list[str]],
    row_ranges: list[tuple[int, int]],
) -> CircuitDiffSide:
    """Join one side's row-label prefix with its gate content blocks into
    the final drawing text, then shift its gate-content-relative diff
    ranges by the prefix width and expand them into absolute
    character-offset spans across every row (see :class:`CircuitSplitDiff`).
    """
    rows = len(prefix)
    prefix_width = len(prefix[0]) if rows else 0
    lines = [prefix[row] + "".join(blocks[row]) for row in range(rows)]
    row_width = len(lines[0]) if rows else prefix_width
    shifted_ranges = [(prefix_width + start, prefix_width + end) for start, end in row_ranges]
    spans = [
        (row * (row_width + 1) + start, row * (row_width + 1) + end)
        for row in range(rows)
        for start, end in shifted_ranges
    ]
    return CircuitDiffSide(
        label=label, text="\n".join(lines), spans=spans, row_ranges=shifted_ranges
    )


def render_gate_diff_circuits(
    gate_diff: list[GateDiffEntry],
    circuit_a: Any,
    circuit_b: Any,
    label_a: str,
    label_b: str,
) -> CircuitSplitDiff:
    """Render the two circuits' *real* Qiskit text drawings side by side,
    with the gate columns that actually differ marked in red (removed,
    side A only) or green (added, side B only) -- built from ``gate_diff``
    (:attr:`~.._analysis.TimelineDiff.gate_diff`, aligned by gate name +
    qubits, the same alignment :func:`~._render.render_state_diff`'s text
    listing uses), not from diffing the drawn text itself.
    """
    prefix_a, gate_chunks_a = _circuit_diagram_chunks(circuit_a)
    prefix_b, gate_chunks_b = _circuit_diagram_chunks(circuit_b)
    blocks_a, blocks_b, row_ranges_a, row_ranges_b = _align_gate_chunks(
        gate_diff, gate_chunks_a, gate_chunks_b, len(prefix_a), len(prefix_b)
    )
    return CircuitSplitDiff(
        a=_assemble_diff_side(label_a, prefix_a, blocks_a, row_ranges_a),
        b=_assemble_diff_side(label_b, prefix_b, blocks_b, row_ranges_b),
    )


def _resolve_diff_side(
    stage: StageMark, timelines_by_id: dict[str, Timeline], element: str | None, side: str
) -> tuple[Timeline, str, Any] | str:
    """Resolve one side of a diff to ``(timeline, track, circuit)``, or an
    error message listing that stage's elements when the pick is ambiguous
    or the circuit has nothing resolvable yet.
    """
    resolved = resolve_stage_element(stage, timelines_by_id, element)
    if resolved is None:
        options = ", ".join(stage.elements_at(timelines_by_id))
        return (
            f"stage {side} has multiple elements ({options}) -- "
            "specify which one, e.g. diff <a> <b>"
        )
    timeline, track = resolved
    circuit = resolve_track_circuit(timeline, track, stage.seq)
    if circuit is None:
        return "(no circuit available yet for one or both sides)"
    return timeline, track, circuit


def render_circuit_split_diff(
    stage_a: StageMark,
    stage_b: StageMark,
    timelines_by_id: dict[str, Timeline],
    element_a: str | None = None,
    element_b: str | None = None,
) -> CircuitSplitDiff | str:
    """Render a gate-aligned split view of the two chosen elements' *real*
    circuit drawings -- circuit A's Qiskit diagram next to circuit B's,
    with the gate columns that actually differ marked in red/green (see
    :func:`render_gate_diff_circuits`). Companion to
    :func:`~._render.render_stage_diff`, which covers the same alignment
    as text and numbers; this covers what it looks like laid out on wires.
    Same element resolution and ambiguous-pick handling as that function,
    returning a message string instead of a :class:`CircuitSplitDiff` when
    a pick is ambiguous or unresolved.
    """
    side_a = _resolve_diff_side(stage_a, timelines_by_id, element_a, "A")
    if isinstance(side_a, str):
        return side_a
    side_b = _resolve_diff_side(stage_b, timelines_by_id, element_b, "B")
    if isinstance(side_b, str):
        return side_b
    timeline_a, track_a, circuit_a = side_a
    timeline_b, track_b, circuit_b = side_b
    diff = diff_at_marks(stage_a, (timeline_a, track_a), stage_b, (timeline_b, track_b))
    return render_gate_diff_circuits(
        diff.gate_diff,
        circuit_a,
        circuit_b,
        label_a=track_label(f"{timeline_a.name or timeline_a.circuit_id}:{track_a}"),
        label_b=track_label(f"{timeline_b.name or timeline_b.circuit_id}:{track_b}"),
    )
