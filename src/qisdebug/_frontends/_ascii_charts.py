"""Text/Unicode chart rendering for the CLI/TUI frontends.

The terminal-only counterparts to :mod:`._render`'s image-based histogram
and Bloch-sphere renderers (:func:`~._render.render_histogram`,
:func:`~._render.render_bloch`), which return a PNG a GUI can embed but a
plain terminal has no way to display. Split into its own module purely to
keep :mod:`._render` under this project's line-count threshold -- this is
still frontend-agnostic, shared computation, just addressed directly by
:mod:`._cli_session` rather than through :mod:`._render`'s namespace.
"""

from __future__ import annotations

import numpy as np

from .._events import StateSnapshotEvent
from ._render import clean_float, reduced_bloch_vector

_FILLED_CHAR = "▓"
_EMPTY_CHAR = "░"


def _shaded_bar(fraction: float, width: int) -> str:
    """Render ``fraction`` (0..1) as a ``width``-character bar: a dense
    stipple (``▓``) for the filled portion and a light stipple (``░``) for
    the empty remainder -- a patterned texture rather than a solid color
    block, and the empty portion stays visibly drawn (not blank space) so
    the bar's full range reads at a glance even where a count is small.
    """
    if width <= 0:
        return ""
    fraction = max(0.0, min(1.0, fraction))
    filled = min(width, round(fraction * width))
    return _FILLED_CHAR * filled + _EMPTY_CHAR * (width - filled)


def render_histogram_ascii(counts: dict[str, int], max_width: int = 48) -> str:
    """Render execution counts as a shaded bar chart, scaled so every line
    (outcome label, bar, and count/percentage together) fits within
    ``max_width`` columns -- a caller passes the actual available render
    width (a panel's width, or the real terminal's) so the chart never
    overflows it, rather than assuming a fixed bar width always fits.
    One row per line, no blank lines between them -- a presentation layer
    that wants visual separation between outcomes without spending extra
    height on it can style rows instead (see the TUI's
    ``_underline_row_dividers``).
    """
    if not counts:
        return "(no counts)"
    total = sum(counts.values())
    outcomes = sorted(counts)
    label_width = max(len(outcome) for outcome in outcomes)
    lines = []
    for outcome in outcomes:
        count = counts[outcome]
        suffix = f" {count} ({count / total:.1%})"
        bar_width = max(1, max_width - label_width - len(suffix) - 1)
        # Scaled against the true 100% (the full shot count), not against
        # whichever outcome happens to be tallest -- a 14% outcome must
        # fill ~14% of the bar even when it's the largest one recorded, not
        # be stretched to fill the whole width.
        fraction = count / total if total else 0.0
        lines.append(f"{outcome:>{label_width}} {_shaded_bar(fraction, bar_width)}{suffix}")
    return "\n".join(lines)


_BLOCH_RADIUS = 5


def _bloch_grid(x: float, z: float, radius: int) -> list[str]:
    """Build the sphere-outline-plus-axes grid, with a marker placed at the
    projected ``(x, z)`` position -- split out of :func:`render_bloch_ascii`
    purely to keep that function's own local-variable count down.
    """
    size = 2 * radius + 1
    grid = [[" "] * size for _ in range(size)]
    for row in range(size):
        for col in range(size):
            gx = (col - radius) / radius
            gz = (radius - row) / radius
            if 0.9 <= (gx * gx + gz * gz) ** 0.5 <= 1.08:
                grid[row][col] = "."
    for col in range(size):
        if grid[radius][col] == " ":
            grid[radius][col] = "-"
    for row in range(size):
        if grid[row][radius] == " ":
            grid[row][radius] = "|"
    grid[radius][radius] = "+"
    marker_col = max(0, min(size - 1, round((x + 1) * radius)))
    marker_row = max(0, min(size - 1, round((1 - z) * radius)))
    grid[marker_row][marker_col] = "@"
    return ["".join(row) for row in grid]


def render_bloch_ascii(event: StateSnapshotEvent, qubit: int = 0) -> str | None:
    """Render a single qubit's Bloch vector as a small text sphere outline
    (front view: +X right, +Z up) with a marker at its projected position --
    the CLI's text-only counterpart to :func:`~._render.render_bloch`'s
    matplotlib image, since a terminal has no way to show that image form.
    Only meaningful for a snapshot of exactly one qubit (a reduced
    multi-qubit Bloch vector still exists mathematically, but picking which
    qubit to draw for an N-qubit state is a choice this renderer doesn't
    make); returns ``None`` for anything else, or if there's no statevector
    yet.
    """
    amplitudes = event.amplitudes()
    if not amplitudes:
        return None
    num_qubits = event.num_qubits or int(np.log2(len(amplitudes)))
    if num_qubits != 1:
        return None
    x, y, z = (clean_float(v) for v in reduced_bloch_vector(amplitudes, qubit, num_qubits))

    size = 2 * _BLOCH_RADIUS + 1
    header = (
        f"qubit {qubit} bloch: x={x:+.3f}  y={y:+.3f}  z={z:+.3f}  "
        "(front view: +X right, +Z up, Y not shown)"
    )
    return "\n".join(
        [header, "|0⟩".center(size), *_bloch_grid(x, z, _BLOCH_RADIUS), "|1⟩".center(size)]
    )
