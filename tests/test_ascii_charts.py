"""Text/Unicode chart rendering: bar charts and the Bloch-sphere outline."""

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qisdebug._events import StateSnapshotEvent
from qisdebug._frontends._ascii_charts import render_bloch_ascii, render_histogram_ascii


def test_render_histogram_ascii_fits_requested_width():
    """Every line -- label, bar, and count/percentage together -- stays
    within max_width, however that budget is split between long and short
    outcome labels."""
    counts = {"00000000": 3, "00000001": 997}
    text = render_histogram_ascii(counts, max_width=36)
    for line in text.split("\n"):
        assert len(line) <= 36


def test_render_histogram_ascii_one_row_per_line_no_blank_lines():
    """Rows are tightly packed, one per line -- no blank-line padding
    between outcomes (that's a presentation-layer choice, not baked into
    the shared string; see the TUI's own row-divider styling)."""
    text = render_histogram_ascii({"00": 1, "01": 2, "10": 3, "11": 4}, max_width=50)
    rows = text.split("\n")
    assert len(rows) == 4
    assert all(row.strip() for row in rows)


def test_render_histogram_ascii_scales_against_total_not_the_tallest_bar():
    """Bars are scaled against the true 100% (the total shot count), not
    against whichever outcome happens to be tallest -- a 50/50 split
    renders two roughly half-filled bars, not two fully solid ones (which
    is what scaling against the tallest/peak bar would wrongly produce,
    since each side would then equal the peak)."""
    text = render_histogram_ascii({"00": 500, "11": 500}, max_width=60)
    lines = {line.split()[0]: line for line in text.split("\n")}
    for outcome in ("00", "11"):
        filled = lines[outcome].count("▓")
        bar_width = filled + lines[outcome].count("░")
        assert filled < bar_width  # not fully solid -- correctly ~50%, not 100%
        assert abs(filled - bar_width / 2) <= 1


def test_render_histogram_ascii_skewed_split_proportional_to_share():
    """An 86/14 split: the 86% outcome -- the tallest, but still well short
    of 100% -- must NOT render as a fully solid bar (that was the bug: the
    tallest bar was always stretched to fill the whole width regardless of
    its actual share), and the 14% outcome renders visibly short."""
    text = render_histogram_ascii({"heads": 860, "tails": 140}, max_width=60)
    lines = {line.split()[0]: line for line in text.split("\n")}
    heads_filled = lines["heads"].count("▓")
    bar_width = heads_filled + lines["heads"].count("░")
    tails_filled = lines["tails"].count("▓")

    assert heads_filled < bar_width  # 86% must leave a visible unfilled remainder
    assert abs(heads_filled - round(0.86 * bar_width)) <= 1
    assert abs(tails_filled - round(0.14 * bar_width)) <= 1
    assert tails_filled < heads_filled


def test_render_histogram_ascii_empty_counts():
    assert render_histogram_ascii({}) == "(no counts)"


def _snapshot(statevector, num_qubits):
    return StateSnapshotEvent(seq=1, step=0, statevector=list(statevector), num_qubits=num_qubits)


def test_render_bloch_ascii_plus_state_marker_on_the_right():
    """|+> = H|0> has bloch x=+1, so the marker sits on the sphere's right edge."""
    qc = QuantumCircuit(1)
    qc.h(0)
    statevector = Statevector(qc)
    text = render_bloch_ascii(_snapshot(statevector.data, 1), qubit=0)
    assert text is not None
    assert "x=+1.000" in text
    grid_lines = text.split("\n")[2:-1]
    marker_line = next(line for line in grid_lines if "@" in line)
    assert marker_line.index("@") == len(marker_line) - 1


def test_render_bloch_ascii_zero_state_marker_at_top():
    """|0> has bloch z=+1, the "north pole" -- the marker sits on the top row."""
    qc = QuantumCircuit(1)
    statevector = Statevector(qc)
    text = render_bloch_ascii(_snapshot(statevector.data, 1), qubit=0)
    assert text is not None
    grid_lines = text.split("\n")[2:-1]
    assert "@" in grid_lines[0]


def test_render_bloch_ascii_none_for_multi_qubit_snapshot():
    """Gated to single-qubit snapshots -- picking a qubit to draw for a
    multi-qubit state isn't a choice this renderer makes."""
    qc = QuantumCircuit(2)
    qc.h(0)
    statevector = Statevector(qc)
    assert render_bloch_ascii(_snapshot(statevector.data, 2), qubit=0) is None


def test_render_bloch_ascii_none_without_amplitudes():
    event = StateSnapshotEvent(seq=1, step=0, statevector=None, num_qubits=1)
    assert render_bloch_ascii(event) is None
