"""Visualizer: decoupled from instrumentation."""

from _helpers import traced_bell

from qisdebug import visualizer


def test_format_statevector_renders_amplitudes(debug):
    """Snapshot formatting shows basis states with probabilities."""
    timeline = traced_bell(debug)
    text = visualizer.format_statevector(timeline.snapshots[-1])
    assert "|00>" in text
    assert "p=0.5000" in text
