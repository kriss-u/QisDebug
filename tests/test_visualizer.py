"""Visualizer: decoupled from instrumentation, degrades without ipywidgets."""

import importlib.util

import pytest
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit_aer import AerSimulator

import qdebug
from conftest import get_timeline
from qdebug import visualizer
from qdebug._frontends.jupyter import JupyterFrontend

HAS_IPYWIDGETS = importlib.util.find_spec("ipywidgets") is not None


def _traced_bell():
    """Run a bell circuit under instrumentation and return its timeline."""
    qdebug.activate()
    qc = QuantumCircuit(2, 2, name="viz")
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    AerSimulator().run(qc, shots=10).result()
    return get_timeline(qc)


def test_format_statevector_renders_amplitudes():
    """Snapshot formatting shows basis states with probabilities."""
    timeline = _traced_bell()
    text = visualizer.format_statevector(timeline.snapshots[-1])
    assert "|00>" in text
    assert "p=0.5000" in text


@pytest.mark.skipif(HAS_IPYWIDGETS, reason="ipywidgets installed")
def test_show_jupyter_without_ipywidgets_raises_helpful_error():
    """Without ipywidgets, forcing the jupyter frontend explains how to install the extra."""
    timeline = _traced_bell()
    with pytest.raises(RuntimeError, match="qdebug\\[viz\\]"):
        timeline.show(frontend="jupyter")


@pytest.mark.skipif(not HAS_IPYWIDGETS, reason="ipywidgets not installed")
def test_show_jupyter_builds_widget():
    """With ipywidgets, the jupyter frontend returns a renderable widget tree."""
    timeline = _traced_bell()
    widget = timeline.show(frontend="jupyter")
    assert widget is not None
    assert widget.children


@pytest.mark.skipif(not HAS_IPYWIDGETS, reason="ipywidgets not installed")
def test_jupyter_frontend_is_available():
    """JupyterFrontend reports availability based on the ipywidgets import."""
    assert JupyterFrontend().is_available() is True
