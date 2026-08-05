"""Construction hook: every append path is recorded without changing behavior."""

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from qiskit.circuit import CircuitInstruction
from qiskit.circuit.library import XGate
from qiskit.circuit.quantumcircuit import QuantumCircuit

import qisdebug


def _recorded(debug, qc):
    """Return (gate, qubits) tuples recorded for a circuit."""
    return [(e.gate, e.qubits) for e in debug.get(qc).gate_events]


def test_gate_helper_methods_are_recorded(debug):
    """Standard-gate helpers (the `_append_standard_gate` fast path) record."""
    debug.start()
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.swap(1, 2)
    assert _recorded(debug, qc) == [("h", (0,)), ("cx", (0, 1)), ("swap", (1, 2))]


def test_explicit_append_and_circuit_instruction_are_recorded(debug):
    """Both public append forms record exactly one event per instruction."""
    debug.start()
    qc = QuantumCircuit(2)
    qc.append(XGate(), [0])
    qc.append(CircuitInstruction(XGate(), [qc.qubits[1]], []))
    assert _recorded(debug, qc) == [("x", (0,)), ("x", (1,))]


def test_measure_broadcast_records_each_instruction(debug):
    """A single broadcasting call records one event per resolved instruction."""
    debug.start()
    qc = QuantumCircuit(2, 2)
    qc.measure([0, 1], [0, 1])
    events = debug.get(qc).gate_events
    assert [(e.gate, e.qubits, e.clbits) for e in events] == [
        ("measure", (0,), (0,)),
        ("measure", (1,), (1,)),
    ]


def test_barrier_and_reset_are_recorded(debug):
    """Non-gate instructions that flow through `_append` are captured too."""
    debug.start()
    qc = QuantumCircuit(2)
    qc.barrier()
    qc.reset(0)
    assert [e.gate for e in debug.get(qc).gate_events] == ["barrier", "reset"]


def test_parameters_are_recorded(debug):
    """Gate parameters are stored as floats on the event."""
    debug.start()
    qc = QuantumCircuit(1)
    qc.rz(0.25, 0)
    event = debug.get(qc).gate_events[0]
    assert event.gate == "rz"
    assert event.params == (0.25,)


def _sub_circuit():
    """Two-gate subcircuit used by the compose tests."""
    sub = QuantumCircuit(2, name="sub")
    sub.h(0)
    sub.cx(0, 1)
    return sub


def test_compose_inplace_records_merged_instructions(debug):
    """compose(inplace=True) bypasses append but is still recorded, remapped."""
    debug.start()
    qc = QuantumCircuit(3)
    qc.x(0)
    qc.compose(_sub_circuit(), qubits=[1, 2], inplace=True)
    assert _recorded(debug, qc) == [("x", (0,)), ("h", (1,)), ("cx", (1, 2))]


def test_compose_copy_records_nothing_on_the_source(debug):
    """compose(inplace=False) leaves the source circuit's timeline untouched."""
    debug.start()
    qc = QuantumCircuit(2)
    qc.x(0)
    qc.compose(_sub_circuit())
    assert _recorded(debug, qc) == [("x", (0,))]


def test_compose_front_records_no_misattributed_events(debug):
    """front=True prepends, which a data diff cannot attribute, so skip it."""
    debug.start()
    qc = QuantumCircuit(2)
    qc.x(0)
    qc.compose(_sub_circuit(), front=True, inplace=True)
    assert _recorded(debug, qc) == [("x", (0,))]
    assert len(qc.data) == 3


def test_control_flow_body_params_are_compact(debug):
    """Control-flow block bodies render as op summaries, not circuit drawings."""
    debug.start()
    qc = QuantumCircuit(2, 1)
    qc.measure(0, 0)
    with qc.if_test((qc.clbits[0], 1)):
        qc.x(1)
        qc.x(1)
    event = next(e for e in debug.get(qc).gate_events if e.gate == "if_else")
    assert event.params[0] == "[x*2]"
    assert all("\n" not in str(p) for p in event.params)
    assert "\n" not in event.describe()


def test_recording_does_not_change_circuit_semantics(debug):
    """A circuit built while active equals one built while inactive."""
    debug.start()
    active = QuantumCircuit(2, 2)
    active.h(0)
    active.cx(0, 1)
    active.measure([0, 1], [0, 1])
    debug.end()
    plain = QuantumCircuit(2, 2)
    plain.h(0)
    plain.cx(0, 1)
    plain.measure([0, 1], [0, 1])
    assert active == plain


_GATE_POOL = [
    ("h", 1, 0),
    ("x", 1, 0),
    ("s", 1, 0),
    ("rz", 1, 1),
    ("rx", 1, 1),
    ("cx", 2, 0),
    ("cz", 2, 0),
    ("swap", 2, 0),
]


@st.composite
def _gate_sequences(draw):
    """Random gate applications on a 3-qubit circuit."""
    count = draw(st.integers(min_value=0, max_value=30))
    sequence = []
    for _ in range(count):
        name, arity, param_count = draw(st.sampled_from(_GATE_POOL))
        qubit_values = st.integers(min_value=0, max_value=2)
        qubits = draw(st.lists(qubit_values, min_size=arity, max_size=arity, unique=True))
        angles = st.floats(min_value=-3.14, max_value=3.14, allow_nan=False)
        params = [draw(angles)] * param_count
        sequence.append((name, qubits, params))
    return sequence


@settings(
    deadline=None, max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(sequence=_gate_sequences())
def test_every_random_gate_sequence_is_fully_recorded(sequence):
    """The recorded event list always mirrors circuit.data exactly."""
    debug = qisdebug.QisDebug()
    debug.start()
    try:
        qc = QuantumCircuit(3)
        for name, qubits, params in sequence:
            getattr(qc, name)(*params, *qubits)
        recorded = debug.get(qc).gate_events if sequence else []
        assert len(recorded) == len(qc.data)
        for event, instruction in zip(recorded, qc.data):
            assert event.gate == instruction.operation.name
            assert event.qubits == tuple(qc.find_bit(b).index for b in instruction.qubits)
    finally:
        debug.end()
