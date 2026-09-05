"""QisDebug conditional breakpoints: pause a *live* run, not just replay one.

Run with::

    python examples/breakpoints.py

A breakpoint condition pauses whichever thread triggers it -- here that's a
worker thread building and running a circuit -- until the main thread (or,
in the MCP tools, an agent calling ``resume_run``) releases it. This example
hits two different condition kinds in one run: a ``gate`` condition (pause
the instant a specific gate is added, during construction) and a ``watch``
condition (pause once a qubit's measured probability crosses a threshold,
checked against each new state snapshot during execution).
"""

import threading

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

import qisdebug
from qisdebug import GateMatchCondition, WatchThresholdCondition


def build_and_run() -> None:
    """Ordinary Qiskit code, run on its own thread so a breakpoint hit here
    doesn't block the main thread's calls to wait_for_breakpoint()/resume().
    """
    qc = QuantumCircuit(3, 3, name="flagged")
    qc.x(0)  # qubit 0 -> |1>; the watch breakpoint below fires once this
    #          shows up in a state snapshot during execution
    qc.h(1)
    qc.cx(1, 2)  # <- the gate breakpoint below pauses right here
    qc.measure(range(3), range(3))
    AerSimulator().run(qc, shots=1000).result()


def main() -> None:
    """Hit a gate breakpoint during construction, then a watch breakpoint
    during execution, resuming the worker thread between the two."""
    debug = qisdebug.QisDebug()
    debug.start()

    gate_handle = debug.add_breakpoint(GateMatchCondition(gate="cx"))
    worker = threading.Thread(target=build_and_run)
    worker.start()

    hit = debug.wait_for_breakpoint(timeout=5)
    assert hit is not None
    print(f"paused: breakpoint {gate_handle} matched {hit.event_summary!r}")
    print("  (construction only -- execution hasn't started yet)")
    debug.remove_breakpoint(gate_handle)

    # Now watch qubit 0's measured P(1) going forward, so the run pauses
    # again the moment execution actually produces a high-probability state.
    circuit_id = debug.get("flagged").circuit_id
    watch_handle = debug.add_breakpoint(
        WatchThresholdCondition(
            circuit_id=circuit_id,
            watch_kind="qubit",
            ref="0",
            comparator="gt",
            threshold=0.9,
        )
    )
    debug.resume()

    hit = debug.wait_for_breakpoint(timeout=5)
    assert hit is not None
    print(f"paused: breakpoint {watch_handle} matched {hit.event_summary!r}")
    print("  (qubit 0's P(1) just crossed 0.9)")
    debug.remove_breakpoint(watch_handle)
    debug.resume()

    worker.join(timeout=5)
    timeline = debug.get("flagged")
    print(f"\nrun finished: {len(timeline.events)} events captured")

    debug.end()


if __name__ == "__main__":
    main()
