"""Layer three: record per-pass circuit snapshots during transpilation.

``PassManager.run`` is the single choke point for both ``qiskit.transpile``
and user-constructed pass managers (StagedPassManager delegates to it via
``super()``). Per-pass recording goes through Qiskit's own supported
``callback`` parameter rather than patching pass internals, which makes this
the most version-stable layer. The wrapper also propagates the hidden circuit
id onto transpiled outputs, since Qiskit drops custom attributes there; that
is what links construction, transpilation and execution into one timeline.
"""

from __future__ import annotations

import functools
import threading
import time
from typing import Any, Callable

from .._compat import PatchRegistry, logger, resolve
from .._config import get_config
from .._events import (
    NoteEvent,
    Timeline,
    TranspileFinishedEvent,
    TranspilePassEvent,
    TranspileStartedEvent,
)
from .._store import ID_ATTRIBUTE, STORE

_local = threading.local()


def install(registry: PatchRegistry) -> bool:
    """Patch ``PassManager.run``; True when applied."""
    pass_manager_cls = resolve("qiskit.transpiler.passmanager", "PassManager")
    if pass_manager_cls is None:
        registry.warn("PassManager not found; transpile layer disabled")
        return False
    return registry.patch(pass_manager_cls, "run", _run_wrapper)


def _run_wrapper(original: Callable) -> Callable:
    """Wrap ``PassManager.run`` to record start, per-pass and finish events.

    The signature mirrors Qiskit's so the injected callback lands in the
    right positional slot whether callers pass it positionally or by keyword.
    Per-pass recording is only injected for single-circuit runs: multi-circuit
    runs may fan out to worker processes, where a closure callback would
    either fail to pickle or write to a store copy that is thrown away.
    """

    @functools.wraps(original)
    def run(
        self: Any,
        circuits: Any,
        output_name: Any = None,
        callback: Any = None,
        num_processes: Any = None,
        **kwargs: Any,
    ) -> Any:
        if STORE.is_suspended() or getattr(_local, "depth", 0) > 0:
            return original(self, circuits, output_name, callback, num_processes, **kwargs)
        circuit_cls = resolve("qiskit.circuit.quantumcircuit", "QuantumCircuit")
        single = circuit_cls is not None and isinstance(circuits, circuit_cls)
        if single:
            items = [circuits]
        elif (
            circuit_cls is not None
            and isinstance(circuits, (list, tuple))
            and circuits
            and all(isinstance(c, circuit_cls) for c in circuits)
        ):
            items = list(circuits)
        else:
            return original(self, circuits, output_name, callback, num_processes, **kwargs)

        timelines = [_timeline_unless_suspended(c) for c in items]
        for circuit, timeline in zip(items, timelines):
            if timeline is not None:
                STORE.add_event(
                    timeline,
                    TranspileStartedEvent,
                    pass_manager=type(self).__name__,
                    depth=circuit.depth(),
                    size=circuit.size(),
                )
        if len(items) == 1 and timelines[0] is not None:
            callback = _chained_callback(callback, timelines[0])

        started = time.perf_counter()
        _local.depth = 1
        try:
            with STORE.suspended():
                result = original(self, circuits, output_name, callback, num_processes, **kwargs)
        finally:
            _local.depth = 0
        duration = time.perf_counter() - started

        outputs = [result] if single else list(result)
        for timeline, output in zip(timelines, outputs):
            if timeline is None:
                continue
            snapshot = _output_snapshot(output)
            try:
                setattr(output, ID_ATTRIBUTE, timeline.circuit_id)
            except Exception:
                pass
            STORE.add_event(
                timeline,
                TranspileFinishedEvent,
                duration=duration,
                depth=output.depth(),
                size=output.size(),
                circuit=snapshot,
            )
        return result

    return run


def _timeline_unless_suspended(circuit: Any) -> Timeline | None:
    """Resolve the circuit's timeline, or None while its id is suspended.

    Backends compile handed-off circuits on internal worker threads where
    context-var suspension is invisible, so the execution hook marks their
    ids as suspended instead; honoring that here keeps backend-internal
    pass manager runs (e.g. Aer lowering control flow to jumps and marks)
    from overwriting the user-facing transpile history. A note event is
    left behind so the timeline says why nothing was recorded. Suspension
    is checked after resolution so copies that lost the id but re-adopt a
    suspended timeline by name are silenced too.
    """
    timeline = STORE.timeline_for(circuit, adopt_by_name=True)
    if timeline is None:
        return None
    if STORE.is_circuit_suspended(circuit):
        STORE.add_event(
            timeline,
            NoteEvent,
            message=(
                "pass manager run during execution not recorded "
                "(backend-internal compilation)"
            ),
        )
        return None
    return timeline


def _output_snapshot(output: Any) -> Any | None:
    """Copy the final transpiled circuit for the finished event.

    Runs before the timeline id is propagated onto the output, so the copy
    stays untracked; freezing a copy means later user mutations of the
    returned circuit don't rewrite the timeline. Returns None when snapshots
    are disabled or the copy fails.
    """
    if get_config().transpile_snapshots == "none":
        return None
    try:
        with STORE.suspended():
            return output.copy()
    except Exception:
        return None


def _chained_callback(user_callback: Callable | None, timeline: Timeline) -> Callable:
    """Build a pass callback that records telemetry then defers to the user's."""
    state: dict[str, Any] = {"signature": None}

    def callback(**details: Any) -> None:
        try:
            _record_pass(timeline, details, state)
        except Exception:
            logger.debug("qdebug: failed to record transpiler pass", exc_info=True)
        if user_callback is not None:
            user_callback(**details)

    return callback


def _fingerprint(dag: Any, depth: int | None, size: int | None, ops: dict) -> tuple:
    """Return a structural signature of a dag for change detection.

    Depth, size and op counts alone are blind to passes that only add wires
    or move operations between them (ancilla allocation, layout, routing),
    so bit counts and per-node placements are folded in as well.
    """
    try:
        qubits = {bit: i for i, bit in enumerate(dag.qubits)}
        clbits = {bit: i for i, bit in enumerate(dag.clbits)}
        nodes = tuple(
            (
                getattr(node, "name", None),
                tuple(qubits.get(q, -1) for q in node.qargs),
                tuple(clbits.get(c, -1) for c in node.cargs),
            )
            for node in dag.topological_op_nodes()
        )
        bits = (len(qubits), len(clbits))
    except Exception:
        nodes = None
        bits = None
    return (bits, depth, size, tuple(sorted(ops.items())), nodes)


def _record_pass(timeline: Timeline, details: dict, state: dict) -> None:
    """Record one transpiler pass, snapshotting the circuit when configured."""
    pass_instance = details.get("pass_")
    dag = details.get("dag")
    name = None
    if pass_instance is not None:
        namer = getattr(pass_instance, "name", None)
        name = namer() if callable(namer) else str(pass_instance)
    try:
        depth = dag.depth(recurse=True) if dag is not None else None
        size = dag.size(recurse=True) if dag is not None else None
    except Exception:
        depth = None
        size = None
    ops = dict(dag.count_ops()) if dag is not None else {}
    signature = _fingerprint(dag, depth, size, ops)
    changed = state["signature"] is None or signature != state["signature"]
    state["signature"] = signature

    config = get_config()
    circuit = None
    wants_snapshot = config.transpile_snapshots == "all" or (
        config.transpile_snapshots == "changed" and changed
    )
    if wants_snapshot and dag is not None:
        converter = resolve("qiskit.converters", "dag_to_circuit")
        if converter is not None:
            with STORE.suspended():
                try:
                    circuit = converter(dag)
                except Exception:
                    circuit = None
    STORE.add_event(
        timeline,
        TranspilePassEvent,
        pass_name=name or "unknown",
        duration=float(details.get("time") or 0.0),
        depth=depth,
        size=size,
        ops=ops,
        changed=changed,
        circuit=circuit,
    )
