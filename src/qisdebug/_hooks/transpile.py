"""Layer three: record per-pass circuit snapshots during transpilation.

``PassManager.run`` is the choke point for ``qiskit.transpile``, preset and
user-constructed pass managers alike (StagedPassManager delegates to it via
``super()``); ``BasePass.__call__`` covers the standalone
``SomePass()(circuit)`` form that never enters a pass manager. Per-pass
recording goes through Qiskit's own supported ``callback`` parameter rather
than patching pass internals, which makes this the most version-stable
layer. The wrapper also propagates the hidden circuit id onto transpiled
outputs, since Qiskit drops custom attributes there; that is what links
construction, transpilation and execution into one timeline.

Pass manager runs a backend performs internally while executing a circuit
(detected via the execution hook's suspended circuit ids) are recorded as
``phase="backend"`` events so backend-internal lowering is visible without
masquerading as user transpilation.
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
    """Patch ``PassManager.run`` and ``BasePass.__call__``; True when the former applied."""
    pass_manager_cls = resolve("qiskit.transpiler.passmanager", "PassManager")
    if pass_manager_cls is None:
        registry.warn("PassManager not found; transpile layer disabled")
        return False
    applied = registry.patch(pass_manager_cls, "run", _run_wrapper)
    base_pass_cls = resolve("qiskit.transpiler.basepasses", "BasePass")
    if base_pass_cls is not None:
        registry.patch(base_pass_cls, "__call__", _pass_call_wrapper)
    else:
        registry.warn("BasePass not found; directly-applied passes not recorded")
    return applied


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

        resolved = [_resolve_timeline(c) for c in items]
        timelines = [timeline for timeline, _ in resolved]
        phases = [phase for _, phase in resolved]
        for circuit, timeline, phase in zip(items, timelines, phases):
            if timeline is not None:
                STORE.add_event(
                    timeline,
                    TranspileStartedEvent,
                    pass_manager=type(self).__name__,
                    depth=circuit.depth(),
                    size=circuit.size(),
                    phase=phase,
                )
        if len(items) == 1 and timelines[0] is not None:
            callback = _chained_callback(callback, timelines[0], phases[0])

        started = time.perf_counter()
        _local.depth = 1
        try:
            with STORE.suspended():
                result = original(self, circuits, output_name, callback, num_processes, **kwargs)
        finally:
            _local.depth = 0
        duration = time.perf_counter() - started

        outputs = [result] if single else list(result)
        for timeline, phase, output in zip(timelines, phases, outputs):
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
                phase=phase,
                circuit=snapshot,
            )
        return result

    return run


def _resolve_timeline(circuit: Any) -> tuple[Timeline | None, str]:
    """Resolve the circuit's timeline and which compilation phase this run is.

    A pass manager run on a circuit whose id the execution hook marked
    suspended is the backend compiling internally while executing it (Aer
    does this on worker threads, where context-var suspension is invisible).
    Those runs are recorded with ``phase="backend"`` so they land on their
    own visualizer track instead of overwriting the user-facing transpile
    history -- or, when ``capture_backend_compilation`` is off, suppressed
    with a note event. Suspension is checked after resolution so copies that
    lost the id but re-adopt a suspended timeline by name are classified too.
    """
    timeline = STORE.timeline_for(circuit, adopt_by_name=True)
    if timeline is None:
        return None, "user"
    if not STORE.is_circuit_suspended(circuit):
        return timeline, "user"
    if get_config().capture_backend_compilation:
        return timeline, "backend"
    STORE.add_event(
        timeline,
        NoteEvent,
        message=(
            "pass manager run during execution not recorded "
            "(backend-internal compilation)"
        ),
    )
    return None, "user"


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


def _chained_callback(
    user_callback: Callable | None, timeline: Timeline, phase: str = "user"
) -> Callable:
    """Build a pass callback that records telemetry then defers to the user's."""
    state: dict[str, Any] = {"signature": None}

    def callback(**details: Any) -> None:
        try:
            _record_pass(timeline, details, state, phase)
        except Exception:
            logger.debug("qisdebug: failed to record transpiler pass", exc_info=True)
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


def _record_pass(timeline: Timeline, details: dict, state: dict, phase: str = "user") -> None:
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
        phase=phase,
        circuit=circuit,
    )


def _pass_call_wrapper(original: Callable) -> Callable:
    """Wrap ``BasePass.__call__`` so directly-applied passes are recorded.

    Covers the standalone form ``SomePass()(circuit)``, which converts and
    runs the pass itself without ever entering ``PassManager.run``.
    """

    @functools.wraps(original)
    def call(self: Any, circuit: Any, *args: Any, **kwargs: Any) -> Any:
        if (
            STORE.is_suspended()
            or STORE.is_circuit_suspended(circuit)
            or getattr(_local, "depth", 0) > 0
        ):
            return original(self, circuit, *args, **kwargs)
        started = time.perf_counter()
        _local.depth = 1
        try:
            with STORE.suspended():
                result = original(self, circuit, *args, **kwargs)
        finally:
            _local.depth = 0
        try:
            _record_standalone_pass(self, circuit, result, time.perf_counter() - started)
        except Exception:
            logger.debug("qisdebug: failed to record standalone pass", exc_info=True)
        return result

    return call


def _record_standalone_pass(
    pass_instance: Any, circuit: Any, result: Any, duration: float
) -> None:
    """Record one directly-applied pass as a transpile_pass event.

    The result circuit inherits the timeline id, so later transpiles or runs
    of it stay on the same timeline, mirroring ``PassManager.run`` outputs.
    """
    timeline = STORE.timeline_for(circuit, adopt_by_name=True)
    if timeline is None:
        return
    namer = getattr(pass_instance, "name", None)
    name = namer() if callable(namer) else str(pass_instance)
    if result is circuit:
        changed = False
    else:
        try:
            changed = result != circuit
        except Exception:
            changed = True
    try:
        depth = result.depth()
        size = result.size()
        ops = dict(result.count_ops())
    except Exception:
        depth = None
        size = None
        ops = {}
    policy = get_config().transpile_snapshots
    snapshot = None
    if policy == "all" or (policy == "changed" and changed):
        snapshot = _output_snapshot(result)
    try:
        setattr(result, ID_ATTRIBUTE, timeline.circuit_id)
    except Exception:
        pass
    STORE.add_event(
        timeline,
        TranspilePassEvent,
        pass_name=name or "unknown",
        duration=duration,
        depth=depth,
        size=size,
        ops=ops,
        changed=changed,
        circuit=snapshot,
    )
