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

As with the other layers, the patch is shared process-wide (see
:mod:`.._dispatcher`); one pass manager run may serve several independently
configured :class:`~qisdebug.QisDebug` instances at once. The pass callback
and circuit snapshot are computed once per pass and handed to every listener
that wants one; each listener still applies its own ``transpile_snapshots``
policy to decide whether it keeps the snapshot on its own event.
"""

from __future__ import annotations

import functools
import threading
import time
from typing import TYPE_CHECKING, Any, Callable

from .. import _dispatcher
from .._compat import logger, resolve
from .._dispatcher import InstallResult, all_suspended
from .._events import (
    NoteEvent,
    Timeline,
    TranspileFinishedEvent,
    TranspilePassEvent,
    TranspileStartedEvent,
)
from .._store import ID_ATTRIBUTE

if TYPE_CHECKING:
    from .._debug import QisDebug

_local = threading.local()

TranspileTriple = tuple["QisDebug", Timeline, str]


def install() -> InstallResult:
    """Patch ``PassManager.run`` and ``BasePass.__call__``; ``applied`` when the former did."""
    warnings: list[str] = []
    targets: list[tuple[Any, str]] = []
    pass_manager_cls = resolve("qiskit.transpiler.passmanager", "PassManager")
    if pass_manager_cls is None:
        warnings.append("PassManager not found; transpile layer disabled")
        return InstallResult(False, warnings, targets)
    applied, warning = _dispatcher.ensure_patched(pass_manager_cls, "run", _run_wrapper)
    if applied:
        targets.append((pass_manager_cls, "run"))
    if warning:
        warnings.append(warning)
    base_pass_cls = resolve("qiskit.transpiler.basepasses", "BasePass")
    if base_pass_cls is not None:
        ok, warning = _dispatcher.ensure_patched(base_pass_cls, "__call__", _pass_call_wrapper)
        if ok:
            targets.append((base_pass_cls, "__call__"))
        if warning:
            warnings.append(warning)
    else:
        warnings.append("BasePass not found; directly-applied passes not recorded")
    return InstallResult(applied, warnings, targets)


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
        if getattr(_local, "depth", 0) > 0:
            return original(self, circuits, output_name, callback, num_processes, **kwargs)
        listeners = [d for d in _dispatcher.active_instances() if not d.store.is_suspended()]
        if not listeners:
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

        resolved = [_resolve_triples(listeners, c) for c in items]
        for circuit, triples in zip(items, resolved):
            for debug, timeline, phase in triples:
                debug.store.add_event(
                    timeline,
                    TranspileStartedEvent,
                    pass_manager=type(self).__name__,
                    depth=circuit.depth(),
                    size=circuit.size(),
                    phase=phase,
                )
        if len(items) == 1 and resolved[0]:
            callback = _chained_callback(callback, resolved[0])

        started = time.perf_counter()
        _local.depth = 1
        try:
            with all_suspended(listeners):
                result = original(self, circuits, output_name, callback, num_processes, **kwargs)
        finally:
            _local.depth = 0
        duration = time.perf_counter() - started

        outputs = [result] if single else list(result)
        for triples, output in zip(resolved, outputs):
            if not triples:
                continue
            snapshot = _maybe_output_snapshot(triples, output)
            try:
                setattr(output, ID_ATTRIBUTE, triples[0][1].circuit_id)
            except Exception:
                pass
            for debug, timeline, phase in triples:
                debug.store.add_event(
                    timeline,
                    TranspileFinishedEvent,
                    duration=duration,
                    depth=output.depth(),
                    size=output.size(),
                    phase=phase,
                    circuit=snapshot if debug.config.transpile_snapshots != "none" else None,
                )
        return result

    return run


def _resolve_triples(listeners: list["QisDebug"], circuit: Any) -> list[TranspileTriple]:
    """Resolve each listener's timeline for ``circuit`` and its compilation phase.

    A pass manager run on a circuit whose id an instance's own execution hook
    marked suspended is the backend compiling internally while executing it
    (Aer does this on worker threads, where context-var suspension is
    invisible). Those runs are recorded with ``phase="backend"`` so they land
    on their own visualizer track instead of overwriting the user-facing
    transpile history -- or, when that instance's ``capture_backend_compilation``
    is off, suppressed with a note event.
    """
    triples: list[TranspileTriple] = []
    for debug in listeners:
        timeline = debug.store.timeline_for(
            circuit, max_circuits=debug.config.max_circuits, adopt_by_name=True
        )
        if timeline is None:
            continue
        if not debug.store.is_circuit_suspended(circuit):
            triples.append((debug, timeline, "user"))
        elif debug.config.capture_backend_compilation:
            triples.append((debug, timeline, "backend"))
        else:
            debug.store.add_event(
                timeline,
                NoteEvent,
                message=(
                    "pass manager run during execution not recorded "
                    "(backend-internal compilation)"
                ),
            )
    return triples


def _maybe_output_snapshot(triples: list[TranspileTriple], output: Any) -> Any | None:
    """Copy the final transpiled circuit, if any listener wants snapshots at all.

    Runs before the timeline id is propagated onto the output, so the copy
    stays untracked; freezing a copy means later user mutations of the
    returned circuit don't rewrite the timeline. Returns None when no
    listener wants snapshots or the copy fails.
    """
    if not any(debug.config.transpile_snapshots != "none" for debug, _, _ in triples):
        return None
    try:
        with all_suspended([debug for debug, _, _ in triples]):
            return output.copy()
    except Exception:
        return None


def _chained_callback(user_callback: Callable | None, triples: list[TranspileTriple]) -> Callable:
    """Build a pass callback that records telemetry then defers to the user's."""
    state: dict[str, Any] = {"signature": None}

    def callback(**details: Any) -> None:
        try:
            _record_pass(triples, details, state)
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


def _record_pass(triples: list[TranspileTriple], details: dict, state: dict) -> None:
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

    circuit = None
    wants_snapshot = any(debug.config.transpile_snapshots != "none" for debug, _, _ in triples)
    if dag is not None and wants_snapshot:
        converter = resolve("qiskit.converters", "dag_to_circuit")
        if converter is not None:
            with all_suspended([debug for debug, _, _ in triples]):
                try:
                    circuit = converter(dag)
                except Exception:
                    circuit = None

    for debug, timeline, phase in triples:
        policy = debug.config.transpile_snapshots
        wants_snapshot = policy == "all" or (policy == "changed" and changed)
        debug.store.add_event(
            timeline,
            TranspilePassEvent,
            pass_name=name or "unknown",
            duration=float(details.get("time") or 0.0),
            depth=depth,
            size=size,
            ops=ops,
            changed=changed,
            phase=phase,
            circuit=circuit if wants_snapshot else None,
        )


def _pass_call_wrapper(original: Callable) -> Callable:
    """Wrap ``BasePass.__call__`` so directly-applied passes are recorded.

    Covers the standalone form ``SomePass()(circuit)``, which converts and
    runs the pass itself without ever entering ``PassManager.run``.
    """

    @functools.wraps(original)
    def call(self: Any, circuit: Any, *args: Any, **kwargs: Any) -> Any:
        if getattr(_local, "depth", 0) > 0:
            return original(self, circuit, *args, **kwargs)
        listeners = [
            d
            for d in _dispatcher.active_instances()
            if not d.store.is_suspended() and not d.store.is_circuit_suspended(circuit)
        ]
        if not listeners:
            return original(self, circuit, *args, **kwargs)
        started = time.perf_counter()
        _local.depth = 1
        try:
            with all_suspended(listeners):
                result = original(self, circuit, *args, **kwargs)
        finally:
            _local.depth = 0
        try:
            _record_standalone_pass(
                self, circuit, result, time.perf_counter() - started, listeners
            )
        except Exception:
            logger.debug("qisdebug: failed to record standalone pass", exc_info=True)
        return result

    return call


def _record_standalone_pass(
    pass_instance: Any, circuit: Any, result: Any, duration: float, listeners: list["QisDebug"]
) -> None:
    """Record one directly-applied pass as a transpile_pass event.

    The result circuit inherits the timeline id, so later transpiles or runs
    of it stay on the same timeline, mirroring ``PassManager.run`` outputs.
    """
    pairs: list[tuple["QisDebug", Timeline]] = []
    for debug in listeners:
        timeline = debug.store.timeline_for(
            circuit, max_circuits=debug.config.max_circuits, adopt_by_name=True
        )
        if timeline is not None:
            pairs.append((debug, timeline))
    if not pairs:
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
    snapshot = None
    if any(
        debug.config.transpile_snapshots == "all"
        or (debug.config.transpile_snapshots == "changed" and changed)
        for debug, _ in pairs
    ):
        try:
            with all_suspended([debug for debug, _ in pairs]):
                snapshot = result.copy()
        except Exception:
            snapshot = None
    try:
        setattr(result, ID_ATTRIBUTE, pairs[0][1].circuit_id)
    except Exception:
        pass
    for debug, timeline in pairs:
        policy = debug.config.transpile_snapshots
        wants_snapshot = policy == "all" or (policy == "changed" and changed)
        debug.store.add_event(
            timeline,
            TranspilePassEvent,
            pass_name=name or "unknown",
            duration=duration,
            depth=depth,
            size=size,
            ops=ops,
            changed=changed,
            circuit=snapshot if wants_snapshot else None,
        )
