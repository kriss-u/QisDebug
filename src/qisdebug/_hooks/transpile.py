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
from typing import TYPE_CHECKING, Any, Callable, NamedTuple

from .. import _dispatcher
from .._compat import logger, normalize_circuits, resolve
from .._dispatcher import InstallResult, all_suspended
from .._events import (
    NoteEvent,
    PropertySetChanges,
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
        ok, warning = _dispatcher.ensure_patched(
            base_pass_cls, "__call__", _pass_call_wrapper
        )
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
        fallback = _RunCall(
            original, self, circuits, output_name, callback, num_processes, kwargs
        )
        if getattr(_local, "depth", 0) > 0:
            return fallback.invoke()
        listeners = [
            d for d in _dispatcher.active_instances() if not d.store.is_suspended()
        ]
        if not listeners:
            return fallback.invoke()
        selection = normalize_circuits(circuits)
        if selection is None:
            return fallback.invoke()
        items, single = selection

        resolved = [_resolve_triples(listeners, c) for c in items]
        _record_started_events(items, resolved, type(self).__name__)
        if len(items) == 1 and resolved[0]:
            fallback = fallback._replace(callback=_chained_callback(callback, resolved[0]))

        result, duration = _run_instrumented(fallback, listeners)

        outputs = [result] if single else list(result)
        _record_finished_events(resolved, outputs, duration)
        return result

    return run


class _RunCall(NamedTuple):
    """The exact positional/keyword shape ``PassManager.run`` was invoked with."""

    original: Callable
    instance: Any
    circuits: Any
    output_name: Any
    callback: Any
    num_processes: Any
    kwargs: dict

    def invoke(self) -> Any:
        """Call ``original`` with the arguments it was originally given."""
        return self.original(
            self.instance,
            self.circuits,
            self.output_name,
            self.callback,
            self.num_processes,
            **self.kwargs,
        )


def _run_instrumented(
    call: "_RunCall", listeners: list["QisDebug"]
) -> tuple[Any, float]:
    """Invoke ``call`` with every listener suspended, timing the call.

    Suspension keeps the construction/execution hooks from recording the
    pass manager's own internal circuit manipulation as if it were user
    activity; ``_local.depth`` reentrancy-guards nested ``PassManager.run``
    calls the same way.
    """
    started = time.perf_counter()
    _local.depth = 1
    try:
        with all_suspended(listeners):
            result = call.invoke()
    finally:
        _local.depth = 0
    return result, time.perf_counter() - started


def _record_started_events(
    items: list[Any], resolved: list[list[TranspileTriple]], pass_manager_name: str
) -> None:
    """Emit one :class:`TranspileStartedEvent` per listener per circuit."""
    for circuit, triples in zip(items, resolved):
        for debug, timeline, phase in triples:
            debug.store.add_event(
                timeline,
                TranspileStartedEvent,
                pass_manager=pass_manager_name,
                depth=circuit.depth(),
                size=circuit.size(),
                phase=phase,
            )


def _record_finished_events(
    resolved: list[list[TranspileTriple]], outputs: list[Any], duration: float
) -> None:
    """Emit one :class:`TranspileFinishedEvent` per listener per output circuit."""
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
                circuit=(
                    snapshot if debug.config.transpile_snapshots != "none" else None
                ),
            )


def _resolve_triples(
    listeners: list["QisDebug"], circuit: Any
) -> list[TranspileTriple]:
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


def _chained_callback(
    user_callback: Callable | None, triples: list[TranspileTriple]
) -> Callable:
    """Build a pass callback that records telemetry then defers to the user's."""
    state: dict[str, Any] = {"signature": None, "property_set": None}

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


def _serialize_property_set(property_set: Any) -> dict[str, str]:
    """Best-effort snapshot of a pass's property_set as ``{key: repr(value)}``.

    Values like a ``Layout`` or ``TranspileLayout`` aren't JSON-native, so
    this falls back to ``repr()`` per key and never raises -- a property_set
    entry that can't be repr'd just gets a placeholder instead of losing the
    whole snapshot.
    """
    result: dict[str, str] = {}
    try:
        items = list(dict(property_set).items())
    except Exception:
        return result
    for key, value in items:
        try:
            result[key] = repr(value)
        except Exception:
            result[key] = f"<unrepr-able {type(value).__name__}>"
    return result


def _diff_property_set(
    old: dict[str, str] | None, new: dict[str, str]
) -> PropertySetChanges:
    """Return the added/removed/changed keys between two property_set snapshots."""
    old = old or {}
    added = {k: v for k, v in new.items() if k not in old}
    removed = {k: v for k, v in old.items() if k not in new}
    changed = {k: (old[k], new[k]) for k in new if k in old and old[k] != new[k]}
    return PropertySetChanges(added=added, removed=removed, changed=changed)


def _capture_circuit_snapshot(triples: list[TranspileTriple], dag: Any) -> Any | None:
    """Convert the pass's dag to a circuit, if any listener wants a snapshot at all."""
    wants_snapshot = any(
        debug.config.transpile_snapshots != "none" for debug, _, _ in triples
    )
    if dag is None or not wants_snapshot:
        return None
    converter = resolve("qiskit.converters", "dag_to_circuit")
    if converter is None:
        return None
    with all_suspended([debug for debug, _, _ in triples]):
        try:
            return converter(dag)
        except Exception:
            return None


def _policy_wants(policy: str, happened: bool) -> bool:
    """Return whether an "all"/"changed"/"none" capture policy keeps this pass."""
    return policy == "all" or (policy == "changed" and happened)


def _capture_property_set(
    triples: list[TranspileTriple], details: dict, state: dict
) -> tuple[dict[str, str] | None, PropertySetChanges | None]:
    """Snapshot and diff this pass's property_set, if any listener wants one."""
    if not any(debug.config.capture_property_set != "none" for debug, _, _ in triples):
        return None, None
    try:
        property_set = _serialize_property_set(details.get("property_set"))
        property_changes = _diff_property_set(state["property_set"], property_set)
        state["property_set"] = property_set
        return property_set, property_changes
    except Exception:
        logger.debug("qisdebug: failed to snapshot property_set", exc_info=True)
        return None, None


class _PassRecord(NamedTuple):
    """Everything a transpile_pass event needs, independent of who's listening."""

    name: str
    duration: float
    depth: int | None
    size: int | None
    ops: dict
    changed: bool
    circuit: Any
    property_set: Any
    property_changes: Any


def _pass_name(pass_instance: Any) -> str | None:
    """Return a pass's display name, whether ``.name`` is a method or attribute."""
    if pass_instance is None:
        return None
    namer = getattr(pass_instance, "name", None)
    return str(namer()) if callable(namer) else str(pass_instance)


def _dag_depth_size(dag: Any) -> tuple[int | None, int | None]:
    """Return ``(depth, size)`` for ``dag``, or ``(None, None)`` if unavailable."""
    if dag is None:
        return None, None
    try:
        return dag.depth(recurse=True), dag.size(recurse=True)
    except Exception:
        return None, None


def _record_pass(triples: list[TranspileTriple], details: dict, state: dict) -> None:
    """Record one transpiler pass, snapshotting the circuit when configured."""
    dag = details.get("dag")
    depth, size = _dag_depth_size(dag)
    ops = dict(dag.count_ops()) if dag is not None else {}
    signature = _fingerprint(dag, depth, size, ops)
    changed = state["signature"] is None or signature != state["signature"]
    state["signature"] = signature

    circuit = _capture_circuit_snapshot(triples, dag)
    property_set, property_changes = _capture_property_set(triples, details, state)

    record = _PassRecord(
        name=_pass_name(details.get("pass_")) or "unknown",
        duration=float(details.get("time") or 0.0),
        depth=depth,
        size=size,
        ops=ops,
        changed=changed,
        circuit=circuit,
        property_set=property_set,
        property_changes=property_changes,
    )
    _emit_pass_events(triples, record)


def _emit_pass_events(triples: list[TranspileTriple], record: "_PassRecord") -> None:
    """Add one transpile_pass event per listener, applying each one's own policy."""
    for debug, timeline, phase in triples:
        wants_snapshot = _policy_wants(debug.config.transpile_snapshots, record.changed)
        wants_props = record.property_changes is not None and _policy_wants(
            debug.config.capture_property_set, bool(record.property_changes)
        )
        debug.store.add_event(
            timeline,
            TranspilePassEvent,
            pass_name=record.name,
            duration=record.duration,
            depth=record.depth,
            size=record.size,
            ops=record.ops,
            changed=record.changed,
            phase=phase,
            circuit=record.circuit if wants_snapshot else None,
            property_changes=record.property_changes if wants_props else None,
            property_set=record.property_set if wants_props else None,
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
    pass_instance: Any,
    circuit: Any,
    result: Any,
    duration: float,
    listeners: list["QisDebug"],
) -> None:
    """Record one directly-applied pass as a transpile_pass event.

    The result circuit inherits the timeline id, so later transpiles or runs
    of it stay on the same timeline, mirroring ``PassManager.run`` outputs.
    """
    pairs = _resolve_pairs(listeners, circuit)
    if not pairs:
        return
    changed = _standalone_changed(result, circuit)
    depth, size, ops = _standalone_depth_size_ops(result)
    snapshot = _maybe_standalone_snapshot(pairs, result, changed)
    try:
        setattr(result, ID_ATTRIBUTE, pairs[0][1].circuit_id)
    except Exception:
        pass
    record = _PassRecord(
        name=_pass_name(pass_instance) or "unknown",
        duration=duration,
        depth=depth,
        size=size,
        ops=ops,
        changed=changed,
        circuit=snapshot,
        property_set=None,
        property_changes=None,
    )
    _emit_standalone_events(pairs, record)


def _resolve_pairs(
    listeners: list["QisDebug"], circuit: Any
) -> list[tuple["QisDebug", Timeline]]:
    """Resolve each listener's timeline for ``circuit``, dropping non-matches."""
    pairs: list[tuple["QisDebug", Timeline]] = []
    for debug in listeners:
        timeline = debug.store.timeline_for(
            circuit, max_circuits=debug.config.max_circuits, adopt_by_name=True
        )
        if timeline is not None:
            pairs.append((debug, timeline))
    return pairs


def _standalone_changed(result: Any, circuit: Any) -> bool:
    """Whether a directly-applied pass produced a different circuit object."""
    if result is circuit:
        return False
    try:
        return result != circuit
    except Exception:
        return True


def _standalone_depth_size_ops(result: Any) -> tuple[int | None, int | None, dict]:
    """Return ``(depth, size, ops)`` for ``result``, degrading gracefully."""
    try:
        return result.depth(), result.size(), dict(result.count_ops())
    except Exception:
        return None, None, {}


def _maybe_standalone_snapshot(
    pairs: list[tuple["QisDebug", Timeline]], result: Any, changed: bool
) -> Any | None:
    """Copy ``result`` if any listener's policy wants a snapshot for this pass."""
    wants_any = any(
        debug.config.transpile_snapshots == "all"
        or (debug.config.transpile_snapshots == "changed" and changed)
        for debug, _ in pairs
    )
    if not wants_any:
        return None
    try:
        with all_suspended([debug for debug, _ in pairs]):
            return result.copy()
    except Exception:
        return None


def _emit_standalone_events(
    pairs: list[tuple["QisDebug", Timeline]], record: "_PassRecord"
) -> None:
    """Add one transpile_pass event per listener for a directly-applied pass."""
    for debug, timeline in pairs:
        policy = debug.config.transpile_snapshots
        wants_snapshot = policy == "all" or (policy == "changed" and record.changed)
        debug.store.add_event(
            timeline,
            TranspilePassEvent,
            pass_name=record.name,
            duration=record.duration,
            depth=record.depth,
            size=record.size,
            ops=record.ops,
            changed=record.changed,
            circuit=record.circuit if wants_snapshot else None,
        )
