"""Layer two: record execution and capture statevector snapshots on Aer.

The backend hook intercepts ``AerBackend.run``. For eligible circuits it
builds a copy with ``SaveStatevector`` instructions inserted after every
gate, runs that copy instead of the original, then strips the snapshot data
out of the result before the user sees it. ``BasicSimulator.run`` gets the
same wrapper without instrumentation (it cannot execute Aer save
instructions), and primitive classes (Sampler and Estimator flavors,
including the Backend* wrappers) are patched for execution timing events
only. Aer factory methods that mass-produce internal circuits
(``from_backend``) run with recording suspended so they cannot flood the
store.
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field

from .._compat import PatchRegistry, logger, resolve
from .._config import QisDebugConfig, get_config
from .._events import (
    ExecutionFinishedEvent,
    ExecutionSubmittedEvent,
    NoteEvent,
    StateSnapshotEvent,
    Timeline,
)
from .._store import ID_ATTRIBUTE, STORE

SNAPSHOT_LABEL_PREFIX = "qisdebug_sv_"

_UNSUPPORTED_OPS = frozenset(
    {"reset", "initialize", "kraus", "superop", "set_statevector", "set_density_matrix"}
)

_PRIMITIVE_TARGETS = (
    ("qiskit.primitives", "StatevectorSampler"),
    ("qiskit.primitives", "StatevectorEstimator"),
    ("qiskit.primitives", "BackendSamplerV2"),
    ("qiskit.primitives", "BackendEstimatorV2"),
    ("qiskit.primitives", "Sampler"),
    ("qiskit.primitives", "Estimator"),
    ("qiskit_aer.primitives", "SamplerV2"),
    ("qiskit_aer.primitives", "EstimatorV2"),
)


class SnapshotRef(BaseModel):
    """Where one SaveStatevector instruction sits in the instrumented copy."""

    label: str
    step: int
    gate_index: int | None = None
    gate: str | None = None


class RunEntry(BaseModel):
    """Telemetry bookkeeping for one circuit within a backend run."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    index: int
    timeline: Any = None
    snapshots: list[SnapshotRef] = Field(default_factory=list)
    backend_name: str | None = None
    shots: int | None = None


class RunPlan(BaseModel):
    """The payload actually submitted to the backend plus its bookkeeping."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    payload: Any
    entries: list[RunEntry] = Field(default_factory=list)
    submitted_at: float = Field(default_factory=time.time)


_INTERNAL_FACTORY_TARGETS = (
    ("qiskit_aer.backends.aer_simulator", "AerSimulator", "from_backend"),
    ("qiskit_aer.noise", "NoiseModel", "from_backend"),
)


def install(registry: PatchRegistry) -> bool:
    """Patch backend and primitive run methods; True if any applied."""
    applied = False
    aer_backend = resolve("qiskit_aer.backends.aerbackend", "AerBackend")
    if aer_backend is not None:
        applied = registry.patch(aer_backend, "run", _backend_run_wrapper) or applied
        for module_path, class_name, attribute in _INTERNAL_FACTORY_TARGETS:
            owner = resolve(module_path, class_name)
            if owner is not None:
                registry.patch(owner, attribute, _suspended_call_wrapper)
    else:
        logger.info("qisdebug: qiskit-aer not installed; backend statevector capture unavailable")
    basic_simulator = resolve("qiskit.providers.basic_provider.basic_simulator", "BasicSimulator")
    if basic_simulator is not None:
        applied = registry.patch(basic_simulator, "run", _plain_backend_run_wrapper) or applied
    for module_path, class_name in _PRIMITIVE_TARGETS:
        primitive_cls = resolve(module_path, class_name)
        if primitive_cls is not None:
            wrapper_factory = _primitive_run_factory(class_name)
            applied = registry.patch(primitive_cls, "run", wrapper_factory) or applied
    if not applied:
        registry.warn("no backend or primitive run method found; execution layer disabled")
    return applied


def _suspended_call_wrapper(original: Callable) -> Callable:
    """Run the wrapped callable with all recording suspended.

    Applied to backend and noise-model factory methods that internally build
    many small circuits (e.g. ``NoiseModel.from_backend`` composes one per
    quantum error); without suspension those would flood the store and evict
    the user's real timelines.
    """

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with STORE.suspended():
            return original(*args, **kwargs)

    return wrapper


def _backend_run_wrapper(original: Callable, instrument: bool = True) -> Callable:
    """Wrap a backend ``run`` with telemetry and result interception.

    With ``instrument`` the payload may be swapped for a statevector-saving
    copy; that must stay False for non-Aer backends, which would reject the
    Aer-specific save instructions.
    """

    @functools.wraps(original)
    def run(self: Any, circuits: Any, *args: Any, **run_options: Any) -> Any:
        if STORE.is_suspended():
            return original(self, circuits, *args, **run_options)
        try:
            plan = _plan_backend_run(self, circuits, run_options, instrument)
        except Exception:
            logger.debug("qisdebug: failed to plan instrumented run", exc_info=True)
            plan = None
        if plan is None:
            return original(self, circuits, *args, **run_options)
        circuit_ids = _plan_circuit_ids(plan)
        STORE.suspend_circuit_ids(circuit_ids)
        try:
            with STORE.suspended():
                job = original(self, plan.payload, *args, **run_options)
        except BaseException:
            STORE.resume_circuit_ids(circuit_ids)
            raise
        try:
            _record_submission(plan, job)
            _intercept_result(job, plan, circuit_ids)
        except Exception:
            logger.debug("qisdebug: failed to attach result interception", exc_info=True)
            STORE.resume_circuit_ids(circuit_ids)
        return job

    return run


def _plain_backend_run_wrapper(original: Callable) -> Callable:
    """Wrap a non-Aer backend's ``run`` for submission and completion telemetry."""
    return _backend_run_wrapper(original, instrument=False)


def _plan_circuit_ids(plan: RunPlan) -> set[str]:
    """Return the qisdebug ids of every circuit actually submitted to the backend.

    ``AerBackend.run`` hands these circuits to a persistent internal thread
    pool for compilation (e.g. inlining control-flow bodies), so they need
    to stay recognized as "ours" for the life of the job -- see
    :meth:`TelemetryStore.suspend_circuit_ids`.
    """
    payload = plan.payload if isinstance(plan.payload, list) else [plan.payload]
    return {circuit_id for c in payload if (circuit_id := getattr(c, ID_ATTRIBUTE, None))}


def _plan_backend_run(
    backend: Any, circuits: Any, run_options: dict, instrument: bool = True
) -> RunPlan | None:
    """Build the run plan, swapping eligible circuits for instrumented copies.

    Returns None when the input is not made of QuantumCircuits, which tells
    the wrapper to fall through to the original call untouched.
    """
    circuit_cls = resolve("qiskit.circuit.quantumcircuit", "QuantumCircuit")
    if circuit_cls is None:
        return None
    single = isinstance(circuits, circuit_cls)
    if single:
        items = [circuits]
    elif (
        isinstance(circuits, (list, tuple))
        and circuits
        and all(isinstance(c, circuit_cls) for c in circuits)
    ):
        items = list(circuits)
    else:
        return None
    backend_name_attr = getattr(backend, "name", None)
    backend_name = backend_name_attr() if callable(backend_name_attr) else backend_name_attr
    backend_name = str(backend_name) if backend_name is not None else None
    shots = run_options.get("shots")
    config = get_config()
    has_binds = bool(run_options.get("parameter_binds"))
    entries: list[RunEntry] = []
    payload: list[Any] = []
    for index, circuit in enumerate(items):
        timeline = STORE.timeline_for(circuit, adopt_by_name=True)
        entry = RunEntry(index=index, timeline=timeline, backend_name=backend_name, shots=shots)
        replacement = circuit
        if instrument and timeline is not None and config.capture_statevectors:
            if has_binds:
                STORE.add_event(
                    timeline,
                    NoteEvent,
                    message="statevector capture skipped: parameter_binds present",
                )
            else:
                instrumented = _instrument(circuit, timeline, config)
                if instrumented is not None:
                    replacement, entry.snapshots = instrumented
        payload.append(replacement)
        entries.append(entry)
    return RunPlan(payload=payload[0] if single else payload, entries=entries)


def _instrument(
    circuit: Any, timeline: Timeline, config: QisDebugConfig
) -> tuple[Any, list[SnapshotRef]] | None:
    """Copy ``circuit`` with a SaveStatevector inserted after every gate."""
    save_cls = resolve("qiskit_aer.library", "SaveStatevector")
    if save_cls is None:
        return None
    positions = _snapshot_positions(circuit, timeline, config)
    if positions is None:
        return None
    refs: list[SnapshotRef] = []
    with STORE.suspended():
        copy = circuit.copy_empty_like()

        def save(step: int, gate_index: int | None, gate: str | None) -> None:
            label = f"{SNAPSHOT_LABEL_PREFIX}{step}"
            copy.append(save_cls(circuit.num_qubits, label=label), copy.qubits)
            refs.append(SnapshotRef(label=label, step=step, gate_index=gate_index, gate=gate))

        save(0, None, None)
        step = 0
        for index, instruction in enumerate(circuit.data):
            copy.append(
                instruction.operation,
                [copy.qubits[circuit.find_bit(b).index] for b in instruction.qubits],
                [copy.clbits[circuit.find_bit(b).index] for b in instruction.clbits],
            )
            if index in positions:
                step += 1
                save(step, index, instruction.operation.name)
    return copy, refs


def _snapshot_positions(
    circuit: Any, timeline: Timeline, config: QisDebugConfig
) -> set[int] | None:
    """Return data indices to snapshot after, or None when capture must be skipped.

    Statevector saves are only deterministic while the circuit is a pure
    unitary evolution, so anything nondeterministic (reset, mid-circuit
    measurement, control flow, noise instructions) disables capture for the
    whole circuit. A trailing measurement layer is fine because snapshots
    stop before it.
    """

    def skip(reason: str) -> None:
        STORE.add_event(timeline, NoteEvent, message=f"statevector capture skipped: {reason}")

    if circuit.num_qubits == 0:
        return None
    if circuit.num_qubits > config.max_statevector_qubits:
        limit = config.max_statevector_qubits
        skip(f"{circuit.num_qubits} qubits exceeds max_statevector_qubits={limit}")
        return None
    control_flow = _control_flow_names()
    first_measure: int | None = None
    positions: list[int] = []
    for index, instruction in enumerate(circuit.data):
        name = instruction.operation.name
        if name in control_flow or name in _UNSUPPORTED_OPS:
            skip(f"'{name}' is not supported")
            return None
        if name == "measure":
            if first_measure is None:
                first_measure = index
            continue
        if name in ("barrier", "delay"):
            continue
        if first_measure is not None:
            skip("mid-circuit measurement")
            return None
        positions.append(index)
    if len(positions) + 1 > config.max_snapshots_per_circuit:
        keep = config.max_snapshots_per_circuit - 1
        STORE.add_event(
            timeline, NoteEvent, message=f"statevector capture truncated to the first {keep} gates"
        )
        positions = positions[:keep]
    return set(positions)


def _control_flow_names() -> frozenset[str]:
    """Return Qiskit's control-flow op names, with a fallback list."""
    names = resolve("qiskit.circuit.controlflow", "CONTROL_FLOW_OP_NAMES")
    if names:
        return frozenset(names)
    return frozenset({"if_else", "while_loop", "for_loop", "switch_case", "box"})


def _record_submission(plan: RunPlan, job: Any) -> None:
    """Record an execution_submitted event for every planned circuit."""
    job_id = getattr(job, "job_id", None)
    if callable(job_id):
        job_id = job_id()
    for entry in plan.entries:
        if entry.timeline is None:
            continue
        STORE.add_event(
            entry.timeline,
            ExecutionSubmittedEvent,
            backend=entry.backend_name,
            shots=entry.shots,
            job_id=str(job_id) if job_id is not None else None,
            instrumented=bool(entry.snapshots),
        )


def _intercept_result(job: Any, plan: RunPlan, circuit_ids: set[str]) -> None:
    """Override ``job.result`` so telemetry is ingested when results arrive.

    Also releases ``circuit_ids`` from suspension here, once the backend's
    internal compilation of them is guaranteed to be finished, rather than
    when ``run`` merely returns the (still in-flight) job.
    """
    original_result = job.result
    state = {"resumed": False, "ingested": False}

    @functools.wraps(original_result)
    def result(*args: Any, **kwargs: Any) -> Any:
        try:
            outcome = original_result(*args, **kwargs)
        finally:
            if not state["resumed"]:
                state["resumed"] = True
                STORE.resume_circuit_ids(circuit_ids)
        if not state["ingested"]:
            state["ingested"] = True
            try:
                _ingest_result(outcome, plan)
            except Exception:
                logger.debug("qisdebug: failed to ingest execution result", exc_info=True)
        return outcome

    job.result = result


def _ingest_result(result: Any, plan: RunPlan) -> None:
    """Pull snapshots out of the result, strip them, and record completion."""
    fallback_duration = time.time() - plan.submitted_at
    for entry in plan.entries:
        if entry.timeline is None:
            continue
        experiment = _experiment_for(result, entry.index)
        if entry.snapshots and experiment is not None:
            _extract_snapshots(experiment, entry)
        counts = None
        try:
            counts = dict(result.get_counts(entry.index))
        except Exception:
            counts = None
        duration = getattr(experiment, "time_taken", None) or getattr(result, "time_taken", None)
        STORE.add_event(
            entry.timeline,
            ExecutionFinishedEvent,
            backend=entry.backend_name,
            duration=float(duration) if duration is not None else fallback_duration,
            success=getattr(result, "success", None),
            shots=getattr(experiment, "shots", None) or entry.shots,
            counts=counts,
        )


def _experiment_for(result: Any, index: int) -> Any | None:
    """Return the per-circuit experiment result at ``index``, if present."""
    experiments = getattr(result, "results", None)
    if experiments and index < len(experiments):
        return experiments[index]
    return None


def _extract_snapshots(experiment: Any, entry: RunEntry) -> None:
    """Record saved statevectors and remove them from the user's result.

    Aer tracks data fields in ``_data_attributes``; the label must be removed
    from there too, otherwise ``Result.data()`` raises AttributeError on the
    deleted attribute.
    """
    data = getattr(experiment, "data", None)
    if data is None:
        return
    tracked = getattr(data, "_data_attributes", None)
    for ref in entry.snapshots:
        value = getattr(data, ref.label, None)
        if value is not None:
            STORE.add_event(
                entry.timeline,
                StateSnapshotEvent,
                step=ref.step,
                gate_index=ref.gate_index,
                gate=ref.gate,
                num_qubits=getattr(value, "num_qubits", None) or entry.timeline.num_qubits,
                statevector=value,
            )
            delattr(data, ref.label)
        if tracked is not None and ref.label in tracked:
            tracked.remove(ref.label)


def _primitive_run_factory(primitive_name: str) -> Callable[[Callable], Callable]:
    """Build a run wrapper for a Sampler/Estimator primitive class."""

    def factory(original: Callable) -> Callable:
        @functools.wraps(original)
        def run(self: Any, pubs: Any, *args: Any, **kwargs: Any) -> Any:
            if STORE.is_suspended():
                return original(self, pubs, *args, **kwargs)
            # Pubs may be a generator; materialize it so telemetry extraction
            # does not consume the caller's iterable.
            try:
                pubs = list(pubs)
            except TypeError:
                return original(self, pubs, *args, **kwargs)
            timelines = _pub_timelines(pubs)
            shots = kwargs.get("shots")
            started = time.time()
            # Primitives bind and rework id-carrying copies of the pub
            # circuits internally (often on job threads); suspending the ids
            # keeps that rework from being recorded as user construction.
            circuit_ids = {t.circuit_id for t in timelines}
            STORE.suspend_circuit_ids(circuit_ids)
            try:
                job = original(self, pubs, *args, **kwargs)
            except BaseException:
                STORE.resume_circuit_ids(circuit_ids)
                raise
            try:
                job_id = getattr(job, "job_id", None)
                if callable(job_id):
                    job_id = job_id()
                for timeline in timelines:
                    STORE.add_event(
                        timeline,
                        ExecutionSubmittedEvent,
                        backend=primitive_name,
                        shots=shots,
                        job_id=str(job_id) if job_id is not None else None,
                        instrumented=False,
                    )
                _intercept_primitive_result(
                    job, timelines, primitive_name, started, shots, circuit_ids
                )
            except Exception:
                logger.debug("qisdebug: failed to record primitive submission", exc_info=True)
                STORE.resume_circuit_ids(circuit_ids)
            return job

        return run

    return factory


def _pub_timelines(pubs: list) -> list[Timeline]:
    """Resolve one timeline per pub whose circuit can be identified."""
    circuit_cls = resolve("qiskit.circuit.quantumcircuit", "QuantumCircuit")
    if circuit_cls is None:
        return []
    timelines = []
    for pub in pubs:
        circuit = _pub_circuit(pub, circuit_cls)
        if circuit is None:
            continue
        timeline = STORE.timeline_for(circuit, adopt_by_name=True)
        if timeline is not None:
            timelines.append(timeline)
    return timelines


def _pub_circuit(pub: Any, circuit_cls: type) -> Any | None:
    """Extract the QuantumCircuit from a pub-like object."""
    if isinstance(pub, circuit_cls):
        return pub
    circuit = getattr(pub, "circuit", None)
    if isinstance(circuit, circuit_cls):
        return circuit
    if isinstance(pub, (tuple, list)) and pub and isinstance(pub[0], circuit_cls):
        return pub[0]
    return None


def _intercept_primitive_result(
    job: Any,
    timelines: list[Timeline],
    primitive_name: str,
    started: float,
    shots: int | None,
    circuit_ids: set[str],
) -> None:
    """Override the primitive job's result to record completion timing.

    Like the backend variant, this also releases ``circuit_ids`` from
    suspension once results are in and the primitive's internal circuit
    rework is guaranteed to be over.
    """
    original_result = job.result
    state = {"done": False, "resumed": False}

    @functools.wraps(original_result)
    def result(*args: Any, **kwargs: Any) -> Any:
        try:
            outcome = original_result(*args, **kwargs)
        finally:
            if not state["resumed"]:
                state["resumed"] = True
                STORE.resume_circuit_ids(circuit_ids)
        if not state["done"]:
            state["done"] = True
            duration = time.time() - started
            for timeline in timelines:
                try:
                    STORE.add_event(
                        timeline,
                        ExecutionFinishedEvent,
                        backend=primitive_name,
                        duration=duration,
                        success=True,
                        shots=shots,
                    )
                except Exception:
                    logger.debug("qisdebug: failed to record primitive completion", exc_info=True)
        return outcome

    job.result = result
