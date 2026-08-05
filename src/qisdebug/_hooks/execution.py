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

Since the patch itself is shared process-wide (see :mod:`.._dispatcher`),
one real backend call may need to satisfy several independently configured
:class:`~qisdebug.QisDebug` instances at once. Statevector capture eligibility
is therefore decided once per run using the *widest* limits among the
instances that want capture at all -- an instrumented copy is built and
executed at most once -- and every one of those instances receives the same
captured snapshots. An instance with a narrower ``max_statevector_qubits``
than another concurrently active instance may thus see snapshots for a
circuit wider than its own configured limit; that's an accepted trade-off of
sharing one physical execution across independently configured sessions.
"""

from __future__ import annotations

import functools
import time
from typing import TYPE_CHECKING, Any, Callable, NamedTuple

from pydantic import BaseModel, ConfigDict, Field

from .. import _dispatcher
from .._compat import logger, resolve
from .._dispatcher import InstallResult, all_suspended
from .._events import (
    ExecutionFinishedEvent,
    ExecutionSubmittedEvent,
    NoteEvent,
    StateSnapshotEvent,
    Timeline,
)
from .._store import ID_ATTRIBUTE

if TYPE_CHECKING:
    from .._debug import QisDebug

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

TimelineHandle = tuple["QisDebug", Timeline]


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
    timelines: list[Any] = Field(default_factory=list)
    snapshot_timelines: list[Any] = Field(default_factory=list)
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


def install() -> InstallResult:
    """Patch backend and primitive run methods; ``applied`` if any did."""
    warnings: list[str] = []
    targets: list[tuple[Any, str]] = []
    applied = False
    aer_backend = resolve("qiskit_aer.backends.aerbackend", "AerBackend")
    if aer_backend is not None:
        ok, warning = _dispatcher.ensure_patched(aer_backend, "run", _backend_run_wrapper)
        if ok:
            targets.append((aer_backend, "run"))
        applied = applied or ok
        if warning:
            warnings.append(warning)
        for module_path, class_name, attribute in _INTERNAL_FACTORY_TARGETS:
            owner = resolve(module_path, class_name)
            if owner is not None:
                ok, warning = _dispatcher.ensure_patched(owner, attribute, _suspended_call_wrapper)
                if ok:
                    targets.append((owner, attribute))
                if warning:
                    warnings.append(warning)
    else:
        logger.info("qisdebug: qiskit-aer not installed; backend statevector capture unavailable")
    basic_simulator = resolve("qiskit.providers.basic_provider.basic_simulator", "BasicSimulator")
    if basic_simulator is not None:
        ok, warning = _dispatcher.ensure_patched(
            basic_simulator, "run", _plain_backend_run_wrapper
        )
        if ok:
            targets.append((basic_simulator, "run"))
        applied = applied or ok
        if warning:
            warnings.append(warning)
    for module_path, class_name in _PRIMITIVE_TARGETS:
        primitive_cls = resolve(module_path, class_name)
        if primitive_cls is not None:
            wrapper_factory = _primitive_run_factory(class_name)
            ok, warning = _dispatcher.ensure_patched(primitive_cls, "run", wrapper_factory)
            if ok:
                targets.append((primitive_cls, "run"))
            applied = applied or ok
            if warning:
                warnings.append(warning)
    if not applied:
        warnings.append("no backend or primitive run method found; execution layer disabled")
    return InstallResult(applied, warnings, targets)


def _suspend_ids(listeners: list["QisDebug"], circuit_ids: set[str]) -> None:
    for debug in listeners:
        debug.store.suspend_circuit_ids(circuit_ids)


def _resume_ids(listeners: list["QisDebug"], circuit_ids: set[str]) -> None:
    for debug in listeners:
        debug.store.resume_circuit_ids(circuit_ids)


def _suspended_call_wrapper(original: Callable) -> Callable:
    """Run the wrapped callable with all recording suspended.

    Applied to backend and noise-model factory methods that internally build
    many small circuits (e.g. ``NoiseModel.from_backend`` composes one per
    quantum error); without suspension those would flood the store and evict
    the user's real timelines.
    """

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with all_suspended(_dispatcher.active_instances()):
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
        listeners = [d for d in _dispatcher.active_instances() if not d.store.is_suspended()]
        if not listeners:
            return original(self, circuits, *args, **run_options)
        try:
            plan = _plan_backend_run(self, circuits, run_options, instrument, listeners)
        except Exception:
            logger.debug("qisdebug: failed to plan instrumented run", exc_info=True)
            plan = None
        if plan is None:
            return original(self, circuits, *args, **run_options)
        circuit_ids = _plan_circuit_ids(plan)
        _suspend_ids(listeners, circuit_ids)
        try:
            with all_suspended(listeners):
                job = original(self, plan.payload, *args, **run_options)
        except BaseException:
            _resume_ids(listeners, circuit_ids)
            raise
        try:
            _record_submission(plan, job)
            _intercept_result(job, plan, listeners, circuit_ids)
        except Exception:
            logger.debug("qisdebug: failed to attach result interception", exc_info=True)
            _resume_ids(listeners, circuit_ids)
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
    backend: Any,
    circuits: Any,
    run_options: dict,
    instrument: bool,
    listeners: list["QisDebug"],
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
    has_binds = bool(run_options.get("parameter_binds"))
    entries: list[RunEntry] = []
    payload: list[Any] = []
    for index, circuit in enumerate(items):
        timelines = _timelines_for(listeners, circuit)
        entry = RunEntry(index=index, timelines=timelines, backend_name=backend_name, shots=shots)
        replacement = circuit
        eligible = [(d, tl) for d, tl in timelines if instrument and d.config.capture_statevectors]
        if eligible:
            if has_binds:
                _add_event(
                    eligible,
                    NoteEvent,
                    message="statevector capture skipped: parameter_binds present",
                )
            else:
                instrumented = _instrument(circuit, eligible)
                if instrumented is not None:
                    replacement, entry.snapshots = instrumented
                    entry.snapshot_timelines = eligible
        payload.append(replacement)
        entries.append(entry)
    return RunPlan(payload=payload[0] if single else payload, entries=entries)


def _timelines_for(listeners: list["QisDebug"], circuit: Any) -> list[TimelineHandle]:
    """Resolve one timeline per listener that recognizes ``circuit``."""
    handles: list[TimelineHandle] = []
    for debug in listeners:
        timeline = debug.store.timeline_for(
            circuit, max_circuits=debug.config.max_circuits, adopt_by_name=True
        )
        if timeline is not None:
            handles.append((debug, timeline))
    return handles


def _add_event(handles: list[TimelineHandle], event_cls: type, **fields: Any) -> None:
    """Record the same event, independently sequenced, on every handle's store."""
    for debug, timeline in handles:
        debug.store.add_event(timeline, event_cls, **fields)


def _instrument(
    circuit: Any, eligible: list[TimelineHandle]
) -> tuple[Any, list[SnapshotRef]] | None:
    """Copy ``circuit`` with a SaveStatevector inserted after every gate."""
    save_cls = resolve("qiskit_aer.library", "SaveStatevector")
    if save_cls is None:
        return None
    positions = _snapshot_positions(circuit, eligible)
    if positions is None:
        return None
    refs: list[SnapshotRef] = []
    with all_suspended([d for d, _ in eligible]):
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


def _snapshot_positions(circuit: Any, eligible: list[TimelineHandle]) -> set[int] | None:
    """Return data indices to snapshot after, or None when capture must be skipped.

    Statevector saves are only deterministic while the circuit is a pure
    unitary evolution, so anything nondeterministic (reset, mid-circuit
    measurement, control flow, noise instructions) disables capture for the
    whole circuit. A trailing measurement layer is fine because snapshots
    stop before it. Limits use the widest value among ``eligible`` instances,
    since one instrumented copy must satisfy all of them at once.
    """

    def skip(reason: str) -> None:
        _add_event(eligible, NoteEvent, message=f"statevector capture skipped: {reason}")

    if circuit.num_qubits == 0:
        return None
    widest_qubits = max(debug.config.max_statevector_qubits for debug, _ in eligible)
    if circuit.num_qubits > widest_qubits:
        skip(f"{circuit.num_qubits} qubits exceeds max_statevector_qubits={widest_qubits}")
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
    widest_snapshots = max(debug.config.max_snapshots_per_circuit for debug, _ in eligible)
    if len(positions) + 1 > widest_snapshots:
        keep = widest_snapshots - 1
        _add_event(
            eligible, NoteEvent, message=f"statevector capture truncated to the first {keep} gates"
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
        _add_event(
            entry.timelines,
            ExecutionSubmittedEvent,
            backend=entry.backend_name,
            shots=entry.shots,
            job_id=str(job_id) if job_id is not None else None,
            instrumented=bool(entry.snapshots),
        )


def _intercept_result(
    job: Any, plan: RunPlan, listeners: list["QisDebug"], circuit_ids: set[str]
) -> None:
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
                _resume_ids(listeners, circuit_ids)
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
        if not entry.timelines:
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
        _add_event(
            entry.timelines,
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
    num_qubits = entry.timelines[0][1].num_qubits if entry.timelines else None
    for ref in entry.snapshots:
        value = getattr(data, ref.label, None)
        if value is not None:
            _add_event(
                entry.snapshot_timelines,
                StateSnapshotEvent,
                step=ref.step,
                gate_index=ref.gate_index,
                gate=ref.gate,
                num_qubits=getattr(value, "num_qubits", None) or num_qubits,
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
            listeners = [d for d in _dispatcher.active_instances() if not d.store.is_suspended()]
            if not listeners:
                return original(self, pubs, *args, **kwargs)
            # Pubs may be a generator; materialize it so telemetry extraction
            # does not consume the caller's iterable.
            try:
                pubs = list(pubs)
            except TypeError:
                return original(self, pubs, *args, **kwargs)
            timelines = _pub_timelines(listeners, pubs)
            shots = kwargs.get("shots")
            started = time.time()
            # Primitives bind and rework id-carrying copies of the pub
            # circuits internally (often on job threads); suspending the ids
            # keeps that rework from being recorded as user construction.
            circuit_ids = {timeline.circuit_id for _, timeline in timelines}
            _suspend_ids(listeners, circuit_ids)
            try:
                job = original(self, pubs, *args, **kwargs)
            except BaseException:
                _resume_ids(listeners, circuit_ids)
                raise
            try:
                job_id = getattr(job, "job_id", None)
                if callable(job_id):
                    job_id = job_id()
                _add_event(
                    timelines,
                    ExecutionSubmittedEvent,
                    backend=primitive_name,
                    shots=shots,
                    job_id=str(job_id) if job_id is not None else None,
                    instrumented=False,
                )
                submission = PrimitiveSubmission(
                    timelines=timelines,
                    primitive_name=primitive_name,
                    started=started,
                    shots=shots,
                    listeners=listeners,
                    circuit_ids=circuit_ids,
                )
                _intercept_primitive_result(job, submission)
            except Exception:
                logger.debug("qisdebug: failed to record primitive submission", exc_info=True)
                _resume_ids(listeners, circuit_ids)
            return job

        return run

    return factory


def _pub_timelines(listeners: list["QisDebug"], pubs: list) -> list[TimelineHandle]:
    """Resolve one timeline handle per (listener, pub) whose circuit can be identified."""
    circuit_cls = resolve("qiskit.circuit.quantumcircuit", "QuantumCircuit")
    if circuit_cls is None:
        return []
    handles: list[TimelineHandle] = []
    for pub in pubs:
        circuit = _pub_circuit(pub, circuit_cls)
        if circuit is None:
            continue
        handles.extend(_timelines_for(listeners, circuit))
    return handles


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


class PrimitiveSubmission(NamedTuple):
    """Bookkeeping for one primitive ``run`` call, carried into its result callback."""

    timelines: list[TimelineHandle]
    primitive_name: str
    started: float
    shots: int | None
    listeners: list["QisDebug"]
    circuit_ids: set[str]


def _intercept_primitive_result(job: Any, submission: PrimitiveSubmission) -> None:
    """Override the primitive job's result to record completion timing.

    Like the backend variant, this also releases the submission's circuit ids
    from suspension once results are in and the primitive's internal circuit
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
                _resume_ids(submission.listeners, submission.circuit_ids)
        if not state["done"]:
            state["done"] = True
            duration = time.time() - submission.started
            try:
                _add_event(
                    submission.timelines,
                    ExecutionFinishedEvent,
                    backend=submission.primitive_name,
                    duration=duration,
                    success=True,
                    shots=submission.shots,
                )
            except Exception:
                logger.debug("qisdebug: failed to record primitive completion", exc_info=True)
        return outcome

    job.result = result
