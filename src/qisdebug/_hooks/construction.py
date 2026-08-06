"""Layer one: record every instruction appended during circuit construction.

On Qiskit 2.x almost every append path (gate helper methods, the public
``append``, ``barrier``, ``measure``, ``initialize``) funnels through
``QuantumCircuit._append``, so patching it alongside the public ``append``
covers most construction activity. ``compose`` is the exception: it
bulk-extends the underlying circuit data without going through either, so it
is patched separately. Instead of parsing the many argument forms those
methods accept, the wrappers diff ``circuit.data`` around the original call
and record exactly the instructions that landed, after broadcasting.

The patch itself is installed once, process-wide, by :mod:`.._dispatcher`;
each wrapper fans the instructions it observes out to every currently
started :class:`~qisdebug.QisDebug` instance that isn't suspended for this
circuit, so multiple independently scoped instances each get their own copy
of the event in their own store.
"""

from __future__ import annotations

import functools
import threading
from typing import TYPE_CHECKING, Any, Callable

from .. import _dispatcher
from .._compat import logger, resolve
from .._dispatcher import InstallResult
from .._events import GateEvent

if TYPE_CHECKING:
    from .._debug import QisDebug

_local = threading.local()


def install() -> InstallResult:
    """Patch the QuantumCircuit append entry points; ``applied`` if any did."""
    warnings: list[str] = []
    targets: list[tuple[Any, str]] = []
    circuit_cls = resolve("qiskit.circuit.quantumcircuit", "QuantumCircuit")
    if circuit_cls is None:
        warnings.append("QuantumCircuit not found; construction layer disabled")
        return InstallResult(False, warnings, targets)
    applied = False
    for attribute in ("append", "_append"):
        if getattr(circuit_cls, attribute, None) is not None:
            ok, warning = _dispatcher.ensure_patched(
                circuit_cls, attribute, _recording_wrapper
            )
            if ok:
                targets.append((circuit_cls, attribute))
            applied = applied or ok
            if warning:
                warnings.append(warning)
    if not applied:
        warnings.append(
            "no QuantumCircuit append method found; construction layer disabled"
        )
        return InstallResult(False, warnings, targets)
    if getattr(circuit_cls, "compose", None) is not None:
        ok, warning = _dispatcher.ensure_patched(
            circuit_cls, "compose", _compose_wrapper
        )
        if ok:
            targets.append((circuit_cls, "compose"))
        if warning:
            warnings.append(warning)
    else:
        warnings.append(
            "QuantumCircuit.compose not found; composed circuits not recorded"
        )
    return InstallResult(applied, warnings, targets)


def _listeners(circuit: Any) -> list["QisDebug"]:
    """Return the active instances that should record events for ``circuit``."""
    return [
        debug
        for debug in _dispatcher.active_instances()
        if not debug.store.is_suspended()
        and not debug.store.is_circuit_suspended(circuit)
    ]


def _recording_wrapper(original: Callable) -> Callable:
    """Wrap an append-family method to diff and record circuit data.

    The thread-local depth guard makes only the outermost patched call
    record, since the public ``append`` calls ``_append`` internally and both
    are patched.
    """

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if getattr(_local, "depth", 0) > 0:
            return original(self, *args, **kwargs)
        listeners = _listeners(self)
        if not listeners:
            return original(self, *args, **kwargs)
        _local.depth = 1
        try:
            try:
                before = len(self.data)
            except Exception:
                before = None
            result = original(self, *args, **kwargs)
        finally:
            _local.depth = 0
        if before is not None:
            try:
                _record_appended(listeners, self, before)
            except Exception:
                logger.debug(
                    "qisdebug: failed to record appended instructions", exc_info=True
                )
        return result

    return wrapper


def _compose_wrapper(original: Callable) -> Callable:
    """Wrap ``compose`` to record instructions merged into ``self``.

    ``compose`` bulk-extends the circuit data without going through the
    append methods, so it gets the same diff-and-record treatment. Only
    ``inplace=True`` calls grow ``self.data`` (otherwise the diff is empty
    and nothing is recorded). ``front=True`` prepends, which a length diff
    cannot attribute correctly, so those calls are left unrecorded.
    """

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if getattr(_local, "depth", 0) > 0:
            return original(self, *args, **kwargs)
        listeners = _listeners(self)
        if not listeners:
            return original(self, *args, **kwargs)
        front = kwargs.get("front", args[3] if len(args) > 3 else False)
        _local.depth = 1
        try:
            try:
                before = len(self.data)
            except Exception:
                before = None
            result = original(self, *args, **kwargs)
        finally:
            _local.depth = 0
        if before is not None and not front:
            try:
                _record_appended(listeners, self, before)
            except Exception:
                logger.debug(
                    "qisdebug: failed to record composed instructions", exc_info=True
                )
        return result

    return wrapper


def _record_appended(listeners: list["QisDebug"], circuit: Any, before: int) -> None:
    """Record every instruction present in ``circuit.data`` beyond ``before``."""
    data = circuit.data
    after = len(data)
    if after <= before:
        return
    appended = list(data[before:after])
    for debug in listeners:
        timeline = debug.store.timeline_for(
            circuit, max_circuits=debug.config.max_circuits
        )
        if timeline is None:
            continue
        for instruction in appended:
            operation = getattr(instruction, "operation", None)
            if operation is None:
                continue
            debug.store.add_event(
                timeline,
                GateEvent,
                gate=getattr(operation, "name", type(operation).__name__),
                qubits=_bit_indices(circuit, instruction.qubits),
                clbits=_bit_indices(circuit, instruction.clbits),
                params=_parameter_values(operation),
                operation=operation,
            )


def _bit_indices(circuit: Any, bits: Any) -> tuple[int, ...]:
    """Map bit objects to their flat indices in the circuit."""
    indices = []
    for bit in bits:
        try:
            indices.append(circuit.find_bit(bit).index)
        except Exception:
            indices.append(-1)
    return tuple(indices)


def _parameter_values(operation: Any) -> tuple[float | str, ...]:
    """Coerce operation params to floats, falling back to compact text."""
    values: list[float | str] = []
    for param in getattr(operation, "params", ()) or ():
        try:
            values.append(float(param))
        except (TypeError, ValueError):
            values.append(_format_param(param))
    return tuple(values)


def _format_param(param: Any) -> str:
    """Render one non-numeric param as short single-line text.

    Control-flow ops carry their block bodies as whole circuits in
    ``params``, and ``str`` of a circuit is its multi-line drawing, so
    circuit-like params are summarized by their op counts instead. Anything
    else falls back to whitespace-collapsed, truncated ``str``.
    """
    counter = getattr(param, "count_ops", None)
    if callable(counter):
        try:
            op_counts: Any = counter()
            ops = ",".join(
                name if count == 1 else f"{name}*{count}"
                for name, count in op_counts.items()
            )
            return f"[{ops}]"
        except Exception:
            pass
    text = " ".join(str(param).split())
    if len(text) > 60:
        text = text[:57] + "..."
    return text
