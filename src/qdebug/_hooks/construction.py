"""Layer one: record every instruction appended during circuit construction.

On Qiskit 2.x almost every append path (gate helper methods, the public
``append``, ``barrier``, ``measure``, ``initialize``) funnels through
``QuantumCircuit._append``, so patching it alongside the public ``append``
covers most construction activity. ``compose`` is the exception: it
bulk-extends the underlying circuit data without going through either, so it
is patched separately. Instead of parsing the many argument forms those
methods accept, the wrappers diff ``circuit.data`` around the original call
and record exactly the instructions that landed, after broadcasting.
"""

from __future__ import annotations

import functools
import threading
from typing import Any, Callable

from .._compat import PatchRegistry, logger, resolve
from .._events import GateEvent
from .._store import STORE

_local = threading.local()


def install(registry: PatchRegistry) -> bool:
    """Patch the QuantumCircuit append entry points; True if any applied."""
    circuit_cls = resolve("qiskit.circuit.quantumcircuit", "QuantumCircuit")
    if circuit_cls is None:
        registry.warn("QuantumCircuit not found; construction layer disabled")
        return False
    applied = False
    for attribute in ("append", "_append"):
        if getattr(circuit_cls, attribute, None) is not None:
            applied = registry.patch(circuit_cls, attribute, _recording_wrapper) or applied
    if not applied:
        registry.warn("no QuantumCircuit append method found; construction layer disabled")
        return False
    if getattr(circuit_cls, "compose", None) is not None:
        registry.patch(circuit_cls, "compose", _compose_wrapper)
    else:
        registry.warn("QuantumCircuit.compose not found; composed circuits not recorded")
    return applied


def _recording_wrapper(original: Callable) -> Callable:
    """Wrap an append-family method to diff and record circuit data.

    The thread-local depth guard makes only the outermost patched call
    record, since the public ``append`` calls ``_append`` internally and both
    are patched.
    """

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if (
            STORE.is_suspended()
            or STORE.is_circuit_suspended(self)
            or getattr(_local, "depth", 0) > 0
        ):
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
                _record_appended(self, before)
            except Exception:
                logger.debug("qdebug: failed to record appended instructions", exc_info=True)
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
        if (
            STORE.is_suspended()
            or STORE.is_circuit_suspended(self)
            or getattr(_local, "depth", 0) > 0
        ):
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
                _record_appended(self, before)
            except Exception:
                logger.debug("qdebug: failed to record composed instructions", exc_info=True)
        return result

    return wrapper


def _record_appended(circuit: Any, before: int) -> None:
    """Record every instruction present in ``circuit.data`` beyond ``before``."""
    data = circuit.data
    after = len(data)
    if after <= before:
        return
    timeline = STORE.timeline_for(circuit)
    if timeline is None:
        return
    for instruction in list(data[before:after]):
        operation = getattr(instruction, "operation", None)
        if operation is None:
            continue
        STORE.add_event(
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
            ops = ",".join(
                name if count == 1 else f"{name}*{count}" for name, count in counter().items()
            )
            return f"[{ops}]"
        except Exception:
            pass
    text = " ".join(str(param).split())
    if len(text) > 60:
        text = text[:57] + "..."
    return text
