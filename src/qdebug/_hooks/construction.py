"""Layer one: record every instruction appended during circuit construction.

On Qiskit 2.x every append path (gate helper methods, the public ``append``,
``barrier``, ``measure``, ``initialize``) funnels through
``QuantumCircuit._append``, so patching it alongside the public ``append``
covers all construction activity. Instead of parsing the many argument forms
those methods accept, the wrapper diffs ``circuit.data`` around the original
call and records exactly the instructions that landed, after broadcasting.
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
    return applied


def _recording_wrapper(original: Callable) -> Callable:
    """Wrap an append-family method to diff and record circuit data.

    The thread-local depth guard makes only the outermost patched call
    record, since the public ``append`` calls ``_append`` internally and both
    are patched.
    """

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if STORE.is_suspended() or getattr(_local, "depth", 0) > 0:
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
    """Coerce operation params to floats, falling back to their repr."""
    values: list[float | str] = []
    for param in getattr(operation, "params", ()) or ():
        try:
            values.append(float(param))
        except (TypeError, ValueError):
            values.append(str(param))
    return tuple(values)
