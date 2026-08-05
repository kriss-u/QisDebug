"""Per-instance telemetry store keyed by hidden per-circuit ids.

Each :class:`~qisdebug.QisDebug` instance owns exactly one
:class:`TelemetryStore`; there is no process-wide singleton, so independent
``QisDebug()`` instances never see each other's timelines.
"""

from __future__ import annotations

import contextvars
import threading
import uuid
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Iterable, Iterator, Type, TypeVar

from ._events import TelemetryEvent, Timeline

ID_ATTRIBUTE = "_qisdebug_id"

EventT = TypeVar("EventT", bound=TelemetryEvent)


class TelemetryStore:
    """Thread-safe LRU store of circuit timelines.

    Recording hooks consult :meth:`is_suspended` before writing, so circuits
    QisDebug builds internally (instrumented copies, transpile snapshots) never
    pollute user telemetry. Suspension is tracked via a context variable
    rather than ``threading.local`` so it also covers ``asyncio`` tasks
    spawned while suspended.

    That alone is not enough for backends like Aer: ``AerBackend.run``
    hands the circuit off to a persistent internal ``ThreadPoolExecutor``
    whose workers capture their ``contextvars.Context`` once, at thread
    creation, and reuse it for every task afterwards, so a context variable
    set immediately before submitting work is never visible inside the
    worker. :meth:`suspend_circuit_ids`/:meth:`resume_circuit_ids` instead
    mark the specific circuit id being handed off as suspended in shared,
    lock-protected state, so recording hooks skip it regardless of which
    thread eventually touches it.
    """

    def __init__(self) -> None:
        self._timelines: OrderedDict[str, Timeline] = OrderedDict()
        self._lock = threading.RLock()
        self._sequence = 0
        self._depth: contextvars.ContextVar[int] = contextvars.ContextVar(
            "qisdebug_suspend_depth", default=0
        )
        self._suspended_ids: dict[str, int] = {}

    def next_sequence(self) -> int:
        """Return the next value of this store's monotonic event counter."""
        with self._lock:
            self._sequence += 1
            return self._sequence

    @contextmanager
    def suspended(self) -> Iterator[None]:
        """Silence all hook recording in the current context while active."""
        token = self._depth.set(self._depth.get() + 1)
        try:
            yield
        finally:
            self._depth.reset(token)

    def is_suspended(self) -> bool:
        """Return True when recording is suspended in the current context."""
        return self._depth.get() > 0

    def suspend_circuit_ids(self, circuit_ids: Iterable[str]) -> None:
        """Mark circuit ids as suspended regardless of which thread touches them.

        Reference-counted so overlapping runs on the same circuit id (e.g. a
        second submission before the first's job has resolved) don't resume
        each other early.
        """
        with self._lock:
            for circuit_id in circuit_ids:
                self._suspended_ids[circuit_id] = self._suspended_ids.get(circuit_id, 0) + 1

    def resume_circuit_ids(self, circuit_ids: Iterable[str]) -> None:
        """Undo one :meth:`suspend_circuit_ids` call for each id given."""
        with self._lock:
            for circuit_id in circuit_ids:
                count = self._suspended_ids.get(circuit_id, 0) - 1
                if count <= 0:
                    self._suspended_ids.pop(circuit_id, None)
                else:
                    self._suspended_ids[circuit_id] = count

    def is_circuit_suspended(self, circuit: Any) -> bool:
        """Return True when ``circuit``'s id was marked by :meth:`suspend_circuit_ids`."""
        circuit_id = getattr(circuit, ID_ATTRIBUTE, None)
        if circuit_id is None:
            return False
        with self._lock:
            return circuit_id in self._suspended_ids

    def timeline_for(
        self,
        circuit: Any,
        *,
        max_circuits: int = 256,
        create: bool = True,
        adopt_by_name: bool = False,
    ) -> Timeline | None:
        """Return the timeline tracking ``circuit``, creating one if needed.

        With ``adopt_by_name`` a circuit that carries no id yet (typically a
        transpiled copy, since Qiskit drops custom attributes there) is
        matched to the most recent timeline with the same circuit name.
        ``max_circuits`` caps how many timelines this store keeps before
        evicting the oldest.
        """
        with self._lock:
            circuit_id = getattr(circuit, ID_ATTRIBUTE, None)
            if circuit_id is None and adopt_by_name:
                adopted = self._adopt_by_name(circuit)
                if adopted is not None:
                    return adopted
            if circuit_id is None:
                if not create:
                    return None
                circuit_id = uuid.uuid4().hex[:12]
                try:
                    setattr(circuit, ID_ATTRIBUTE, circuit_id)
                except Exception:
                    return None
            timeline = self._timelines.get(circuit_id)
            if timeline is not None:
                self._timelines.move_to_end(circuit_id)
                return timeline
            if not create:
                return None
            timeline = Timeline(
                circuit_id=circuit_id,
                name=getattr(circuit, "name", None),
                num_qubits=getattr(circuit, "num_qubits", 0) or 0,
                num_clbits=getattr(circuit, "num_clbits", 0) or 0,
                qregs=_registers(circuit, "qregs"),
                cregs=_registers(circuit, "cregs"),
            )
            timeline.bind_store(self)
            self._timelines[circuit_id] = timeline
            while len(self._timelines) > max_circuits:
                self._timelines.popitem(last=False)
            return timeline

    def _adopt_by_name(self, circuit: Any) -> Timeline | None:
        """Attach an id-less circuit to the newest timeline sharing its name."""
        name = getattr(circuit, "name", None)
        if not name:
            return None
        for timeline in reversed(self._timelines.values()):
            if timeline.name == name:
                try:
                    setattr(circuit, ID_ATTRIBUTE, timeline.circuit_id)
                except Exception:
                    return None
                self._timelines.move_to_end(timeline.circuit_id)
                return timeline
        return None

    def add_event(self, timeline: Timeline, event_cls: Type[EventT], **fields: Any) -> EventT:
        """Construct an event with the next sequence number and append it."""
        event = event_cls(seq=self.next_sequence(), **fields)
        with self._lock:
            timeline.events.append(event)
        return event

    def get(self, circuit_or_id: Any) -> Timeline | None:
        """Look up a timeline by circuit object or by id string."""
        with self._lock:
            if isinstance(circuit_or_id, str):
                return self._timelines.get(circuit_or_id)
            circuit_id = getattr(circuit_or_id, ID_ATTRIBUTE, None)
            if circuit_id is None:
                return None
            return self._timelines.get(circuit_id)

    def get_by_name(self, name: str) -> Timeline | None:
        """Return the most-recently-touched timeline with the given circuit name."""
        with self._lock:
            for timeline in reversed(self._timelines.values()):
                if timeline.name == name:
                    return timeline
            return None

    def latest(self) -> Timeline | None:
        """Return the most recently touched timeline, if any."""
        with self._lock:
            if not self._timelines:
                return None
            return next(reversed(self._timelines.values()))

    def all_timelines(self) -> list[Timeline]:
        """Return every tracked timeline, most recently touched first."""
        with self._lock:
            return list(reversed(self._timelines.values()))

    def clear(self) -> None:
        """Drop all recorded telemetry."""
        with self._lock:
            self._timelines.clear()


def _registers(circuit: Any, attribute: str) -> dict[str, list[int]]:
    """Map register name to flat bit indices, for the watch feature.

    Best-effort: older/foreign circuit-like objects may lack ``qregs``,
    ``cregs``, or ``find_bit``, in which case this degrades to no registers
    rather than raising.
    """
    registers = {}
    for register in getattr(circuit, attribute, None) or ():
        try:
            registers[register.name] = [circuit.find_bit(bit).index for bit in register]
        except Exception:
            continue
    return registers
