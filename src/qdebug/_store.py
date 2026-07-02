"""Global telemetry store keyed by hidden per-circuit ids."""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Iterator, Type, TypeVar

from ._config import get_config
from ._events import TelemetryEvent, Timeline

ID_ATTRIBUTE = "_qdebug_id"

EventT = TypeVar("EventT", bound=TelemetryEvent)


class TelemetryStore:
    """Thread-safe LRU store of circuit timelines.

    Recording hooks consult :meth:`is_suspended` before writing, so circuits
    QDebug builds internally (instrumented copies, transpile snapshots) never
    pollute user telemetry. Suspension is tracked per thread.
    """

    def __init__(self) -> None:
        self._timelines: OrderedDict[str, Timeline] = OrderedDict()
        self._lock = threading.RLock()
        self._sequence = 0
        self._local = threading.local()

    def next_sequence(self) -> int:
        """Return the next value of the process-wide event counter."""
        with self._lock:
            self._sequence += 1
            return self._sequence

    @contextmanager
    def suspended(self) -> Iterator[None]:
        """Silence all hook recording on the current thread while active."""
        self._local.depth = getattr(self._local, "depth", 0) + 1
        try:
            yield
        finally:
            self._local.depth -= 1

    def is_suspended(self) -> bool:
        """Return True when recording is suspended on the current thread."""
        return getattr(self._local, "depth", 0) > 0

    def timeline_for(
        self, circuit: Any, create: bool = True, adopt_by_name: bool = False
    ) -> Timeline | None:
        """Return the timeline tracking ``circuit``, creating one if needed.

        With ``adopt_by_name`` a circuit that carries no id yet (typically a
        transpiled copy, since Qiskit drops custom attributes there) is
        matched to the most recent timeline with the same circuit name.
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
            )
            self._timelines[circuit_id] = timeline
            while len(self._timelines) > get_config().max_circuits:
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


STORE = TelemetryStore()
