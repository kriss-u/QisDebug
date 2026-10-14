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

from ._breakpoints import BreakpointCondition, BreakpointHit, matches
from ._events import TelemetryEvent, Timeline

ID_ATTRIBUTE = "_qisdebug_id"

EventT = TypeVar("EventT", bound=TelemetryEvent)


class BreakpointStopped(BaseException):
    """Raised on the paused thread when :meth:`TelemetryStore.stop` cancels it.

    Deliberately a direct :class:`BaseException` subclass, not
    :class:`Exception` -- the same reason :class:`KeyboardInterrupt` is. The
    recording hooks (:mod:`qisdebug._hooks`) wrap every call into
    :meth:`TelemetryStore.add_event` in a broad ``except Exception:`` so a bug
    in *recording* telemetry can never crash the user's actual program; that
    same broad catch would otherwise also swallow this deliberate
    cancellation signal before it ever reached the caller that requested it.
    Propagates out through whatever Qiskit call the user's code made when the
    breakpoint hit, the same way any other exception from their script would
    -- callers that run user scripts (e.g.
    :func:`qisdebug._mcp.tools.run_script`) already catch it (checking for it
    ahead of any broader ``except Exception:``) and report it as the run's
    status rather than letting it escape further.
    """


class _PauseGate:
    """The ``threading.Event`` block/release/cancel primitives behind one
    live breakpoint pause, split out of :class:`BreakpointRegistry` so that
    class's own attribute count stays about condition bookkeeping.
    """

    def __init__(self) -> None:
        self._pause_event = threading.Event()
        self._pause_event.set()
        self._hit_event = threading.Event()
        self.last_hit: BreakpointHit | None = None
        self.cancelled = False

    @property
    def is_paused(self) -> bool:
        """True while some thread is blocked at a breakpoint hit."""
        return not self._pause_event.is_set()

    def resume(self) -> None:
        """Release a thread currently blocked at a breakpoint hit."""
        self._pause_event.set()

    def stop(self) -> None:
        """Arm cancellation: :meth:`block` raises instead of returning
        normally, whether it's already waiting or hasn't been called yet."""
        self.cancelled = True
        self._pause_event.set()

    def wait_for_hit(self, timeout: float | None = None) -> BreakpointHit | None:
        """Block the calling thread until the next breakpoint hit, or timeout.

        For a caller running the instrumented script on its own thread. Does
        *not* resume the hit thread -- call :meth:`resume` (or :meth:`stop`)
        once the hit has been inspected.
        """
        if self._hit_event.wait(timeout=timeout):
            self._hit_event.clear()
            return self.last_hit
        return None

    def block(self, hit: BreakpointHit) -> None:
        """Record ``hit`` and block the calling thread until :meth:`resume`.

        Raises :class:`BreakpointStopped` instead of returning normally if
        :meth:`stop` was called either before this was reached or while it
        was blocked.
        """
        if self.cancelled:
            self.cancelled = False
            raise BreakpointStopped(
                f"breakpoint {hit.handle} hit after stop() was called"
            )
        self.last_hit = hit
        self._pause_event.clear()
        self._hit_event.set()
        self._pause_event.wait(timeout=None)
        if self.cancelled:
            self.cancelled = False
            raise BreakpointStopped(
                f"breakpoint {hit.handle} hit; stop() cancelled the run while paused"
            )


class BreakpointRegistry:
    """One store's conditional breakpoints, plus the live-pause primitives.

    Split out of :class:`TelemetryStore` so that class's own attribute/method
    surface stays about timeline storage; everything breakpoint-related
    (condition registry, per-circuit transpile-depth tracking for the
    anomaly condition, and the pause/resume/stop machinery in
    :class:`_PauseGate`) lives here as one cohesive, independently testable
    unit, reached from the store as ``store.breakpoints``.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._conditions: dict[str, BreakpointCondition] = {}
        self._next_id = 0
        self._prev_transpile_depth: dict[str, int | None] = {}
        self._pause = _PauseGate()

    def add(self, condition: BreakpointCondition) -> str:
        """Register a condition, returning a handle for :meth:`remove`."""
        with self._lock:
            self._next_id += 1
            handle = str(self._next_id)
            self._conditions[handle] = condition
        return handle

    def remove(self, handle: str) -> None:
        """Unregister a breakpoint by its handle. No-op if already removed."""
        with self._lock:
            self._conditions.pop(handle, None)

    def list(self) -> list[tuple[str, BreakpointCondition]]:
        """Return every active ``(handle, condition)`` pair."""
        with self._lock:
            return list(self._conditions.items())

    def forget_circuit(self, circuit_id: str) -> None:
        """Drop tracked transpile-depth state for an evicted/cleared circuit."""
        with self._lock:
            self._prev_transpile_depth.pop(circuit_id, None)

    def clear_circuit_state(self) -> None:
        """Drop all tracked transpile-depth state (see :meth:`forget_circuit`).

        Called by :meth:`TelemetryStore.clear`; leaves registered conditions
        (and any live pause) untouched, since those are independent of which
        timelines happen to be currently stored.
        """
        with self._lock:
            self._prev_transpile_depth.clear()

    @property
    def is_paused(self) -> bool:
        """True while some thread is blocked at a breakpoint hit."""
        return self._pause.is_paused

    @property
    def last_hit(self) -> BreakpointHit | None:
        """The most recent breakpoint hit, whether or not it's still paused."""
        return self._pause.last_hit

    def resume(self) -> None:
        """Release a thread currently blocked at a breakpoint hit."""
        self._pause.resume()

    def wait_for_hit(self, timeout: float | None = None) -> BreakpointHit | None:
        """See :meth:`_PauseGate.wait_for_hit`."""
        return self._pause.wait_for_hit(timeout=timeout)

    def stop(self) -> None:
        """Best-effort cancellation: release a paused thread by raising there.

        If no thread is currently paused, only arms a flag so the *next*
        hit raises immediately instead of blocking -- Python threads can't be
        forcibly killed, so a script that's actively running (not currently
        blocked at a breakpoint) keeps running until it either finishes on
        its own or reaches another breakpoint.
        """
        self._pause.stop()

    def check(self, timeline: Timeline, event: TelemetryEvent) -> None:
        """Evaluate every registered condition against a newly recorded event.

        Called from :meth:`TelemetryStore.add_event` -- the one choke point
        every hook already funnels through -- right after it releases its
        own lock. The actual blocking wait (:meth:`_PauseGate.block`)
        deliberately happens outside any lock: blocking while holding one
        would stall any other thread needing it (eviction, concurrent
        construction on another thread) for the entire pause, not just the
        calling thread.
        """
        with self._lock:
            prev_depth = self._prev_transpile_depth.get(timeline.circuit_id)
            hit_handle: str | None = None
            hit_condition: BreakpointCondition | None = None
            for handle, condition in self._conditions.items():
                if matches(condition, event, timeline, prev_depth=prev_depth):
                    hit_handle, hit_condition = handle, condition
                    break
            depth = getattr(event, "depth", None)
            if depth is not None:
                self._prev_transpile_depth[timeline.circuit_id] = depth
        if hit_handle is None or hit_condition is None:
            return
        self._pause.block(
            BreakpointHit(
                handle=hit_handle,
                condition=hit_condition,
                circuit_id=timeline.circuit_id,
                seq=event.seq,
                event_summary=event.describe(),
            )
        )


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
        self.breakpoints = BreakpointRegistry()

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
                self._suspended_ids[circuit_id] = (
                    self._suspended_ids.get(circuit_id, 0) + 1
                )

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
                evicted_id, _ = self._timelines.popitem(last=False)
                self.breakpoints.forget_circuit(evicted_id)
            return timeline

    def grow_layout(self, timeline: Timeline, circuit: Any) -> None:
        """Extend ``timeline``'s bit counts and registers to cover ``circuit``.

        The layout captured when a timeline is created goes stale as soon as
        the circuit gains registers mid-construction (``measure_all`` adding
        ``meas``, an explicit ``add_register``), which leaves recorded events
        pointing at bits the timeline doesn't know about. Only ever grows:
        a transpiled copy shares its source's timeline, and shrinking to
        whichever of the two was touched last would orphan events recorded
        against the other.
        """
        num_qubits = getattr(circuit, "num_qubits", 0) or 0
        num_clbits = getattr(circuit, "num_clbits", 0) or 0
        qregs = getattr(circuit, "qregs", None) or ()
        cregs = getattr(circuit, "cregs", None) or ()
        with self._lock:
            if (
                num_qubits <= timeline.num_qubits
                and num_clbits <= timeline.num_clbits
                and all(register.name in timeline.qregs for register in qregs)
                and all(register.name in timeline.cregs for register in cregs)
            ):
                return
            timeline.num_qubits = max(timeline.num_qubits, num_qubits)
            timeline.num_clbits = max(timeline.num_clbits, num_clbits)
            for name, bits in _registers(circuit, "qregs").items():
                timeline.qregs.setdefault(name, bits)
            for name, bits in _registers(circuit, "cregs").items():
                timeline.cregs.setdefault(name, bits)

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

    def add_event(
        self, timeline: Timeline, event_cls: Type[EventT], **fields: Any
    ) -> EventT:
        """Construct an event with the next sequence number and append it."""
        event = event_cls(seq=self.next_sequence(), **fields)
        with self._lock:
            timeline.events.append(event)
        self.breakpoints.check(timeline, event)
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
            self.breakpoints.clear_circuit_state()


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
