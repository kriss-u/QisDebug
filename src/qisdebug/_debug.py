"""The public entry point: an independently scoped instrumentation session.

Each :class:`QisDebug` owns its own :class:`~qisdebug._store.TelemetryStore`
and :class:`~qisdebug.QisDebugConfig`, so multiple instances can be started at
once without seeing each other's timelines or configuration. The actual
Qiskit monkeypatches are shared process-wide (see :mod:`._dispatcher`) so two
instances patching the same Qiskit method don't race each other's restore.
"""

from __future__ import annotations

import warnings
import weakref
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from . import _dispatcher
from ._analysis import TimelineDiff, diff_timelines
from ._compat import logger, qiskit_version
from ._config import QisDebugConfig
from ._errors import TimelineNotFoundError
from ._events import Timeline
from ._frontends import get_frontend
from ._hooks import construction, execution, transpile
from ._store import TelemetryStore

if TYPE_CHECKING:
    # Only ever referenced from string annotations (`from __future__ import
    # annotations` defers all evaluation), so this never needs QuantumCircuit
    # to be importable at runtime -- matching the rest of the codebase's
    # policy of not hard-depending on qiskit internals at import time.
    from qiskit.circuit import QuantumCircuit

    TimelineLike = QuantumCircuit | str | Timeline
    TimelineContainer = (
        list[TimelineLike] | tuple[TimelineLike, ...] | set[TimelineLike]
    )
    # A container type rather than a generic Iterable so isinstance narrows
    # cleanly in _resolve_timelines: list/tuple/set are the only container
    # shapes it actually recognizes.
    TimelineTargets = TimelineLike | TimelineContainer | None


class ActivationReport(BaseModel):
    """Summary of which instrumentation layers attached at :meth:`QisDebug.start`."""

    active: bool
    qiskit_version: str | None = None
    layers: dict[str, bool] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


def _warn_if_still_active(active_state: dict[str, bool]) -> None:
    """Fire at garbage collection if ``end()`` was never called."""
    if active_state["active"]:
        warnings.warn(
            "a QisDebug instance was garbage collected without calling end(); "
            "its instrumentation patches stay installed until every other "
            "started instance using them also ends",
            ResourceWarning,
            stacklevel=2,
        )


class QisDebug:
    """An independently scoped QisDebug instrumentation session.

    Everything that happens to any Qiskit circuit between :meth:`start` and
    :meth:`end` is recorded automatically -- there is no manual "track this
    circuit" step. Query what was captured with :meth:`timelines`,
    :meth:`get`, or :meth:`circuits`::

        debug = QisDebug()
        debug.start()

        # ... build, transpile and run circuits exactly as usual ...

        print(debug.get(qc).summary())
        debug.end()

    ``start()``/``end()`` are the primary pattern -- flat calls that don't
    require re-indenting existing code. ``with QisDebug() as debug:`` also
    works, calling ``start()``/``end()`` automatically, for callers who
    prefer a scoped block.
    """

    def __init__(self, *, config: QisDebugConfig | None = None) -> None:
        self.store = TelemetryStore()
        self.config = config or QisDebugConfig()
        self._patch_targets: list[tuple[Any, str]] = []
        self._active_state = {"active": False}
        self._report: ActivationReport | None = None
        self._finalizer = weakref.finalize(
            self, _warn_if_still_active, self._active_state
        )

    def start(self) -> ActivationReport:
        """Attach all instrumentation layers to Qiskit.

        Idempotent: calling it again while active returns the existing report
        without reinstalling anything. Layers whose Qiskit internals are
        missing are skipped with a logged warning rather than raising, and
        the returned report says which layers actually attached.
        """
        if self._active_state["active"] and self._report is not None:
            return self._report
        results = [construction.install(), execution.install(), transpile.install()]
        layers = {
            "construction": results[0].applied,
            "execution": results[1].applied,
            "transpile": results[2].applied,
        }
        self._patch_targets = [
            target for result in results for target in result.targets
        ]
        _dispatcher.register(self)
        self._active_state["active"] = True
        report = ActivationReport(
            active=True,
            qiskit_version=qiskit_version(),
            layers=layers,
            warnings=[warning for result in results for warning in result.warnings],
        )
        self._report = report
        if not any(layers.values()):
            logger.warning("qisdebug: start() attached no instrumentation layers")
        else:
            attached = ", ".join(name for name, ok in layers.items() if ok)
            logger.info("qisdebug: started (layers: %s)", attached)
        return report

    def end(self) -> None:
        """Restore every patch this instance was using. Safe to call when not started."""
        if not self._active_state["active"]:
            return
        _dispatcher.unregister(self)
        for owner, attribute in self._patch_targets:
            _dispatcher.release_patched(owner, attribute)
        self._patch_targets = []
        self._active_state["active"] = False
        self._report = None

    @property
    def active(self) -> bool:
        """Return True while this instance's patches are applied."""
        return self._active_state["active"]

    def timelines(self) -> list[Timeline]:
        """Return every timeline this instance has recorded, most recently touched first."""
        return self.store.all_timelines()

    def circuits(self) -> list[str]:
        """Return the names of tracked circuits, most recently touched first."""
        return [
            timeline.name for timeline in self.store.all_timelines() if timeline.name
        ]

    def get(self, target: TimelineLike) -> Timeline:
        """Return the timeline for ``target``.

        ``target`` may be a live circuit object, a circuit name (the most
        recently touched timeline with that name is returned), or a
        :class:`Timeline` already obtained from this instance. Raises
        :class:`~qisdebug.TimelineNotFoundError` when nothing matches.
        """
        if isinstance(target, Timeline):
            found = self.store.get(target.circuit_id)
        elif isinstance(target, str):
            found = self.store.get_by_name(target) or self.store.get(target)
        else:
            found = self.store.get(target)
        if found is None:
            raise TimelineNotFoundError(f"no tracked circuit matches {target!r}")
        return found

    def _resolve_timelines(self, target: TimelineTargets) -> list[Timeline]:
        """Resolve a ``show()``/``export()``-style target into a list of timelines.

        ``target`` may be omitted (``None``) to mean every timeline this
        instance has recorded, a single circuit/name/:class:`Timeline`, or a
        list/tuple/set of any of those.
        """
        if target is None:
            return self.timelines()
        if isinstance(target, (list, tuple, set)):
            return [self.get(item) for item in target]
        return [self.get(target)]

    def show(
        self,
        target: TimelineTargets = None,
        *,
        frontend: str | None = None,
    ) -> Any:
        """Display timelines through the best available frontend.

        ``target`` may be omitted to show every timeline this instance has
        recorded, a single circuit/name/:class:`Timeline` to show just that
        one, or a list of any of those to show several together -- events
        from multiple timelines are merged into one chronological sequence,
        with the circuit panel showing each circuit's most recently known
        state side by side. Both frontends let you pin two points in that
        playback (possibly the same circuit at two different moments, or two
        different circuits shown together) and see everything that differs
        between them, computed the same way as :meth:`diff`.

        Auto-detects a native GUI window or a text-based CLI based on the
        running environment. Force a specific one with ``frontend="gui"|"cli"``,
        or set the ``QISDEBUG_FRONTEND`` environment variable to always use the
        same one. Forcing a frontend that isn't actually available (e.g.
        ``"gui"`` with no display) raises ``RuntimeError``.
        """
        return get_frontend(frontend).show(self._resolve_timelines(target))

    def diff(
        self,
        a: TimelineLike,
        b: TimelineLike,
        *,
        seq_a: int | None = None,
        seq_b: int | None = None,
        atol: float = 1e-6,
    ) -> TimelineDiff:
        """Compare two points in time: gate-sequence alignment plus state deltas.

        Thin wrapper around :func:`~qisdebug._analysis.diff_timelines`; ``a``
        and ``b`` take the same circuit/name/:class:`Timeline` shapes as
        :meth:`get`. ``seq_a``/``seq_b`` bound each side to "as of that
        event's seq" (from :attr:`~qisdebug.Timeline.events` or
        :meth:`~qisdebug._mcp.tools.get_events`); omitted, a side means its
        whole timeline / final state. Passing the same target for ``a`` and
        ``b`` with different seqs diffs two moments of the same circuit's
        replay. ``atol`` is the statevector comparison's per-amplitude
        mismatch threshold.
        """
        return diff_timelines(
            self.get(a), self.get(b), seq_a=seq_a, seq_b=seq_b, atol=atol
        )

    def export(self, target: TimelineTargets = None) -> list[dict]:
        """Export timelines as a list of JSON-serializable dicts.

        Takes the same ``target`` shapes as :meth:`show`: omitted for every
        timeline this instance has recorded, a single circuit/name/
        :class:`Timeline` for just that one, or a list of any of those for
        several. Always returns a list, even for a single ``target``, so
        callers don't need to branch on the shape of the result.
        """
        return [timeline.export() for timeline in self._resolve_timelines(target)]

    def clear(self) -> None:
        """Drop all recorded telemetry without touching active patches."""
        self.store.clear()

    def configure(self, **overrides: object) -> QisDebugConfig:
        """Update this instance's configuration and return the resulting config.

        Unknown keys or out-of-range values raise a pydantic ValidationError,
        leaving the previous configuration in place.
        """
        self.config = QisDebugConfig(**{**self.config.model_dump(), **overrides})
        return self.config

    def __enter__(self) -> "QisDebug":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.end()

    def __repr__(self) -> str:
        return (
            f"QisDebug(active={self._active_state['active']}, "
            f"circuits={len(self.store.all_timelines())})"
        )
