"""Public activation lifecycle for QDebug's instrumentation layers."""

from __future__ import annotations

import threading

from pydantic import BaseModel, Field

from ._compat import PatchRegistry, logger, qiskit_version
from ._hooks import construction, execution, transpile

_lock = threading.Lock()
_registry: PatchRegistry | None = None
_report: ActivationReport | None = None


class ActivationReport(BaseModel):
    """Summary of which instrumentation layers attached at activation."""

    active: bool
    qiskit_version: str | None = None
    layers: dict[str, bool] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


def activate() -> ActivationReport:
    """Attach all instrumentation layers to Qiskit.

    Idempotent: calling it again while active returns the existing report
    without stacking a second set of patches. Layers whose Qiskit internals
    are missing are skipped with a logged warning rather than raising, and
    the returned report says which layers actually attached.
    """
    global _registry, _report
    with _lock:
        if _registry is not None and _report is not None:
            return _report
        registry = PatchRegistry()
        layers = {
            "construction": construction.install(registry),
            "execution": execution.install(registry),
            "transpile": transpile.install(registry),
        }
        _registry = registry
        _report = ActivationReport(
            active=True,
            qiskit_version=qiskit_version(),
            layers=layers,
            warnings=list(registry.warnings),
        )
        if not any(layers.values()):
            logger.warning("qdebug: activation attached no instrumentation layers")
        else:
            attached = ", ".join(name for name, ok in layers.items() if ok)
            logger.info("qdebug: active (layers: %s)", attached)
        return _report


def deactivate() -> None:
    """Restore every patched Qiskit method. Safe to call when not active."""
    global _registry, _report
    with _lock:
        if _registry is None:
            return
        _registry.restore_all()
        _registry = None
        _report = None


def is_active() -> bool:
    """Return True while QDebug's patches are applied."""
    return _registry is not None
