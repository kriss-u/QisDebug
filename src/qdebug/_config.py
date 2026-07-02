"""Runtime configuration for QDebug, validated through Pydantic."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class QDebugConfig(BaseModel):
    """Tunable limits and feature switches for the instrumentation layers.

    Attributes:
        max_circuits: Maximum number of circuits tracked before the oldest
            timelines are evicted.
        capture_statevectors: Whether the execution hook inserts statevector
            snapshots on Aer backends.
        max_statevector_qubits: Circuits wider than this skip statevector
            capture, since snapshot memory grows as ``2**num_qubits``.
        max_snapshots_per_circuit: Snapshot insertion stops after this many
            snapshots for a single circuit.
        transpile_snapshots: Which transpiler passes get a full circuit
            snapshot attached: every pass, only passes that changed the
            circuit, or none.
    """

    model_config = ConfigDict(frozen=True)

    max_circuits: int = Field(default=256, ge=1)
    capture_statevectors: bool = True
    max_statevector_qubits: int = Field(default=12, ge=1, le=30)
    max_snapshots_per_circuit: int = Field(default=256, ge=1)
    transpile_snapshots: Literal["all", "changed", "none"] = "changed"


_config = QDebugConfig()


def configure(**overrides: object) -> QDebugConfig:
    """Update the global configuration and return the resulting config.

    Unknown keys or out-of-range values raise a pydantic ValidationError,
    leaving the previous configuration in place.
    """
    global _config
    _config = QDebugConfig(**{**_config.model_dump(), **overrides})
    return _config


def get_config() -> QDebugConfig:
    """Return the current global configuration."""
    return _config
