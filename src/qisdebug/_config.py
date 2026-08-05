"""Runtime configuration for QisDebug, validated through Pydantic.

Each :class:`~qisdebug.QisDebug` instance owns its own :class:`QisDebugConfig`
via ``QisDebug(config=...)``/``debug.configure(...)`` -- there is no
process-wide default, so independent instances can't affect each other's
capture behavior.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class QisDebugConfig(BaseModel):
    """Tunable limits and feature switches for the instrumentation layers."""

    model_config = ConfigDict(frozen=True)

    max_circuits: int = Field(
        default=256, ge=1,
        description="Maximum number of circuits tracked before the oldest timelines are evicted.",
    )
    capture_statevectors: bool = Field(
        default=True,
        description="Whether the execution hook inserts statevector snapshots on Aer backends.",
    )
    max_statevector_qubits: int = Field(
        default=12, ge=1, le=30,
        description="Circuits wider than this skip statevector capture, since snapshot memory grows as 2**num_qubits.",
    )
    max_snapshots_per_circuit: int = Field(
        default=256, ge=1,
        description="Snapshot insertion stops after this many snapshots for a single circuit.",
    )
    transpile_snapshots: Literal["all", "changed", "none"] = Field(
        default="changed",
        description="Which transpiler passes get a full circuit snapshot attached: every pass, only passes that changed the circuit, or none.",
    )
    capture_backend_compilation: bool = Field(
        default=True,
        description=(
            "Whether pass manager runs a backend performs internally while executing a circuit "
            "(e.g. Aer lowering control flow) are recorded as phase=\"backend\" transpile events. "
            "When off they are suppressed with a note event instead."
        ),
    )
