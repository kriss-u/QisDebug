"""Version compatibility helpers and centralized patch bookkeeping.

Every hook resolves its patch targets through this module so that a missing
or renamed Qiskit internal degrades into a logged warning instead of an
ImportError or AttributeError at activation time.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("qisdebug")


def qiskit_version() -> str | None:
    """Return the installed Qiskit version string, or None if unavailable."""
    try:
        import qiskit

        return qiskit.__version__
    except Exception:
        return None


def resolve(module_path: str, attribute: str | None = None) -> Any | None:
    """Import ``module_path`` and optionally fetch ``attribute`` from it.

    Returns None instead of raising when the module or attribute is missing,
    so callers can treat absent Qiskit internals as a degraded layer rather
    than a fatal error.
    """
    try:
        module = importlib.import_module(module_path)
    except ImportError:
        return None
    if attribute is None:
        return module
    return getattr(module, attribute, None)


class AppliedPatch(BaseModel):
    """Record of one replaced attribute, kept so deactivation can undo it."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    owner: Any
    attribute: str
    original: Any
    defined_locally: bool


class PatchRegistry(BaseModel):
    """Applies monkey patches and restores all of them on request.

    Restoration distinguishes attributes that were defined on the patched
    class itself from attributes inherited from a base class: the former are
    reassigned, the latter are deleted so the inherited version becomes
    visible again.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    patches: list[AppliedPatch] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def warn(self, message: str) -> None:
        """Record a degradation warning and emit it through the logger."""
        self.warnings.append(message)
        logger.warning("qisdebug: %s", message)

    def patch(self, owner: Any, attribute: str, factory: Callable[[Callable], Callable]) -> bool:
        """Replace ``owner.attribute`` with ``factory(original)``.

        Refuses to patch, and records a warning, when the attribute does not
        exist. This is the graceful-degradation path for Qiskit versions
        whose internals have moved.
        """
        if owner is None:
            return False
        original = getattr(owner, attribute, None)
        if original is None:
            owner_name = getattr(owner, "__name__", owner)
            self.warn(f"cannot patch {owner_name}.{attribute}: attribute not found")
            return False
        defined_locally = attribute in vars(owner)
        setattr(owner, attribute, factory(original))
        self.patches.append(
            AppliedPatch(
                owner=owner,
                attribute=attribute,
                original=original,
                defined_locally=defined_locally,
            )
        )
        return True

    def restore_all(self) -> None:
        """Undo every applied patch, newest first."""
        while self.patches:
            applied = self.patches.pop()
            try:
                if applied.defined_locally:
                    setattr(applied.owner, applied.attribute, applied.original)
                else:
                    delattr(applied.owner, applied.attribute)
            except Exception:
                logger.warning(
                    "qisdebug: failed to restore %s.%s",
                    getattr(applied.owner, "__name__", applied.owner),
                    applied.attribute,
                )
