"""Environment-detecting timeline frontends: native GUI or CLI.

:func:`get_frontend` is the factory: it auto-detects the running environment
(display available, or not) and returns the matching :class:`TimelineFrontend`,
or honors an explicit override via the ``frontend`` argument or the
``QDEBUG_FRONTEND`` environment variable.
"""

from __future__ import annotations

from .base import TimelineFrontend
from .detect import ENV_VAR, detect_environment, get_frontend

__all__ = ["TimelineFrontend", "ENV_VAR", "detect_environment", "get_frontend"]
