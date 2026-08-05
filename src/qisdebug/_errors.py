"""Public exception hierarchy for QisDebug."""

from __future__ import annotations


class QisDebugError(Exception):
    """Base class for every exception QisDebug raises deliberately."""


class TimelineNotFoundError(QisDebugError, LookupError):
    """Raised by :meth:`QisDebug.get` when the target isn't tracked."""
