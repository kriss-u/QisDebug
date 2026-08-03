"""Common interface every timeline frontend implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from .._events import Timeline


class TimelineFrontend(ABC):
    """Strategy interface for presenting a timeline to the user.

    Each concrete frontend (Jupyter widget, native GUI window, text-based
    CLI) overrides the same three methods, so :func:`get_frontend` can pick
    one without callers needing to know which environment they ended up in.
    """

    name: ClassVar[str]

    @abstractmethod
    def is_available(self) -> bool:
        """Return whether this frontend can actually run right now."""

    @abstractmethod
    def unavailable_reason(self) -> str:
        """Return a human-readable reason this frontend is not available.

        Only meaningful when :meth:`is_available` returns False.
        """

    @abstractmethod
    def show(self, timelines: list[Timeline]) -> Any:
        """Present one or more timelines. Blocking for interactive frontends."""
