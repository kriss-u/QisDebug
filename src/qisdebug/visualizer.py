"""Timeline rendering utilities, decoupled from instrumentation.

Displaying a timeline itself happens through :meth:`~qisdebug.QisDebug.show`;
this module only exposes the frontend-agnostic formatting helpers from
:mod:`qisdebug._frontends._render` for callers that want to format a snapshot
or build a histogram/Bloch-sphere image without opening a full viewer.
"""

from __future__ import annotations

from ._frontends._render import format_statevector, render_bloch, render_histogram

__all__ = ["format_statevector", "render_histogram", "render_bloch"]
