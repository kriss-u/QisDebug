"""Process-wide, refcounted monkeypatch table shared by every QisDebug instance.

Qiskit classes are patched at most once per ``(owner, attribute)`` pair no
matter how many :class:`~qisdebug.QisDebug` instances are started
concurrently -- the first :func:`ensure_patched` call installs the real
wrapper, later calls for the same target just bump a refcount, and
:func:`release_patched` only restores the original once the refcount drops
back to zero. That keeps two instances patching the same Qiskit class from
racing each other's restore, which a naive per-instance patch/unpatch would.

Each installed wrapper looks up the currently started instances via
:func:`active_instances` and fans the event it just observed out to each
one's own store, using that instance's own suspension state and
configuration.
"""

from __future__ import annotations

import threading
import weakref
from contextlib import ExitStack, contextmanager
from typing import TYPE_CHECKING, Any, Callable, Iterator, NamedTuple

from ._compat import logger

if TYPE_CHECKING:
    from ._debug import QisDebug

_lock = threading.RLock()
_refcounts: dict[tuple[Any, str], int] = {}
_originals: dict[tuple[Any, str], tuple[Any, bool]] = {}
# A WeakSet, not a plain list: a QisDebug instance the caller drops without
# calling end() must still become collectible (so its GC warning fires)
# rather than being kept alive forever just for being "started".
_active: "weakref.WeakSet[QisDebug]" = weakref.WeakSet()


class InstallResult(NamedTuple):
    """What one hook layer's ``install()`` accomplished."""

    applied: bool
    warnings: list[str]
    targets: list[tuple[Any, str]]


def ensure_patched(
    owner: Any, attribute: str, factory: Callable[[Callable], Callable]
) -> tuple[bool, str | None]:
    """Install the shared patch for ``owner.attribute`` if not already active.

    Returns ``(True, None)`` on success (whether newly installed or already
    active for another instance), or ``(False, warning)`` when the attribute
    doesn't exist -- the graceful-degradation path for Qiskit versions whose
    internals have moved.
    """
    if owner is None:
        return False, None
    key = (owner, attribute)
    with _lock:
        if key in _refcounts:
            _refcounts[key] += 1
            return True, None
        original = getattr(owner, attribute, None)
        if original is None:
            owner_name = getattr(owner, "__name__", owner)
            message = f"cannot patch {owner_name}.{attribute}: attribute not found"
            logger.warning("qisdebug: %s", message)
            return False, message
        defined_locally = attribute in vars(owner)
        setattr(owner, attribute, factory(original))
        _refcounts[key] = 1
        _originals[key] = (original, defined_locally)
        return True, None


def patch_and_track(
    owner: Any,
    attribute: str,
    factory: Callable[[Callable], Callable],
    targets: list[tuple[Any, str]],
    warnings: list[str],
) -> bool:
    """Patch ``owner.attribute`` via :func:`ensure_patched`, recording the
    outcome into the caller's ``targets``/``warnings`` accumulators.

    Every hook's ``install()`` repeats the same ``ok, warning =
    ensure_patched(...); if ok: targets.append(...); if warning:
    warnings.append(...)`` idiom around each patch site; this collapses it to
    one call. Returns ``ok`` so callers can still fold it into their own
    ``applied`` flag.
    """
    ok, warning = ensure_patched(owner, attribute, factory)
    if ok:
        targets.append((owner, attribute))
    if warning:
        warnings.append(warning)
    return ok


def release_patched(owner: Any, attribute: str) -> None:
    """Undo one :func:`ensure_patched` call; restores the original at refcount 0."""
    if owner is None:
        return
    key = (owner, attribute)
    with _lock:
        if key not in _refcounts:
            return
        _refcounts[key] -= 1
        if _refcounts[key] > 0:
            return
        del _refcounts[key]
        original, defined_locally = _originals.pop(key)
        try:
            if defined_locally:
                setattr(owner, attribute, original)
            else:
                delattr(owner, attribute)
        except Exception:
            logger.warning(
                "qisdebug: failed to restore %s.%s",
                getattr(owner, "__name__", owner),
                attribute,
            )


def register(debug: "QisDebug") -> None:
    """Mark ``debug`` as started, so hook wrappers fan events out to it."""
    with _lock:
        _active.add(debug)


def unregister(debug: "QisDebug") -> None:
    """Mark ``debug`` as ended; hook wrappers stop fanning events out to it."""
    with _lock:
        _active.discard(debug)


def active_instances() -> list["QisDebug"]:
    """Return a snapshot of every currently started QisDebug instance."""
    with _lock:
        return list(_active)


@contextmanager
def all_suspended(listeners: list["QisDebug"]) -> Iterator[None]:
    """Suspend recording on every listener's store for the duration of the block."""
    with ExitStack() as stack:
        for debug in listeners:
            stack.enter_context(debug.store.suspended())
        yield
