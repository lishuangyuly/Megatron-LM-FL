"""Dedicated copy streams (D2H + H2D) backed by ``cur_platform``.

Two stream slots are exposed:

* ``"offload"`` — used by D2H copies in
  :meth:`~megatron.plugin.fl_offload.group.ActivationGroup.offload_issue`;
* ``"onload"`` — used by H2D copies in
  :meth:`onload_issue`.

Both are lazy-initialised on first use and re-used for the rest of the
process lifetime.  Tests can call :func:`reset_streams` between cases to
discard them.

When the current platform reports it does not have an accelerator
available, :func:`get_memcpy_stream` returns ``None``.  Callers must
treat ``None`` as "no async stream available, do the copy synchronously
on the calling stream" and skip ``wait_stream`` plumbing.  This keeps
unit tests usable on CPU-only boxes.
"""

from typing import Dict, Optional


_VALID_KEYS = ("offload", "onload")


def _platform_available() -> bool:
    try:
        from megatron.plugin.platform import get_platform

        return get_platform().is_available()
    except Exception:
        return False


_STREAMS: Dict[str, "object"] = {}


def get_memcpy_stream(key: str) -> Optional["object"]:
    """Return (and lazily create) the dedicated stream for ``key``.

    Returns ``None`` if the platform has no accelerator — the caller must
    fall back to synchronous copies in that case.
    """
    if key not in _VALID_KEYS:
        raise ValueError(
            f"get_memcpy_stream: unsupported key {key!r}; "
            f"expected one of {_VALID_KEYS}"
        )

    if key in _STREAMS:
        return _STREAMS[key]

    if not _platform_available():
        _STREAMS[key] = None
        return None

    from megatron.plugin.platform import get_platform

    stream = get_platform().Stream()
    _STREAMS[key] = stream
    return stream


def current_stream():
    """Return the current accelerator stream, or ``None`` on CPU-only."""
    if not _platform_available():
        return None
    from megatron.plugin.platform import get_platform

    return get_platform().current_stream()


def stream_scope(stream):
    """Context manager that switches the default stream to ``stream``.

    On CPU-only platforms (``stream is None``) returns a no-op context.
    """
    if stream is None:

        class _NullCtx:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        return _NullCtx()

    from megatron.plugin.platform import get_platform

    return get_platform().stream(stream)


def has_async_streams() -> bool:
    """True iff dedicated copy streams will actually be used."""
    return _platform_available()


def reset_streams() -> None:
    """Drop cached streams. Tests only."""
    _STREAMS.clear()


__all__ = [
    "get_memcpy_stream",
    "current_stream",
    "stream_scope",
    "has_async_streams",
    "reset_streams",
]
