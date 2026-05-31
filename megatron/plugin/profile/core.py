"""
Core utilities for pipeline-semantic chrome trace instrumentation.

See profile_docs/design.md for the design and profile_docs/plan.md for the
phased rollout. This module is loaded only when the user opts in via
``--profile-pp-semantics``; otherwise every helper short-circuits to a
no-op so there is no runtime overhead on the non-profile path.
"""

from contextlib import contextmanager
from functools import wraps
from typing import Callable

import torch
from torch.profiler import record_function


# Sentinel prefix that lets downstream parsing scripts find our annotations
# among the many aten / Python frame events in a chrome trace.
PREFIX = "mcfl:"


_profile_enabled = False


def set_profile_enabled(enabled: bool) -> None:
    """Globally enable or disable the mcfl: instrumentation."""
    global _profile_enabled
    _profile_enabled = bool(enabled)


def is_profile_enabled() -> bool:
    return _profile_enabled


def make_name(**kwargs) -> str:
    """Build a record_function name in the format ``"mcfl: k1=v1&k2=v2&..."``.

    Force callers to go through this factory so name strings stay parseable
    by downstream visualization scripts (see design.md §3).
    """
    parts = "&".join(f"{k}={v}" for k, v in kwargs.items())
    return f"{PREFIX} {parts}" if parts else PREFIX


def _current_device_name() -> str:
    """Return a device string usable by ``torch.empty(device=...)``.

    Goes through ``cur_platform`` so vendors (cuda / musa / txda / cpu) all
    pick the right device without hard-coding ``cuda``.
    """
    try:
        from megatron.plugin.platform import get_platform
        return get_platform().current_device_name()
    except Exception:
        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
        return "cpu"


@contextmanager
def gpu_anchor():
    """Pin a CPU record_function range to a GPU stream window.

    Inserts a tiny ``torch.empty(1).fill_(0)`` at scope enter/exit. See
    design.md §2.2 for the rationale (CPU/GPU timeline alignment).
    """
    if not _profile_enabled:
        yield
        return
    dev = _current_device_name()
    t = torch.empty(1, device=dev)
    t.fill_(0)
    try:
        yield
    finally:
        t.fill_(0)


def profile_it(name: str):
    """Decorator: wrap a function in ``record_function(name)``.

    When disabled the wrapper still exists but short-circuits before
    creating the record_function context manager. The wrapper-call
    overhead is one bool check + one Python call frame, which is
    negligible for the per-microbatch call sites this is applied to.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not _profile_enabled:
                return func(*args, **kwargs)
            with record_function(name):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def profile_it_with_anchor(name: str):
    """Same as ``profile_it`` but also runs ``gpu_anchor`` around the call.

    Use for "Python-heavy / comm-heavy" ranges (e.g. P2P callbacks) where
    the body itself has little GPU work and the CPU record_function range
    would otherwise not align with any GPU stream window.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not _profile_enabled:
                return func(*args, **kwargs)
            with record_function(name), gpu_anchor():
                return func(*args, **kwargs)

        return wrapper

    return decorator


@contextmanager
def step_record_function(**kwargs):
    """Outermost step-level record_function for design.md §4.1.

    Builds a name from kwargs via ``make_name`` and pins the scope to the
    current GPU stream via ``gpu_anchor``. Use as::

        with step_record_function(
            func="forward_backward_step",
            f_virtual_microbatch_id=...,
            b_virtual_microbatch_id=...,
            ...
        ):
            combined_forward_backward_step(...)
    """
    if not _profile_enabled:
        yield
        return
    name = make_name(**kwargs)
    with record_function(name), gpu_anchor():
        yield
