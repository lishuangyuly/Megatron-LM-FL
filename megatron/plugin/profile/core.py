"""Pipeline-semantic chrome trace instrumentation utilities."""

from contextlib import contextmanager
from functools import wraps
from typing import Callable

import torch
from torch.profiler import record_function


PREFIX = "mcfl:"

_profile_enabled = False


def set_profile_enabled(enabled: bool) -> None:
    """Globally enable or disable semantic trace annotations."""
    global _profile_enabled
    _profile_enabled = bool(enabled)


def is_profile_enabled() -> bool:
    return _profile_enabled


def make_name(**kwargs) -> str:
    """Build a parseable ``mcfl: key=value&...`` event name."""
    parts = "&".join(f"{key}={value}" for key, value in kwargs.items())
    return f"{PREFIX} {parts}" if parts else PREFIX


def _current_device_name() -> str:
    try:
        from megatron.plugin.platform import get_platform

        return get_platform().current_device_name()
    except Exception:
        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
        return "cpu"


@contextmanager
def gpu_anchor():
    """Pin a CPU record_function range to the current GPU stream."""
    if not _profile_enabled:
        yield
        return
    tensor = torch.empty(1, device=_current_device_name())
    tensor.fill_(0)
    try:
        yield
    finally:
        tensor.fill_(0)


@contextmanager
def semantic_record(**kwargs):
    """Emit a parseable semantic range without adding GPU anchor kernels."""
    if not _profile_enabled:
        yield
        return
    with record_function(make_name(**kwargs)):
        yield


def profile_it(name: str):
    """Decorate a callable with a record_function range when enabled."""

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
    """Decorate a callable with a range and current-stream GPU anchors."""

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
    """Emit the outer combined forward/backward scheduling range."""
    if not _profile_enabled:
        yield
        return
    with record_function(make_name(**kwargs)), gpu_anchor():
        yield
