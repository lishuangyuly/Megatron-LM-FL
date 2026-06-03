"""``get_forward_backward_func`` wrapper / selector (commit 6).

Hosts :func:`get_forward_backward_func_wrapper(original)` registered by
:func:`megatron.plugin.fl_offload.apply.apply` as a monkey-patch on
``megatron.core.pipeline_parallel.schedules.get_forward_backward_func``.

Behaviour:

* ``enable=False`` → returns the original FL callable unchanged
  (identity check passes in tests).
* ``enable=True`` and selected schedule is
  ``forward_backward_pipelining_with_interleaving`` →
  ``wrap_schedule_for_offload(chosen)``.
* ``enable=True`` and selected schedule is anything else →
  ``NotImplementedError("... Commit 7")``.
* Detects dualpipev (``get_dualpipev_pipeline_model_parallel_world_size``
  is set) and refuses to run with a clear error.
"""

from __future__ import annotations

import functools
from typing import Callable

from megatron.plugin.fl_offload.config import get_config


def get_forward_backward_func_wrapper(original: Callable) -> Callable:
    """Return a drop-in replacement for ``get_forward_backward_func``.

    The replacement defers the schedule-selection decision until call
    time so that argparse / config can flip ``enable`` at any point
    before training starts.
    """

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        chosen = original(*args, **kwargs)
        cfg = get_config()
        if not cfg.enable:
            return chosen

        _assert_no_dualpipev()

        # Lazy import: schedules.py only loads when the user actually
        # opts into pipeline parallel, and we want to keep the plugin
        # importable on CPU-only smoke tests.
        from megatron.core.pipeline_parallel import schedules as core_sched

        if chosen is core_sched.forward_backward_pipelining_with_interleaving:
            from megatron.plugin.fl_offload.schedules.wrappers import (
                wrap_schedule_for_offload,
            )

            return wrap_schedule_for_offload(chosen)

        raise NotImplementedError(
            "fl-offload: non-interleaved / no_pipelining schedule wrappers "
            "wired in Commit 7. Selected schedule: "
            f"{getattr(chosen, '__name__', repr(chosen))}"
        )

    return wrapped


def _assert_no_dualpipev() -> None:
    """Refuse to run with dualpipev (mutex enforced in validate + here)."""
    try:
        from megatron.core import parallel_state as ps
    except Exception:
        return
    getter = getattr(ps, "get_dualpipev_pipeline_model_parallel_world_size", None)
    if getter is None:
        return
    try:
        size = getter()
    except Exception:
        size = None
    if size is not None:
        raise AssertionError(
            "fl-offload is incompatible with dualpipev pipeline model parallel"
        )


__all__ = ["get_forward_backward_func_wrapper"]
