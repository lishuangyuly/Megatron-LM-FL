"""``get_forward_backward_func`` wrapper / selector (commit 6).

Hosts :func:`get_forward_backward_func_wrapper(original)` registered by
:func:`megatron.plugin.fl_offload.apply.apply` as a monkey-patch on
``megatron.core.pipeline_parallel.schedules.get_forward_backward_func``.

Behaviour:

* If ``set_config`` has not run yet, the first call lands the config
  from megatron's global args via ``validate_plugin_args`` (FL's
  ``pretrain()`` has no validate hook, so this is the only reliable
  point where parsed args exist).
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

from megatron.plugin.fl_offload.config import config_landed, get_config


def _land_config_from_global_args() -> None:
    """Land the plugin config from megatron's parsed global args.

    FL's ``pretrain()`` exposes no ``validate_args`` hook the plugin
    could chain into, so ``validate_plugin_args`` (which installs the
    config via ``set_config``) would otherwise never run in a real
    training job — ``enable`` would silently stay ``False``.  The first
    schedule selection lands it here instead.

    Outside a megatron training context (CPU unit tests, args not yet
    initialized) this is a silent no-op; validation errors on real args
    (mutex / cuda-graph guard) propagate so the job fails fast.
    """
    try:
        from megatron.training.global_vars import get_args

        args = get_args()
    except Exception:
        return
    if args is None or not hasattr(args, "fl_offload_enable"):
        # Entry script never called apply(); nothing to land.
        return

    from megatron.plugin.fl_offload.validate import validate_plugin_args

    validate_plugin_args(args)


def get_forward_backward_func_wrapper(original: Callable) -> Callable:
    """Return a drop-in replacement for ``get_forward_backward_func``.

    The replacement defers the schedule-selection decision until call
    time so that argparse / config can flip ``enable`` at any point
    before training starts.
    """

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        chosen = original(*args, **kwargs)
        if not config_landed():
            _land_config_from_global_args()
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
