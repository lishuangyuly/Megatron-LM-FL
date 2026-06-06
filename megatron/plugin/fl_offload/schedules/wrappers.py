"""Megatron schedule wrappers (commit 6).

Hosts:

* :func:`wrap_schedule_for_offload` — wraps one
  ``forward_backward_*`` function; while it runs, the module attributes
  ``forward_step`` and ``backward_step`` on
  ``megatron.core.pipeline_parallel.schedules`` are temporarily swapped
  for offload-aware shims.
* :func:`_patch_step_funcs` — the underlying context manager.

Both shims also surround the call with ``nvtx_range_push("fl_offload/...")``
so a nsys profile can identify plugin regions without grepping the
trace.

Backward path: FL's ``backward_step`` does **not** receive
``current_microbatch`` / ``vp_stage``, but the surrounding schedule
hands it the same ``output_tensor`` object that ``forward_step``
returned.  We therefore record ``id(output_tensor) -> key`` in a
thread-local dict on forward and look it up on backward.

Commit 6 wires only the interleaved 1F1B path; commit 7 extends to
no_pipelining + without_interleaving.
"""

from __future__ import annotations

import contextlib
import functools
import threading
from typing import Any, Callable, Dict, Optional

from megatron.plugin.fl_offload.runtime import get_pipeline_offload_runtime
from megatron.plugin.fl_offload.schedules.keys import make_offload_key_interleaved


_PHASE_WARMUP = "warmup"
_PHASE_STEADY = "steady"
_PHASE_COOLDOWN = "cooldown"


# Thread-local registry mapping id(output_tensor_from_forward) -> offload_key
# so the patched backward_step can recover the matching key.  The schedule
# guarantees backward consumes each output_tensor at most once; we pop on
# read so stale ids cannot accumulate.
class _KeyRegistry(threading.local):
    def __init__(self) -> None:
        self.by_output_id: Dict[int, Any] = {}


_REGISTRY = _KeyRegistry()


def _nvtx_push(name: str) -> None:
    try:
        import torch

        torch.cuda.nvtx.range_push(name)
    except Exception:
        pass


def _nvtx_pop() -> None:
    try:
        import torch

        torch.cuda.nvtx.range_pop()
    except Exception:
        pass


def _is_no_offload_boundary(model_chunk_id: int) -> bool:
    """Last PP rank × last model chunk → forward feeds backward immediately.

    Offload there would just churn D2H/H2D without freeing peak VRAM.
    The runtime falls back to ``nullcontext`` so the schedule is
    bit-exact against the no-plugin baseline on that boundary.
    """
    from megatron.core import parallel_state as ps

    pp_rank = ps.get_pipeline_model_parallel_rank()
    pp = ps.get_pipeline_model_parallel_world_size()
    num_chunks = ps.get_virtual_pipeline_model_parallel_world_size() or 1
    return pp_rank == pp - 1 and model_chunk_id == num_chunks - 1


def _make_offload_forward_step(orig_forward_step: Callable) -> Callable:
    """Build the patched forward_step that wraps the original in offload."""

    runtime = get_pipeline_offload_runtime()

    @functools.wraps(orig_forward_step)
    def offload_forward_step(*args, **kwargs):
        current_microbatch = kwargs.get("current_microbatch")
        vp_stage = kwargs.get("vp_stage")
        mc_id = vp_stage if vp_stage is not None else 0
        vmb = current_microbatch if current_microbatch is not None else 0
        try:
            key = make_offload_key_interleaved(vmb, mc_id, forward=True)
        except Exception:
            # parallel_state not initialised in this call context — fall
            # back to a key derived purely from the call args.
            key = ("vmb", vmb, "mc", mc_id)
        enabled = runtime.enabled() and not _is_no_offload_boundary(mc_id)

        _nvtx_push("fl_offload/forward")
        try:
            with runtime.forward_microbatch(
                phase=_PHASE_STEADY,
                virtual_microbatch_id=vmb,
                model_chunk_id=mc_id,
                enabled=enabled,
                offload_key=key,
            ):
                result = orig_forward_step(*args, **kwargs)
        finally:
            _nvtx_pop()

        # FL returns (output_tensor, num_tokens); we need the tensor
        # (or whatever object) so backward can recover the key by id().
        if enabled and isinstance(result, tuple) and len(result) >= 1:
            output_tensor = result[0]
            if output_tensor is not None:
                _REGISTRY.by_output_id[id(output_tensor)] = key
        return result

    return offload_forward_step


def _extract_output_tensor(args, kwargs) -> Any:
    """Pull ``output_tensor`` out of FL ``backward_step`` call args.

    Signature: ``backward_step(input_tensor, output_tensor,
    output_tensor_grad, config)``.
    """
    if len(args) >= 2:
        return args[1]
    return kwargs.get("output_tensor")


def _make_offload_backward_step(orig_backward_step: Callable) -> Callable:
    """Build the patched backward_step that prefetches onload."""

    runtime = get_pipeline_offload_runtime()

    @functools.wraps(orig_backward_step)
    def offload_backward_step(*args, **kwargs):
        output_tensor = _extract_output_tensor(args, kwargs)
        key = _REGISTRY.by_output_id.pop(id(output_tensor), None) if output_tensor is not None else None
        _nvtx_push("fl_offload/backward")
        try:
            if key is None or not runtime.enabled():
                # No prefetch — unpack hooks will fire on demand (slower
                # but still correct because onload restores tw.x).
                return orig_backward_step(*args, **kwargs)
            with runtime.backward_microbatch(
                phase=_PHASE_STEADY,
                virtual_microbatch_id=0,  # advisory
                model_chunk_id=0,         # advisory
                enabled=True,
                offload_key=key,
            ):
                return orig_backward_step(*args, **kwargs)
        finally:
            _nvtx_pop()

    return offload_backward_step


@contextlib.contextmanager
def _patch_step_funcs():
    """Swap ``core_sched.forward_step`` / ``backward_step`` for the duration of the ``with``."""
    from unittest.mock import patch

    from megatron.core.pipeline_parallel import schedules as core_sched

    fwd = _make_offload_forward_step(core_sched.forward_step)
    bwd = _make_offload_backward_step(core_sched.backward_step)
    with patch.object(core_sched, "forward_step", fwd), patch.object(
        core_sched, "backward_step", bwd
    ):
        yield


def wrap_schedule_for_offload(fn: Callable) -> Callable:
    """Return ``fn`` with step funcs patched for the duration of each call.

    On the way out, regardless of success or failure, the wrapper bumps
    the observability step counter so report cadence stays aligned with
    the training loop's calls to ``forward_backward_func``.
    """

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            with _patch_step_funcs():
                return fn(*args, **kwargs)
        finally:
            from megatron.plugin.fl_offload.observability import report_after_step

            report_after_step()

    return wrapped


def _reset_registry_for_tests() -> None:
    """Drop any pending (id, key) entries.  Tests only."""
    _REGISTRY.by_output_id.clear()


__all__ = [
    "_patch_step_funcs",
    "_reset_registry_for_tests",
    "wrap_schedule_for_offload",
]
