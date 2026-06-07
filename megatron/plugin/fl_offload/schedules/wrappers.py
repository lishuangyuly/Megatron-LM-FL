"""Megatron schedule wrappers (commit 6 + combined-path support).

Hosts:

* :func:`wrap_schedule_for_offload` — wraps one
  ``forward_backward_*`` function; while it runs, three module
  attributes on ``megatron.core.pipeline_parallel.schedules`` are
  temporarily swapped for offload-aware shims:

  - ``forward_step`` / ``backward_step`` — the conventional
    interleaved path (``overlap_moe_expert_parallel_comm`` off, plus
    forward-only/eval calls);
  - ``combined_1f1b_schedule_for_interleaved_pipelining`` — the
    combined path taken when ``overlap_moe_expert_parallel_comm`` is
    on.  That path never calls ``forward_step``/``backward_step`` (it
    drives the model through fine-grained ScheduleNodes inside
    ``combined_forward_backward_step``), so it needs its own shim.

* :func:`_patch_step_funcs` — the underlying context manager.

Key derivation:

* conventional path — forward has ``current_microbatch`` (microbatch
  id within the model chunk) + ``vp_stage``; backward has neither, so
  the forward records ``id(output_tensor) -> key`` in a thread-local
  registry and backward pops it.
* combined path — the schedule helper receives both virtual ids plus
  FL's own schedule-table lookups, so forward and backward keys are
  derived independently; ``make_offload_key_interleaved`` returns the
  logical ``(microbatch_id, model_chunk_id)`` pair that both sides
  agree on.

Ordering on the combined path: forward f and backward b are
interleaved at the sub-operator level inside
``combined_forward_backward_step``, so b's activations are onloaded
synchronously *before* the helper runs, and f's record() context spans
the whole helper call (backward execution does not pack new tensors,
so collecting under f's key is safe).
"""

from __future__ import annotations

import contextlib
import functools
import inspect
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


# ---------------------------------------------------------------------- #
# Conventional interleaved path (forward_step / backward_step)           #
# ---------------------------------------------------------------------- #
def _make_offload_forward_step(orig_forward_step: Callable) -> Callable:
    """Build the patched forward_step that wraps the original in offload."""

    runtime = get_pipeline_offload_runtime()

    @functools.wraps(orig_forward_step)
    def offload_forward_step(*args, **kwargs):
        current_microbatch = kwargs.get("current_microbatch")
        vp_stage = kwargs.get("vp_stage")
        mc_id = vp_stage if vp_stage is not None else 0
        mb_id = current_microbatch if current_microbatch is not None else 0
        key = make_offload_key_interleaved(mb_id, mc_id, forward=True)
        enabled = runtime.enabled() and not _is_no_offload_boundary(mc_id)

        _nvtx_push("fl_offload/forward")
        try:
            with runtime.forward_microbatch(
                phase=_PHASE_STEADY,
                virtual_microbatch_id=mb_id,
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


# ---------------------------------------------------------------------- #
# Combined path (overlap_moe_expert_parallel_comm)                       #
# ---------------------------------------------------------------------- #
def _make_offload_combined_helper(orig_combined_helper: Callable) -> Callable:
    """Wrap ``combined_1f1b_schedule_for_interleaved_pipelining``.

    The helper merges one forward microbatch (``f_virtual_microbatch_id``)
    and one backward microbatch (``b_virtual_microbatch_id``) — either may
    be ``None`` during warmup / cooldown.  We:

    1. onload b's activation group synchronously *before* the helper
       (forward and backward interleave at sub-op level inside it, so
       the tensors must already be resident);
    2. run the helper under f's ``record()`` context so f's saved
       activations are collected (backward execution packs nothing, so
       the shared context is safe);
    3. let the forward context's exit fire the offload of f's group.

    Keys come from FL's own schedule-table lookups, which are passed in
    as arguments: ``get_microbatch_id_in_model_chunk`` indexes a
    direction-agnostic table (its ``forward`` parameter only feeds an
    assert), and ``get_model_chunk_id(vmb, forward=False)`` mirrors the
    chunk — so forward and backward agree on the logical
    ``(microbatch_id, model_chunk_id)`` pair for the same microbatch.
    """

    runtime = get_pipeline_offload_runtime()
    try:
        sig = inspect.signature(orig_combined_helper)
    except (TypeError, ValueError):
        sig = None

    @functools.wraps(orig_combined_helper)
    def offload_combined_helper(*args, **kwargs):
        if not runtime.enabled() or sig is None:
            return orig_combined_helper(*args, **kwargs)

        try:
            bound = sig.bind(*args, **kwargs)
        except TypeError:
            return orig_combined_helper(*args, **kwargs)
        ba = bound.arguments
        get_mb_id = ba.get("get_microbatch_id_in_model_chunk")
        get_mc_id = ba.get("get_model_chunk_id")
        f_vmb = ba.get("f_virtual_microbatch_id")
        b_vmb = ba.get("b_virtual_microbatch_id")
        if get_mb_id is None or get_mc_id is None:
            return orig_combined_helper(*args, **kwargs)

        # Backward side first: synchronous onload before the helper runs.
        bwd_cm = None
        if b_vmb is not None:
            b_mc = get_mc_id(b_vmb, forward=False)
            # The microbatch-id table is direction-agnostic; the helper
            # merely asserts forward=True, so we query it as forward.
            b_mb = get_mb_id(b_vmb, forward=True)
            b_key = make_offload_key_interleaved(b_mb, b_mc, forward=False)
            bwd_cm = runtime.backward_microbatch(
                phase=_PHASE_STEADY,
                virtual_microbatch_id=b_vmb,
                model_chunk_id=b_mc,
                enabled=True,
                offload_key=b_key,
            )
            bwd_cm.__enter__()

        # Forward side: record under f's key for the helper's duration.
        fwd_cm = contextlib.nullcontext()
        if f_vmb is not None:
            f_mc = get_mc_id(f_vmb, forward=True)
            f_mb = get_mb_id(f_vmb, forward=True)
            f_key = make_offload_key_interleaved(f_mb, f_mc, forward=True)
            fwd_cm = runtime.forward_microbatch(
                phase=_PHASE_STEADY,
                virtual_microbatch_id=f_vmb,
                model_chunk_id=f_mc,
                enabled=not _is_no_offload_boundary(f_mc),
                offload_key=f_key,
            )

        _nvtx_push("fl_offload/combined")
        try:
            with fwd_cm:
                return orig_combined_helper(*args, **kwargs)
        finally:
            _nvtx_pop()
            if bwd_cm is not None:
                bwd_cm.__exit__(None, None, None)

    return offload_combined_helper


@contextlib.contextmanager
def _patch_step_funcs():
    """Swap the schedule's step entry points for the duration of the ``with``.

    Patches ``forward_step`` / ``backward_step`` (conventional path) and
    ``combined_1f1b_schedule_for_interleaved_pipelining`` (combined path,
    bound into ``schedules``'s module globals by its top-level import).
    """
    from unittest.mock import patch

    from megatron.core.pipeline_parallel import schedules as core_sched

    fwd = _make_offload_forward_step(core_sched.forward_step)
    bwd = _make_offload_backward_step(core_sched.backward_step)
    combined = _make_offload_combined_helper(
        core_sched.combined_1f1b_schedule_for_interleaved_pipelining
    )
    with patch.object(core_sched, "forward_step", fwd), patch.object(
        core_sched, "backward_step", bwd
    ), patch.object(
        core_sched,
        "combined_1f1b_schedule_for_interleaved_pipelining",
        combined,
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
