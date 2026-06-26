"""Install the first-stage direct port on Megatron's ordinary schedule."""

import functools

import torch

from .offload import OffloadAsync, OnloadAsync, enabled, record
from .te_patch import apply_te_patches


_INSTALLED = False
_OUTPUT_KEYS = {}
_SEQUENCE = 0


def _output_id(output):
    if isinstance(output, (list, tuple)) and output:
        return id(output[0])
    return id(output)


def install():
    """Patch schedule entry points without changing behavior while disabled."""
    global _INSTALLED
    if _INSTALLED:
        return

    from megatron.core.pipeline_parallel import schedules

    original_forward = schedules.forward_step
    original_backward = schedules.backward_step

    @functools.wraps(original_forward)
    def forward_step(*args, **kwargs):
        global _SEQUENCE
        if not enabled() or not torch.is_grad_enabled():
            return original_forward(*args, **kwargs)
        apply_te_patches()
        key = (
            kwargs.get("vp_stage", 0) or 0,
            kwargs.get("current_microbatch", 0) or 0,
            _SEQUENCE,
        )
        _SEQUENCE += 1
        with record(key):
            result = original_forward(*args, **kwargs)
        with OffloadAsync(key) as offload:
            offload.issue(offload.group_num - 1)
        output = result[0] if isinstance(result, tuple) else result
        _OUTPUT_KEYS[_output_id(output)] = key
        return result

    @functools.wraps(original_backward)
    def backward_step(*args, **kwargs):
        if not enabled():
            return original_backward(*args, **kwargs)
        output = args[1] if len(args) > 1 else kwargs.get("output_tensor")
        key = _OUTPUT_KEYS.pop(_output_id(output), None)
        if key is None:
            raise RuntimeError("FL offload could not match backward output to its forward group")
        with OnloadAsync(key) as onload:
            onload.issue(onload.group_num - 1)
        return original_backward(*args, **kwargs)

    schedules.forward_step = forward_step
    schedules.backward_step = backward_step
    _INSTALLED = True
