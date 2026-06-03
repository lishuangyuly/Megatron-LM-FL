"""fl-offload plugin: schedule-scoped activation offload for Megatron-LM-FL.

This package is an opt-in plugin that wraps the Megatron pipeline schedule
so that, between a microbatch forward and its backward, activations saved by
``torch.autograd`` can be moved to CPU pinned memory and brought back just
before they are needed again.

Importing this package has no runtime effect on its own — call
:func:`apply` from the training entry point to opt in.

Commit 6 wires the schedule selector: ``apply()`` now also monkey-patches
``megatron.core.pipeline_parallel.schedules.get_forward_backward_func``
so that, when ``FlOffloadConfig.enable`` is True and the chosen schedule
is ``forward_backward_pipelining_with_interleaving``, training runs
through the offload-aware wrappers.  ``enable=False`` short-circuits to
the original FL callable.
"""

from megatron.plugin.fl_offload.apply import apply
from megatron.plugin.fl_offload.arguments import (
    add_fl_offload_args,
    chain_extra_args_provider,
)
from megatron.plugin.fl_offload.config import (
    FlOffloadConfig,
    get_config,
    set_config,
)
from megatron.plugin.fl_offload.filter import is_tensor_eligible
from megatron.plugin.fl_offload.group import ActivationGroup, CopyTaskGroup
from megatron.plugin.fl_offload.hooks import (
    current_collection,
    pack_hook,
    record,
    unpack_hook,
)
from megatron.plugin.fl_offload._patch import MegatronPatchesManager, Patch
from megatron.plugin.fl_offload.pool import (
    PinnedBufferPool,
    get_global_pool,
    reset_global_pool,
)
from megatron.plugin.fl_offload.schedules.keys import make_offload_key_interleaved
from megatron.plugin.fl_offload.schedules.wrappers import (
    _patch_step_funcs,
    wrap_schedule_for_offload,
)
from megatron.plugin.fl_offload.selector import get_forward_backward_func_wrapper
from megatron.plugin.fl_offload.runtime import (
    OffloadAsync,
    OnloadAsync,
    PipelineActivationOffloadRuntime,
    TensorPack,
    TensorWrap,
    byte_view,
    fast_contiguous,
    get_pipeline_offload_runtime,
    has_group,
    register_group,
)
from megatron.plugin.fl_offload.stream import (
    get_memcpy_stream,
    has_async_streams,
    reset_streams,
)
from megatron.plugin.fl_offload.validate import (
    validate_args_wrapper,
    validate_plugin_args,
)

__all__ = [
    "ActivationGroup",
    "CopyTaskGroup",
    "FlOffloadConfig",
    "MegatronPatchesManager",
    "OffloadAsync",
    "OnloadAsync",
    "Patch",
    "PinnedBufferPool",
    "PipelineActivationOffloadRuntime",
    "TensorPack",
    "TensorWrap",
    "_patch_step_funcs",
    "add_fl_offload_args",
    "apply",
    "byte_view",
    "chain_extra_args_provider",
    "current_collection",
    "fast_contiguous",
    "get_config",
    "get_forward_backward_func_wrapper",
    "get_global_pool",
    "get_memcpy_stream",
    "get_pipeline_offload_runtime",
    "has_async_streams",
    "has_group",
    "is_tensor_eligible",
    "make_offload_key_interleaved",
    "pack_hook",
    "record",
    "register_group",
    "reset_global_pool",
    "reset_streams",
    "set_config",
    "unpack_hook",
    "validate_args_wrapper",
    "validate_plugin_args",
    "wrap_schedule_for_offload",
]
