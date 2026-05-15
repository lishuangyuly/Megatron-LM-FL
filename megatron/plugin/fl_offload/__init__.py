"""fl-offload plugin: schedule-scoped activation offload for Megatron-LM-FL.

This package is an opt-in plugin that wraps the Megatron pipeline schedule
so that, between a microbatch forward and its backward, activations saved by
``torch.autograd`` can be moved to CPU pinned memory and brought back just
before they are needed again.

Importing this package has no runtime effect on its own — call
:func:`apply` from the training entry point to opt in.

This commit (commit 1) only lays out the package skeleton and the empty
configuration / runtime singletons.  The real implementation arrives in
later commits according to the plan in ``fl_offload_plan.md``.
"""

from megatron.plugin.fl_offload.apply import apply
from megatron.plugin.fl_offload.config import (
    FlOffloadConfig,
    get_config,
    set_config,
)

__all__ = ["apply", "FlOffloadConfig", "get_config", "set_config"]
