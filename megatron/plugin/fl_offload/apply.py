"""``apply()`` entry point for the fl-offload plugin.

Pretrain entry scripts call ``apply()`` to opt in (``pretrain_gpt.py``
does since commit 7).  The call:

1. chains the plugin's argparse provider after the user's;
2. wraps the caller's ``validate_args`` with the plugin's checks if the
   caller actually has one — FL's ``pretrain()`` exposes no such hook,
   so in the normal path the returned validator is ``None`` and the
   config instead lands lazily on the first schedule selection (see
   ``selector._land_config_from_global_args``);
3. **registers and applies** a monkey-patch on
   ``megatron.core.pipeline_parallel.schedules.get_forward_backward_func``
   so that ``enable=True`` runs route through the offload-aware schedule
   wrappers.  The patch itself short-circuits to identity when
   ``FlOffloadConfig.enable`` is False, so registering it is safe even
   for users who never flip the flag.
"""

from typing import Callable, Optional, Tuple

from megatron.plugin.fl_offload._patch import MegatronPatchesManager
from megatron.plugin.fl_offload.arguments import chain_extra_args_provider
from megatron.plugin.fl_offload.selector import get_forward_backward_func_wrapper
from megatron.plugin.fl_offload.validate import validate_args_wrapper


_PATCH_TARGET = "megatron.core.pipeline_parallel.schedules.get_forward_backward_func"


def apply(
    extra_args_provider: Optional[Callable] = None,
    validate_args: Optional[Callable] = None,
) -> Tuple[Callable, Optional[Callable]]:
    """Install the fl-offload plugin's argparse + validate + schedule hooks.

    Args:
        extra_args_provider: The caller's existing argparse extension hook,
            or ``None``.  The returned provider always runs the caller's
            provider first and then registers the plugin's flags.
        validate_args: The caller's existing ``validate_args`` replacement,
            or ``None``.

    Returns:
        ``(extra_args_provider, validate_args)``.  Pass straight into
        ``pretrain(...)``.  The first element is always a callable
        (never ``None``).
    """
    chained_provider = chain_extra_args_provider(extra_args_provider)
    chained_validator = (
        validate_args_wrapper(validate_args) if validate_args is not None else None
    )

    # Register the selector patch once.  ``register_patch`` is additive,
    # so callers that invoke apply() more than once still get a single
    # active patch.
    if not any(p.target == _PATCH_TARGET for p in MegatronPatchesManager._patches):
        MegatronPatchesManager.register_patch(
            _PATCH_TARGET,
            get_forward_backward_func_wrapper,
            apply_wrapper=True,
        )
    MegatronPatchesManager.apply_patches()

    return chained_provider, chained_validator


__all__ = ["apply"]
