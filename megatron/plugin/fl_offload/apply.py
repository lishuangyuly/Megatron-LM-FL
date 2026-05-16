"""``apply()`` entry point for the fl-offload plugin.

Pretrain entry scripts call ``apply()`` to opt in.  Commit 2 fills in the
argparse + validate wiring; commit 5 will additionally monkey-patch
``megatron.core.pipeline_parallel.schedules.get_forward_backward_func``.

After commit 2, calling ``apply()`` is still safe on the default branch:

* the chained provider only registers ``--fl-offload-*`` flags (their
  defaults are no-ops);
* the wrapped ``validate_args`` only enforces extra checks when those flags
  are present and produces a no-op :class:`FlOffloadConfig` otherwise.
"""

from typing import Callable, Optional, Tuple

from megatron.plugin.fl_offload.arguments import chain_extra_args_provider
from megatron.plugin.fl_offload.validate import validate_args_wrapper


def apply(
    extra_args_provider: Optional[Callable] = None,
    validate_args: Optional[Callable] = None,
) -> Tuple[Callable, Optional[Callable]]:
    """Install the fl-offload plugin's argparse + validate hooks.

    Args:
        extra_args_provider: The caller's existing argparse extension hook,
            or ``None``.  The returned provider always runs the caller's
            provider first and then registers the plugin's flags.
        validate_args: The caller's existing ``validate_args`` replacement,
            or ``None``.  When ``None`` we return ``None`` for the second
            element of the tuple, signalling "use Megatron's default
            ``validate_args``" — commit 7 will revisit this once the plugin
            takes responsibility for installing its own validator.

    Returns:
        ``(extra_args_provider, validate_args)``.  Pass straight into
        ``pretrain(...)``.  The first element is always a callable
        (never ``None``).
    """
    chained_provider = chain_extra_args_provider(extra_args_provider)
    chained_validator = (
        validate_args_wrapper(validate_args) if validate_args is not None else None
    )
    return chained_provider, chained_validator


__all__ = ["apply"]
