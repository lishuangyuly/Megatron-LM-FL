"""``apply()`` entry point for the fl-offload plugin.

Pretrain entry scripts call ``apply()`` to opt in.  In commit 1 this is a
no-op stub that simply returns the caller's ``extra_args_provider`` and
``validate_args`` unchanged, so importing the plugin and calling ``apply()``
on the default branch keeps trunk behaviour bit-exact.

Later commits will fill in:

* commit 2 — wrap ``extra_args_provider`` to inject the eight ``--fl-offload-*``
  CLI flags, and wrap ``validate_args`` to enforce range / dependency / mutex
  rules and call ``set_config(...)``;
* commit 5 — additionally monkey-patch
  ``megatron.core.pipeline_parallel.schedules.get_forward_backward_func`` with
  the plugin's selector.
"""

from typing import Callable, Optional, Tuple


def apply(
    extra_args_provider: Optional[Callable] = None,
    validate_args: Optional[Callable] = None,
) -> Tuple[Optional[Callable], Optional[Callable]]:
    """Install the fl-offload plugin.

    Args:
        extra_args_provider: The caller's existing argparse extension hook,
            or ``None``.  Returned unchanged in commit 1.
        validate_args: The caller's existing validate_args replacement, or
            ``None``.  Returned unchanged in commit 1.

    Returns:
        A ``(extra_args_provider, validate_args)`` tuple that the caller can
        pass straight into ``pretrain(...)``.  In commit 1 these are the
        original callables (or ``None``).
    """
    return extra_args_provider, validate_args


__all__ = ["apply"]
