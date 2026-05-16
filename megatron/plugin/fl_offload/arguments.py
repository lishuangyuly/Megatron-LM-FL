"""Argparse extension for the fl-offload plugin.

Two callables matter:

* :func:`add_fl_offload_args` — registers all CLI flags onto an
  existing ``argparse.ArgumentParser``.  Every flag is guarded by
  :func:`_has_option` so calling it twice on the same parser is a no-op,
  which keeps things safe when a user also chains their own provider that
  happens to add the same flag.
* :func:`chain_extra_args_provider` — wraps the caller's existing
  ``extra_args_provider`` so the plugin's flags are added *after* the
  caller's flags.  The caller's provider may be ``None``; in that case
  this returns a provider that only adds the plugin's flags.

Defaults match those documented in ``fl_offload_plan.md``.
"""

import argparse
from typing import Callable, Optional


_BACKEND_CHOICES = ("vanilla", "interleaved_1f1b")


def _has_option(parser: argparse.ArgumentParser, option_string: str) -> bool:
    """Return True if ``parser`` already knows about ``option_string``."""
    return option_string in parser._option_string_actions


def add_fl_offload_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register fl-offload CLI flags onto ``parser``.

    Adds a dedicated ``argparse`` group so ``--help`` output stays tidy.
    Returns ``parser`` so this function can sit in a provider chain.
    """
    group = parser.add_argument_group(title="fl-offload activation offload")

    # Schedule backend selector.  The primary flag is the generic
    # ``--pipeline-schedule-backend`` so existing scripts that already use
    # ``--schedule-method`` keep working.  Both spellings dest to the same
    # attribute.
    if not _has_option(parser, "--pipeline-schedule-backend"):
        group.add_argument(
            "--pipeline-schedule-backend",
            "--fl-offload-pipeline-schedule-backend",
            "--schedule-method",
            dest="pipeline_schedule_backend",
            default="vanilla",
            choices=list(_BACKEND_CHOICES),
            help=(
                "Pipeline schedule backend. 'vanilla' keeps the FL "
                "default behaviour; 'interleaved_1f1b' forces the "
                "interleaved schedule and requires PP>1 with VPP set."
            ),
        )

    if not _has_option(parser, "--fl-offload-enable"):
        group.add_argument(
            "--fl-offload-enable",
            action="store_true",
            default=False,
            dest="fl_offload_enable",
            help="Master switch for the fl-offload activation offload plugin.",
        )

    if not _has_option(parser, "--fl-offload-min-bytes"):
        group.add_argument(
            "--fl-offload-min-bytes",
            type=int,
            default=1 << 20,
            dest="fl_offload_min_bytes",
            help=(
                "Minimum CUDA tensor size, in bytes, eligible for offload. "
                "Default 1 MiB."
            ),
        )

    if not _has_option(parser, "--fl-offload-non-contiguous"):
        group.add_argument(
            "--fl-offload-non-contiguous",
            action="store_true",
            default=False,
            dest="fl_offload_non_contiguous",
            help=(
                "Allow offload for non-contiguous / view-backed saved "
                "tensors. Disabled by default because some fused kernels "
                "require the original saved-tensor layout in backward."
            ),
        )

    if not _has_option(parser, "--fl-offload-no-pin-memory"):
        group.add_argument(
            "--fl-offload-no-pin-memory",
            action="store_false",
            default=True,
            dest="fl_offload_pin_memory",
            help="Use regular CPU memory instead of pinned memory for offload.",
        )

    if not _has_option(parser, "--fl-offload-ratio"):
        group.add_argument(
            "--fl-offload-ratio",
            type=float,
            default=1.0,
            dest="fl_offload_ratio",
            help=(
                "Fraction of eligible saved activations to offload per "
                "microbatch. Ignored when --fl-offload-per-batch-size > 0."
            ),
        )

    if not _has_option(parser, "--fl-offload-per-batch-size"):
        group.add_argument(
            "--fl-offload-per-batch-size",
            type=float,
            default=0.0,
            dest="fl_offload_per_batch_size",
            help=(
                "Fixed per-microbatch activation offload budget in MiB. "
                "0 falls back to the ratio-based budget."
            ),
        )

    if not _has_option(parser, "--fl-offload-stages"):
        group.add_argument(
            "--fl-offload-stages",
            type=int,
            default=1,
            dest="fl_offload_stages",
            help="Number of copy buckets the offload work is split into.",
        )

    return parser


def chain_extra_args_provider(
    extra_args_provider: Optional[Callable] = None,
) -> Callable:
    """Compose the caller's ``extra_args_provider`` with ours.

    The returned provider:

    1. delegates to ``extra_args_provider`` first (if not ``None``);
    2. then calls :func:`add_fl_offload_args` so plugin flags appear after
       any caller-defined flags;
    3. returns the parser (Megatron's ``parse_args`` ignores the return
       value but downstream chains may use it).

    Calling this when ``extra_args_provider`` is ``None`` is equivalent to
    handing back :func:`add_fl_offload_args` directly.
    """

    def provider(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        if extra_args_provider is not None:
            extra_parser = extra_args_provider(parser)
            if extra_parser is not None:
                parser = extra_parser
        return add_fl_offload_args(parser)

    return provider


__all__ = ["add_fl_offload_args", "chain_extra_args_provider"]
