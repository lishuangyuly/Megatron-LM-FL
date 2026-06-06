"""``validate_args`` wrapper for the fl-offload plugin.

The wrapper layers three kinds of checks on top of the caller's existing
``validate_args``:

1. **Range** — every numeric knob is within its documented domain.
2. **Dependency** — ``interleaved_1f1b`` requires PP > 1 and VPP set.
3. **Mutual exclusion** — ``fl_offload_enable`` cannot be combined with
   ``fine_grained_activation_offloading``, ``cpu_offloading`` /
   ``cpu_offloading_num_layers > 0`` or ``use_dualpipev``.

When everything passes the wrapper lifts ``args.fl_offload_*`` into a fresh
:class:`~megatron.plugin.fl_offload.config.FlOffloadConfig` and installs it
via :func:`~megatron.plugin.fl_offload.config.set_config`.
"""

from functools import wraps
from typing import Any, Callable

from megatron.plugin.fl_offload.config import FlOffloadConfig, set_config


_BACKENDS = {"vanilla", "interleaved_1f1b"}

# Attribute names on argparse ``Namespace`` we need to consult.  Kept as
# constants so tests can mock them without re-typing strings.
_BACKEND_ATTR = "pipeline_schedule_backend"
_ENABLE_ATTR = "fl_offload_enable"
_MIN_BYTES_ATTR = "fl_offload_min_bytes"
_NON_CONTIG_ATTR = "fl_offload_non_contiguous"
_PIN_MEMORY_ATTR = "fl_offload_pin_memory"
_RATIO_ATTR = "fl_offload_ratio"
_PER_BATCH_ATTR = "fl_offload_per_batch_size"
_STAGES_ATTR = "fl_offload_stages"
_REPORT_INTERVAL_ATTR = "fl_offload_report_interval"
_ALLOW_CUDA_GRAPH_ATTR = "fl_offload_allow_cuda_graph"


def _check_ranges(args: Any) -> None:
    backend = getattr(args, _BACKEND_ATTR, "vanilla")
    if backend not in _BACKENDS:
        raise AssertionError(
            f"--pipeline-schedule-backend must be one of {sorted(_BACKENDS)}, "
            f"got {backend!r}"
        )

    if getattr(args, _MIN_BYTES_ATTR, 0) < 0:
        raise AssertionError("--fl-offload-min-bytes must be non-negative")

    ratio = getattr(args, _RATIO_ATTR, 1.0)
    if not 0.0 <= ratio <= 1.0:
        raise AssertionError(
            f"--fl-offload-ratio must be in [0, 1], got {ratio}"
        )

    per_batch = getattr(args, _PER_BATCH_ATTR, 0.0)
    if per_batch < 0:
        raise AssertionError("--fl-offload-per-batch-size must be non-negative")

    stages = getattr(args, _STAGES_ATTR, 1)
    if stages < 1:
        raise AssertionError("--fl-offload-stages must be >= 1")

    interval = getattr(args, _REPORT_INTERVAL_ATTR, 0)
    if interval < 0:
        raise AssertionError("--fl-offload-report-interval must be >= 0")


def _check_backend_dependencies(args: Any) -> None:
    backend = getattr(args, _BACKEND_ATTR, "vanilla")
    if backend != "interleaved_1f1b":
        return

    pp = getattr(args, "pipeline_model_parallel_size", 1) or 1
    vpp = getattr(args, "virtual_pipeline_model_parallel_size", None)
    if pp <= 1:
        raise AssertionError(
            "interleaved_1f1b requires --pipeline-model-parallel-size > 1"
        )
    if vpp is None:
        raise AssertionError(
            "interleaved_1f1b requires --num-layers-per-virtual-pipeline-stage "
            "(or another knob that sets virtual_pipeline_model_parallel_size)"
        )


def _check_mutual_exclusion(args: Any) -> None:
    if not getattr(args, _ENABLE_ATTR, False):
        return

    conflicts = []
    if getattr(args, "fine_grained_activation_offloading", False):
        conflicts.append("fine_grained_activation_offloading")
    if getattr(args, "cpu_offloading", False):
        conflicts.append("cpu_offloading")
    if getattr(args, "cpu_offloading_num_layers", 0) > 0:
        # ``cpu_offloading_num_layers > 0`` implies the same path; flag it
        # explicitly so the error message matches the user's CLI.
        if "cpu_offloading" not in conflicts:
            conflicts.append("cpu_offloading_num_layers>0")
    if getattr(args, "use_dualpipev", False):
        conflicts.append("use_dualpipev")

    if conflicts:
        raise AssertionError(
            "--fl-offload-enable is mutually exclusive with: "
            + ", ".join(conflicts)
            + ". Disable one before retrying."
        )


def _check_cuda_graph_guard(args: Any) -> None:
    """Refuse fl-offload + non-trivial cuda_graph_impl unless opted in.

    ``saved_tensors_hooks`` behaviour during CUDA graph capture is
    restricted; until that interaction is audited, fl-offload defaults
    to refusing combinations like ``--cuda-graph-impl=local``.  Users
    can override with ``--fl-offload-allow-cuda-graph``.
    """
    if not getattr(args, _ENABLE_ATTR, False):
        return
    impl = getattr(args, "cuda_graph_impl", None)
    if impl is None or impl == "none":
        return
    if getattr(args, _ALLOW_CUDA_GRAPH_ATTR, False):
        return
    raise AssertionError(
        f"--fl-offload-enable refused with --cuda-graph-impl={impl!r}. "
        "Pass --fl-offload-allow-cuda-graph to override (saved_tensors_hooks "
        "behaviour during graph capture is not yet audited)."
    )


def _config_from_args(args: Any) -> FlOffloadConfig:
    """Lift ``args.fl_offload_*`` into a :class:`FlOffloadConfig`."""
    return FlOffloadConfig(
        enable=getattr(args, _ENABLE_ATTR, False),
        pipeline_schedule_backend=getattr(args, _BACKEND_ATTR, "vanilla"),
        min_bytes=getattr(args, _MIN_BYTES_ATTR, 1 << 20),
        non_contiguous=getattr(args, _NON_CONTIG_ATTR, False),
        pin_memory=getattr(args, _PIN_MEMORY_ATTR, True),
        ratio=float(getattr(args, _RATIO_ATTR, 1.0)),
        per_batch_size=float(getattr(args, _PER_BATCH_ATTR, 0.0)),
        stages=int(getattr(args, _STAGES_ATTR, 1)),
        report_interval=int(getattr(args, _REPORT_INTERVAL_ATTR, 10)),
        allow_cuda_graph=bool(getattr(args, _ALLOW_CUDA_GRAPH_ATTR, False)),
    )


def validate_plugin_args(args: Any) -> FlOffloadConfig:
    """Run all checks and install the resulting config.

    Tests use this directly; production code goes through
    :func:`validate_args_wrapper` so the caller's existing ``validate_args``
    runs first.
    """
    _check_ranges(args)
    _check_backend_dependencies(args)
    _check_mutual_exclusion(args)
    _check_cuda_graph_guard(args)
    cfg = _config_from_args(args)
    set_config(cfg)
    return cfg


def validate_args_wrapper(validate_args: Callable) -> Callable:
    """Wrap an existing ``validate_args`` so plugin checks run after it.

    The wrapper preserves the upstream signature
    ``validate_args(args, defaults=None) -> args`` used by
    ``megatron.training.arguments``.
    """

    @wraps(validate_args)
    def wrapper(args, defaults=None):
        if defaults is None:
            defaults = {}
        args = validate_args(args, defaults)
        validate_plugin_args(args)
        return args

    return wrapper


__all__ = ["validate_args_wrapper", "validate_plugin_args"]
