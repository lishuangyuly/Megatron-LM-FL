"""Per-batch reporting & degradation warnings (commit 7).

Single-process, single-thread state that the runtime façade and schedule
wrapper feed during training.  Three call sites matter:

* ``record_microbatch_offload(num_tensors, num_bytes)`` — invoked from
  :class:`~megatron.plugin.fl_offload.runtime._ForwardMicrobatchContext`
  right after ``OffloadAsync.__enter__()`` populates the active
  :class:`~megatron.plugin.fl_offload.group.ActivationGroup`.
* ``record_degradation_warning()`` — invoked from
  :class:`~megatron.plugin.fl_offload.runtime.OffloadAsync` /
  :class:`~megatron.plugin.fl_offload.runtime.OnloadAsync` ``__exit__``
  when ``issued_stages < stages`` on entry (i.e. the caller short-changed
  the staged issue contract).
* ``report_after_step()`` — invoked from
  :func:`~megatron.plugin.fl_offload.schedules.wrappers.wrap_schedule_for_offload`
  after the wrapped schedule call returns.  Increments the step counter
  and, if ``report_interval > 0`` and the step is interval-aligned,
  prints a single line on rank 0.

Sample report line::

    [fl-offload] step=10 tensors=4960 act_bytes=10.2GiB \
        peak_pinned_MiB=2048 groups_resident=0 degradation=0

Disabled with ``--fl-offload-report-interval 0`` (the default in
Commit 7 is 10).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ObservabilityState:
    """Mutable accumulators owned by the singleton.

    ``interval_*`` reset every print; ``total_*`` and
    ``degradation_warnings`` are cumulative across training.
    """

    total_tensors: int = 0
    total_bytes: int = 0
    interval_tensors: int = 0
    interval_bytes: int = 0
    degradation_warnings: int = 0
    unpack_fallbacks: int = 0
    train_steps_seen: int = 0

    def reset_interval(self) -> None:
        self.interval_tensors = 0
        self.interval_bytes = 0


_STATE: ObservabilityState = ObservabilityState()


def get_state() -> ObservabilityState:
    """Return the singleton state.

    Tests use this to inspect counters; production callers go through
    the recorder functions below.
    """
    return _STATE


def _reset_state_for_tests() -> None:
    """Force a fresh singleton.  Tests only."""
    global _STATE
    _STATE = ObservabilityState()


def record_microbatch_offload(num_tensors: int, num_bytes: int) -> None:
    """Accumulate one microbatch's worth of offloaded tensors / bytes."""
    if num_tensors <= 0 and num_bytes <= 0:
        return
    _STATE.total_tensors += int(num_tensors)
    _STATE.total_bytes += int(num_bytes)
    _STATE.interval_tensors += int(num_tensors)
    _STATE.interval_bytes += int(num_bytes)


def record_degradation_warning() -> None:
    """Bump the running degradation-warning counter.

    Called from ``OffloadAsync`` / ``OnloadAsync`` ``__exit__`` when the
    caller failed to issue every stage.  Commit 7's wrappers explicitly
    pre-issue all stages, so this should never fire on the happy path;
    Commit 8's staged-issue queue will trip it on misuse.
    """
    _STATE.degradation_warnings += 1


_UNPACK_FALLBACK_WARN_LIMIT = 8


def _rank_tag() -> str:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return f"rank{dist.get_rank()}"
    except Exception:
        pass
    return "rank?"


def record_unpack_fallback(op_name, shape, owner_key, backward_key) -> None:
    """One saved tensor reached ``unpack`` with ``tw.x is None``.

    The defensive sync-onload in ``TensorPack.get`` restored it, but a
    non-zero count means the scheduling layer onloaded the *wrong* group
    (or none) before that backward — i.e. a key-pairing bug, not a user
    error.  The first few occurrences are printed with the tensor's
    owning group key vs. the key the backward context was opened with,
    which is exactly the pairing evidence needed to debug it.
    """
    _STATE.unpack_fallbacks += 1
    if _STATE.unpack_fallbacks <= _UNPACK_FALLBACK_WARN_LIMIT:
        print(
            f"[fl-offload][WARN][{_rank_tag()}] unpack fallback #{_STATE.unpack_fallbacks}: "
            f"op={op_name} shape={tuple(shape)} owner_key={owner_key} "
            f"current_backward_key={backward_key}",
            flush=True,
        )


def _format_bytes(num_bytes: int) -> str:
    """Human-readable byte size with binary units."""
    if num_bytes < 1024:
        return f"{num_bytes}B"
    val = float(num_bytes)
    for unit in ("KiB", "MiB", "GiB", "TiB"):
        val /= 1024.0
        if val < 1024.0:
            return f"{val:.2f}{unit}"
    return f"{val:.2f}PiB"


def format_report(state: ObservabilityState, step_id: int) -> str:
    """Build the rank-0 report line.

    Pool peak and ``_GROUPS`` residency are sampled at call time (so
    they reflect the moment the report is emitted) rather than being
    accumulated in state.
    """
    # Late imports keep this module importable from the runtime even
    # before the pool / runtime modules have finished initialising.
    from megatron.plugin.fl_offload.pool import get_global_pool
    from megatron.plugin.fl_offload.runtime import _GROUPS

    peak_pinned_bytes = get_global_pool().peak_pinned_bytes()
    peak_pinned_mib = peak_pinned_bytes >> 20
    groups_resident = len(_GROUPS)

    return (
        f"[fl-offload] step={step_id} "
        f"tensors={state.interval_tensors} "
        f"act_bytes={_format_bytes(state.interval_bytes)} "
        f"peak_pinned_MiB={peak_pinned_mib} "
        f"groups_resident={groups_resident} "
        f"degradation={state.degradation_warnings} "
        f"unpack_fallback={state.unpack_fallbacks}"
    )


def _is_rank_zero() -> bool:
    """Best-effort rank-0 check that also works in single-process tests."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:
        pass
    return True


def report_after_step() -> None:
    """Called once per ``forward_backward_func`` invocation.

    Increments the step counter and prints a report line if the step
    aligns with the interval and the caller is rank 0.  ``interval <= 0``
    disables printing while still bumping the counter (so cumulative
    counters remain available via :func:`get_state` even when reports
    are off).
    """
    from megatron.plugin.fl_offload.config import get_config

    _STATE.train_steps_seen += 1
    interval = int(get_config().report_interval)
    if interval <= 0:
        return
    if _STATE.train_steps_seen % interval != 0:
        return
    if not _is_rank_zero():
        _STATE.reset_interval()
        return
    print(format_report(_STATE, _STATE.train_steps_seen), flush=True)
    _STATE.reset_interval()


__all__ = [
    "ObservabilityState",
    "format_report",
    "get_state",
    "record_degradation_warning",
    "record_microbatch_offload",
    "record_unpack_fallback",
    "report_after_step",
]
