"""Pipeline activation offload runtime façade.

Surface frozen since commit 1:

* :class:`PipelineActivationOffloadRuntime` with
  :meth:`forward_microbatch` / :meth:`backward_microbatch` context
  managers and an :meth:`enabled` predicate;
* :func:`get_pipeline_offload_runtime` — process-singleton getter;
* :class:`OffloadAsync` / :class:`OnloadAsync` — context managers driving
  one microbatch's D2H / H2D work respectively.

Commit 3 adds:

* :func:`byte_view` / :func:`fast_contiguous` — typeless / contiguity
  helpers used by :class:`~megatron.plugin.fl_offload.group.ActivationGroup`;
* :class:`TensorWrap` / :class:`TensorPack` — minimal data classes that
  outlive the pack/unpack lifecycle hooked up in commit 4;
* :data:`_GROUPS` plus :func:`register_group` / :func:`get_group` /
  :func:`pop_group` / :func:`has_group` — the per-key registry that
  bridges forward-side ``OffloadAsync`` and backward-side ``OnloadAsync``;
* real bodies for :class:`OffloadAsync` / :class:`OnloadAsync`.

The microbatch façade (``forward_microbatch`` /
``backward_microbatch``) still returns ``contextlib.nullcontext()`` —
commit 4 will replace it with real ``record()``-based contexts.
"""

import contextlib
from typing import Any, Dict, Optional, Tuple

import torch

from megatron.plugin.fl_offload.config import get_config


# ---------------------------------------------------------------------- #
# Byte-level helpers                                                     #
# ---------------------------------------------------------------------- #
def byte_view(tensor: torch.Tensor) -> torch.Tensor:
    """Return a 1-D ``uint8`` view of ``tensor``'s underlying storage.

    The caller must guarantee ``tensor`` is contiguous — use
    :func:`fast_contiguous` first if unsure.  PyTorch requires contiguity
    for the ``view(dtype)`` dtype-reinterpretation path.
    """
    return tensor.view(torch.uint8).reshape(-1)


def fast_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    """Return ``tensor`` itself if already contiguous, else its ``.contiguous()`` copy."""
    if tensor.is_contiguous():
        return tensor
    return tensor.contiguous()


# ---------------------------------------------------------------------- #
# Activation wrappers — outlive the autograd pack/unpack cycle           #
# ---------------------------------------------------------------------- #
class TensorWrap:
    """Mutable holder for one saved activation.

    The ``x`` slot mutates across the life of one microbatch:

    * starts as the live GPU tensor produced by forward;
    * is set to ``None`` once :meth:`offload_epilogue` confirms its bytes
      live on CPU — releasing the GPU reference, but **not** resizing
      the underlying storage (a pipeline boundary may still alias it);
    * is replaced by a freshly-allocated GPU tensor when
      :meth:`onload_issue` copies the CPU buffer back.
    """

    __slots__ = ("x", "shape", "dtype", "device", "cpu_buffer", "owner_key")

    def __init__(self, tensor: torch.Tensor) -> None:
        self.x: Optional[torch.Tensor] = tensor
        self.shape = tensor.shape
        self.dtype = tensor.dtype
        self.device = tensor.device
        self.cpu_buffer: Optional[torch.Tensor] = None
        # Key of the ActivationGroup that offloaded this wrap (set by
        # ``ActivationGroup.__init__``); diagnostic only.
        self.owner_key: Optional[Any] = None


class TensorPack:
    """Object returned from :func:`pack_hook` (commit 4).

    Wrapping the :class:`TensorWrap` lets the ``unpack_hook`` see the
    current ``x`` value (post-onload) without baking in the original
    GPU reference.
    """

    __slots__ = ("tensor_wrap", "op_name")

    def __init__(self, tensor_wrap: TensorWrap, op_name: Optional[str] = None) -> None:
        self.tensor_wrap = tensor_wrap
        self.op_name = op_name

    def get(self) -> Optional[torch.Tensor]:
        tw = self.tensor_wrap
        if tw.x is None and tw.cpu_buffer is not None:
            # Defensive sync onload: backward reached this saved tensor
            # while its group was still offloaded — the scheduling layer
            # failed to pair this microbatch's forward and backward keys.
            # Restore inline so training stays *correct* (the leaked
            # group's pinned buffer is never returned to the pool, so its
            # bytes are intact), and report loudly so the pairing bug is
            # visible instead of surfacing as silent wgrad skips /
            # ``param.grad is None`` asserts downstream.
            new_tensor = torch.empty(tw.shape, dtype=tw.dtype, device=tw.device)
            byte_view(new_tensor).copy_(tw.cpu_buffer)
            tw.x = new_tensor
            from megatron.plugin.fl_offload.observability import (
                record_unpack_fallback,
            )

            record_unpack_fallback(
                self.op_name, tw.shape, tw.owner_key, _CURRENT_BACKWARD_KEY
            )
        return tw.x


# Diagnostic: the offload key of the backward microbatch currently being
# executed (None outside a backward context).  Read by the unpack
# fallback so its warning can show "tensor belongs to group X, but the
# backward running now onloaded group Y".
_CURRENT_BACKWARD_KEY: Optional[Any] = None

# Key-trace debug log, enabled with FL_OFFLOAD_DEBUG_KEYS=1: prints one
# line per forward registration and per backward lookup (hit/miss) so a
# failing run shows the full key-pairing picture even when the failure
# itself produces no fallback warning.  Capped per process to keep logs
# sane on long runs.
_DEBUG_KEYS_LIMIT = 200
_debug_keys_printed = 0


def _debug_keys(message: str) -> None:
    global _debug_keys_printed
    import os

    if os.environ.get("FL_OFFLOAD_DEBUG_KEYS") != "1":
        return
    if _debug_keys_printed >= _DEBUG_KEYS_LIMIT:
        return
    _debug_keys_printed += 1
    from megatron.plugin.fl_offload.observability import _rank_tag

    print(f"[fl-offload][KEYS][{_rank_tag()}] {message}", flush=True)


# NOTE on saved Parameters: with ``saved_tensors_hooks`` active, torch
# strips the Parameter wrapper at save time and *rewraps* the unpack
# result into a fresh attribute-less tensor (verified by
# ``probe_saved_hooks.py`` on torch 2.6.0) — so no pack/unpack-side
# identity restoration can ever make backward see the original weight
# object.  TE's fused-wgrad protocol therefore cannot work under hooks;
# until the explicit-pack TE patch (Commit 7.2) replaces the hooks
# collection channel, ``validate.py`` refuses
# ``gradient_accumulation_fusion=True`` combined with fl-offload.


# ---------------------------------------------------------------------- #
# Cross-context group registry                                           #
# ---------------------------------------------------------------------- #
# Keys are tuples produced by ``schedules/keys.py`` (commit 5+) — they
# must hash and compare reliably across forward / backward.
_GROUPS: Dict[Any, "object"] = {}


def register_group(key: Any, group: "object") -> None:
    """Install ``group`` under ``key``. Overwrites any existing entry."""
    _GROUPS[key] = group


def get_group(key: Any) -> Optional["object"]:
    return _GROUPS.get(key)


def pop_group(key: Any) -> Optional["object"]:
    return _GROUPS.pop(key, None)


def has_group(key: Any) -> bool:
    return key in _GROUPS


def _reset_groups_for_tests() -> None:
    _GROUPS.clear()


# ---------------------------------------------------------------------- #
# Async D2H / H2D context managers                                       #
# ---------------------------------------------------------------------- #
class OffloadAsync:
    """Drive one :class:`ActivationGroup`'s offload over ``stages`` buckets.

    Usage::

        with OffloadAsync(key, stages=4) as ctx:
            ctx.issue(0)
            ctx.issue(1)
            ...

    Calls to :meth:`issue` are idempotent (re-issuing the same bucket is
    a no-op) and the ``__exit__`` flushes any unissued buckets so the
    schedule cannot accidentally strand work.

    When ``key`` is not registered in :data:`_GROUPS` the context degrades
    to a no-op so the caller does not need to special-case "we decided
    not to record this microbatch".
    """

    def __init__(self, key: Any, stages: int = 1) -> None:
        self.key = key
        self.stages = max(1, int(stages))
        self.issued_stages = 0
        group = get_group(key)
        self.disabled = group is None
        self.group = group

    def __enter__(self) -> "OffloadAsync":
        if self.disabled:
            return self
        self.group.offload_prologue()
        return self

    def issue(self, stage_id: int) -> None:
        if self.disabled:
            return
        target = max(self.issued_stages, min(stage_id + 1, self.stages))
        while self.issued_stages < target:
            self.group.offload_issue(self.issued_stages)
            self.issued_stages += 1

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self.disabled:
            return False
        # Degradation hook: caller exited without externally issuing every
        # stage.  Commit 7's microbatch contexts pre-issue all stages, so
        # this only fires when an upstream caller (or a future staged
        # scheduler) short-changes us.
        if self.issued_stages < self.stages:
            from megatron.plugin.fl_offload.observability import (
                record_degradation_warning,
            )
            record_degradation_warning()
        # Self-rescue: flush any unissued buckets before epilogue.
        while self.issued_stages < self.stages:
            self.group.offload_issue(self.issued_stages)
            self.issued_stages += 1
        self.group.offload_epilogue()
        return False


class OnloadAsync:
    """Counterpart of :class:`OffloadAsync` for the backward path.

    On a clean ``__exit__`` the group is popped from :data:`_GROUPS` so
    its per-microbatch resources are not retained for the next step.
    """

    def __init__(self, key: Any, stages: int = 1) -> None:
        self.key = key
        self.stages = max(1, int(stages))
        self.issued_stages = 0
        group = get_group(key)
        self.disabled = group is None
        self.group = group

    def __enter__(self) -> "OnloadAsync":
        if self.disabled:
            return self
        self.group.onload_prologue()
        return self

    def issue(self, stage_id: int) -> None:
        if self.disabled:
            return
        target = max(self.issued_stages, min(stage_id + 1, self.stages))
        while self.issued_stages < target:
            self.group.onload_issue(self.issued_stages)
            self.issued_stages += 1

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self.disabled:
            return False
        if self.issued_stages < self.stages:
            from megatron.plugin.fl_offload.observability import (
                record_degradation_warning,
            )
            record_degradation_warning()
        while self.issued_stages < self.stages:
            self.group.onload_issue(self.issued_stages)
            self.issued_stages += 1
        self.group.onload_epilogue()
        # Each microbatch's group is single-use; drop it on the way out.
        pop_group(self.key)
        return False


# ---------------------------------------------------------------------- #
# Microbatch contexts                                                    #
# ---------------------------------------------------------------------- #
def _make_key(virtual_microbatch_id: int, model_chunk_id: int) -> Tuple[int, int]:
    """Fallback offload key when the caller does not provide one.

    Tuple ordering matches bgl2's convention: ``(model_chunk_id,
    virtual_microbatch_id)``.  Schedule-specific code in commits 6 / 7
    builds richer tuples and passes them via the explicit
    ``offload_key`` parameter of the façade methods.
    """
    return (model_chunk_id, virtual_microbatch_id)


class _ForwardMicrobatchContext:
    """Wrap one microbatch's forward in a ``record()`` + ``OffloadAsync``.

    Enter sequence:
      1. open ``record(key, stages)`` so saved_tensors_hooks are
         installed and the collection bound to ``key``.

    Exit sequence (clean path):
      1. close ``record()`` (uninstalls hooks, registers the group).
      2. fire ``OffloadAsync(key)`` end-to-end so D2H starts immediately
         and the GPU references are dropped before the next microbatch
         begins forward.

    Exit on exception:
      1. close ``record()`` so hooks are not leaked.
      2. **skip** the OffloadAsync — backward will not run, so the
         tensors stay on the GPU until the surrounding training loop
         tears the iteration down.  ``_GROUPS`` still holds the group;
         the caller should drain it (e.g. in a finally) if it intends to
         retry the iteration.
    """

    def __init__(self, key: Any, stages: int) -> None:
        self.key = key
        self.stages = stages
        self._record_ctx = None

    def __enter__(self) -> "_ForwardMicrobatchContext":
        # Lazy import to break the runtime↔hooks dependency cycle.
        from megatron.plugin.fl_offload.hooks import record

        self._record_ctx = record(self.key, self.stages)
        self._record_ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        # Always close the record() so saved_tensors_hooks are uninstalled.
        suppress = self._record_ctx.__exit__(exc_type, exc, tb)
        grp = get_group(self.key)
        _debug_keys(
            f"fwd register key={self.key} "
            f"collected={len(getattr(grp, 'tensors', ()) or ())} "
            f"exc={exc_type.__name__ if exc_type else None}"
        )
        if exc_type is None:
            # Forward succeeded — open OffloadAsync, capture stats now
            # that ``offloaded_tensors`` is populated, then explicitly
            # issue every stage before exit so the degradation hook
            # inside ``OffloadAsync.__exit__`` does not fire.
            offload_ctx = OffloadAsync(self.key, stages=self.stages)
            offload_ctx.__enter__()
            _record_offload_stats(self.key)
            for stage_id in range(offload_ctx.stages):
                offload_ctx.issue(stage_id)
            offload_ctx.__exit__(None, None, None)
        return bool(suppress)


def _record_offload_stats(key: Any) -> None:
    """Push (num_tensors, total_bytes) for ``key``'s group to observability."""
    grp = get_group(key)
    if grp is None:
        return
    offloaded = getattr(grp, "offloaded_tensors", None) or ()
    if not offloaded:
        return
    total_bytes = 0
    for tw in offloaded:
        buf = getattr(tw, "cpu_buffer", None)
        if buf is None:
            continue
        try:
            total_bytes += buf.numel() * buf.element_size()
        except Exception:
            # ``buf`` may be a plain bytes-like in fallback paths.
            try:
                total_bytes += len(buf)
            except Exception:
                pass
    from megatron.plugin.fl_offload.observability import record_microbatch_offload

    record_microbatch_offload(len(offloaded), total_bytes)


class _BackwardMicrobatchContext:
    """Wrap one microbatch's backward in an ``OnloadAsync``.

    Onload runs synchronously on ``__enter__`` so that the autograd
    engine sees fully-restored tensors when it dispatches the unpack
    hooks.  Like the forward path we pre-issue every stage before
    ``OnloadAsync.__exit__`` so the degradation hook does not trip.
    """

    def __init__(self, key: Any, stages: int) -> None:
        self.key = key
        self.stages = stages
        self._prev_backward_key: Optional[Any] = None

    def __enter__(self) -> "_BackwardMicrobatchContext":
        global _CURRENT_BACKWARD_KEY
        self._prev_backward_key = _CURRENT_BACKWARD_KEY
        _CURRENT_BACKWARD_KEY = self.key
        grp = get_group(self.key)
        _debug_keys(
            f"bwd lookup key={self.key} "
            f"{'hit' if grp is not None else 'MISS'} "
            f"offloaded={len(getattr(grp, 'offloaded_tensors', ()) or ())}"
        )
        onload_ctx = OnloadAsync(self.key, stages=self.stages)
        onload_ctx.__enter__()
        for stage_id in range(onload_ctx.stages):
            onload_ctx.issue(stage_id)
        onload_ctx.__exit__(None, None, None)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        global _CURRENT_BACKWARD_KEY
        _CURRENT_BACKWARD_KEY = self._prev_backward_key
        return False


# ---------------------------------------------------------------------- #
# Schedule-facing façade                                                 #
# ---------------------------------------------------------------------- #
class PipelineActivationOffloadRuntime:
    """Schedule-facing façade.

    All schedule wrappers go through these two context-manager methods.
    When the plugin is disabled (default config) or the caller passes
    ``enabled=False`` for a specific microbatch (e.g. the last PP rank /
    last model chunk), the methods return ``contextlib.nullcontext()``
    so the schedule path is bit-exact against the no-plugin baseline.
    """

    def enabled(self) -> bool:
        return bool(get_config().enable)

    def _stages(self) -> int:
        return max(1, int(get_config().stages))

    def forward_microbatch(
        self,
        *,
        phase: str,
        virtual_microbatch_id: int,
        model_chunk_id: int,
        enabled: bool = True,
        offload_key: Optional[Any] = None,
    ) -> contextlib.AbstractContextManager:
        del phase  # advisory only; reserved for observability in commit 8
        if not enabled or not self.enabled():
            return contextlib.nullcontext()
        key = (
            offload_key
            if offload_key is not None
            else _make_key(virtual_microbatch_id, model_chunk_id)
        )
        return _ForwardMicrobatchContext(key, self._stages())

    def backward_microbatch(
        self,
        *,
        phase: str,
        virtual_microbatch_id: int,
        model_chunk_id: int,
        enabled: bool = True,
        offload_key: Optional[Any] = None,
    ) -> contextlib.AbstractContextManager:
        del phase
        if not enabled or not self.enabled():
            return contextlib.nullcontext()
        key = (
            offload_key
            if offload_key is not None
            else _make_key(virtual_microbatch_id, model_chunk_id)
        )
        return _BackwardMicrobatchContext(key, self._stages())


_RUNTIME = PipelineActivationOffloadRuntime()


def get_pipeline_offload_runtime() -> PipelineActivationOffloadRuntime:
    """Return the singleton runtime façade."""
    return _RUNTIME


__all__ = [
    "OffloadAsync",
    "OnloadAsync",
    "PipelineActivationOffloadRuntime",
    "TensorPack",
    "TensorWrap",
    "byte_view",
    "fast_contiguous",
    "get_group",
    "get_pipeline_offload_runtime",
    "has_group",
    "pop_group",
    "register_group",
    "_make_key",
]
