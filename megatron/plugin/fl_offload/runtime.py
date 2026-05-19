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
from typing import Any, Dict, Optional

import torch


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

    __slots__ = ("x", "shape", "dtype", "device", "cpu_buffer")

    def __init__(self, tensor: torch.Tensor) -> None:
        self.x: Optional[torch.Tensor] = tensor
        self.shape = tensor.shape
        self.dtype = tensor.dtype
        self.device = tensor.device
        self.cpu_buffer: Optional[torch.Tensor] = None


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
        return self.tensor_wrap.x


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
        while self.issued_stages < self.stages:
            self.group.onload_issue(self.issued_stages)
            self.issued_stages += 1
        self.group.onload_epilogue()
        # Each microbatch's group is single-use; drop it on the way out.
        pop_group(self.key)
        return False


# ---------------------------------------------------------------------- #
# Schedule-facing façade (real bodies arrive in commit 4)                #
# ---------------------------------------------------------------------- #
class PipelineActivationOffloadRuntime:
    """Schedule-facing façade.

    Commit 4 replaces ``forward_microbatch`` / ``backward_microbatch``
    with real ``record()``-based contexts.  Until then both return
    ``contextlib.nullcontext()`` so any caller that runs ahead of the
    real wiring still works.
    """

    def enabled(self) -> bool:
        # Always False in commit 3 — the real implementation in commit 4
        # will read ``get_config().enable``.
        return False

    def forward_microbatch(
        self,
        *,
        phase: str,
        virtual_microbatch_id: int,
        model_chunk_id: int,
        enabled: bool = True,
        offload_key: Optional[Any] = None,
    ) -> contextlib.AbstractContextManager:
        del phase, virtual_microbatch_id, model_chunk_id, enabled, offload_key
        return contextlib.nullcontext()

    def backward_microbatch(
        self,
        *,
        phase: str,
        virtual_microbatch_id: int,
        model_chunk_id: int,
        enabled: bool = True,
        offload_key: Optional[Any] = None,
    ) -> contextlib.AbstractContextManager:
        del phase, virtual_microbatch_id, model_chunk_id, enabled, offload_key
        return contextlib.nullcontext()


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
]
