"""``ActivationGroup`` — one microbatch's offload / onload pipeline.

Lifecycle from the schedule's point of view:

1. ``record()`` (commit 4) collects eligible ``TensorWrap`` instances and
   constructs an ``ActivationGroup`` for the microbatch, registering it
   under a stable key.
2. ``OffloadAsync(key).__enter__`` calls :meth:`offload_prologue` (sort +
   budget + bucket + CPU buffer alloc), then ``OffloadAsync.issue(s)``
   calls :meth:`offload_issue` once per stage to launch D2H copies on the
   ``offload`` stream.
3. ``OffloadAsync.__exit__`` flushes any unissued stages and calls
   :meth:`offload_epilogue`, which waits for the offload stream and drops
   the GPU references (``tensor_wrap.x = None``).
4. The backward path runs ``OnloadAsync(key)`` symmetrically, ending in
   :meth:`onload_epilogue` that returns buffers to the pool.

This module deliberately knows nothing about ``saved_tensors_hooks`` —
that wiring lives in :mod:`megatron.plugin.fl_offload.hooks` (commit 4).
"""

import math
from typing import Any, List

import torch

from megatron.plugin.fl_offload.config import get_config
from megatron.plugin.fl_offload.pool import get_global_pool
from megatron.plugin.fl_offload.runtime import (
    TensorWrap,
    byte_view,
    fast_contiguous,
)
from megatron.plugin.fl_offload.stream import (
    current_stream,
    get_memcpy_stream,
    stream_scope,
)


def _nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


class CopyTaskGroup:
    """Round-robin bucket assignment of TensorWraps to copy stages."""

    def __init__(self, stages: int) -> None:
        self.stages = max(1, int(stages))
        self.buckets: List[List[TensorWrap]] = [[] for _ in range(self.stages)]

    def add(self, idx: int, tensor_wrap: TensorWrap) -> None:
        """Place ``tensor_wrap`` into bucket ``idx % stages``."""
        self.buckets[idx % self.stages].append(tensor_wrap)

    def get_buckets(self) -> List[List[TensorWrap]]:
        return self.buckets


class ActivationGroup:
    """Manages offload + onload for one microbatch's saved activations."""

    def __init__(self, tensors: List[TensorWrap], key: Any, stages: int = 1) -> None:
        # Sort so contiguous + large tensors are considered first by the
        # budget loop.  Sort key is documented in the plan: prefer
        # contiguous, prefer larger.
        self.tensors: List[TensorWrap] = sorted(
            tensors,
            key=lambda tw: (
                False if (tw.x is not None and tw.x.is_contiguous()) else True,
                -(tw.x.numel() if tw.x is not None else 0),
            ),
        )
        self.key = key
        self.stages = max(1, int(stages))
        for tw in self.tensors:
            tw.owner_key = key

        # State that grows during ``offload_prologue`` and clears in
        # ``onload_epilogue``.
        self.offloaded_tensors: List[TensorWrap] = []
        self.cpu_buffers: List[torch.Tensor] = []
        self.copy_buckets: List[List[TensorWrap]] = []

        # The onload stream may differ from the offload stream when the
        # platform supports independent D2H / H2D pipelines.
        self._onload_stream = None

    # ------------------------------------------------------------------ #
    # Budget                                                              #
    # ------------------------------------------------------------------ #
    def _total_bytes(self) -> int:
        return sum(_nbytes(tw.x) for tw in self.tensors if tw.x is not None)

    def _offload_budget(self) -> int:
        cfg = get_config()
        total = self._total_bytes()
        per_batch_size = float(getattr(cfg, "per_batch_size", 0.0) or 0.0)
        if per_batch_size > 0.0:
            return min(total, int(per_batch_size * (2 ** 20)))
        ratio = float(getattr(cfg, "ratio", 1.0) or 0.0)
        return int(math.ceil(total * max(0.0, min(1.0, ratio))))

    # ------------------------------------------------------------------ #
    # Offload                                                             #
    # ------------------------------------------------------------------ #
    def offload_prologue(self) -> None:
        if not self.tensors:
            self.copy_buckets = []
            return

        budget = self._offload_budget()
        copied = 0
        pool = get_global_pool()
        buckets = CopyTaskGroup(self.stages)

        for tw in self.tensors:
            tensor = tw.x
            if tensor is None:
                continue
            n = _nbytes(tensor)
            if copied + n > budget:
                # Budget exhausted — skip this and any remaining tensors
                # (they are all >= this size after the sort).  We still
                # try the rest because tensors after the cut may be
                # *smaller* (the sort key is ``(non_contiguous, -numel)``,
                # so within each contiguity bucket sizes decrease).
                continue
            if not tensor.is_contiguous():
                tw.x = fast_contiguous(tensor)
                tensor = tw.x
            cpu_buf = pool.pin_get(n)
            tw.cpu_buffer = cpu_buf
            self.cpu_buffers.append(cpu_buf)
            self.offloaded_tensors.append(tw)
            buckets.add(len(self.offloaded_tensors) - 1, tw)
            copied += n

        self.copy_buckets = buckets.get_buckets()

    def offload_issue(self, stage_id: int) -> None:
        if not self.copy_buckets:
            return
        if stage_id < 0 or stage_id >= self.stages:
            raise IndexError(
                f"offload_issue: stage_id {stage_id} out of range "
                f"[0, {self.stages})"
            )

        stream = get_memcpy_stream("offload")
        cur = current_stream()
        if stream is not None and cur is not None:
            stream.wait_stream(cur)
        with stream_scope(stream):
            for tw in self.copy_buckets[stage_id]:
                if tw.x is None or tw.cpu_buffer is None:
                    continue
                src = byte_view(tw.x)
                # ``non_blocking=True`` is harmless on CPU-only (no stream).
                tw.cpu_buffer.copy_(src, non_blocking=stream is not None)

    def offload_epilogue(self) -> None:
        if not self.copy_buckets:
            return
        stream = get_memcpy_stream("offload")
        cur = current_stream()
        if stream is not None and cur is not None:
            cur.wait_stream(stream)
        # Diagnostic bisect switch: run the full offload machinery but
        # keep the GPU tensors resident, so backward consumes the
        # *original* objects.  If a failure persists with this on, the
        # cause is the hook/context machinery; if it disappears, the
        # cause is the restored-copy substitution.
        import os

        if os.environ.get("FL_OFFLOAD_DIAG_NO_FREE") == "1":
            return
        # Drop only the saved-tensor hook reference.  The same Tensor
        # object may also be a pipeline boundary tensor in scheduler-
        # scoped hooks; resizing its storage would corrupt that live
        # alias before send/backward uses it.
        for tw in self.offloaded_tensors:
            tw.x = None

    # ------------------------------------------------------------------ #
    # Onload                                                              #
    # ------------------------------------------------------------------ #
    def onload_prologue(self) -> None:
        if not self.offloaded_tensors:
            return
        self._onload_stream = get_memcpy_stream("onload")
        cur = current_stream()
        if self._onload_stream is not None and cur is not None:
            self._onload_stream.wait_stream(cur)

    def onload_issue(self, stage_id: int) -> None:
        if not self.copy_buckets:
            return
        if stage_id < 0 or stage_id >= self.stages:
            raise IndexError(
                f"onload_issue: stage_id {stage_id} out of range "
                f"[0, {self.stages})"
            )

        stream = self._onload_stream
        with stream_scope(stream):
            for tw in self.copy_buckets[stage_id]:
                if tw.x is not None:
                    # Still resident (e.g. FL_OFFLOAD_DIAG_NO_FREE kept
                    # it alive) — nothing to restore.
                    continue
                if tw.cpu_buffer is None:
                    continue
                new_tensor = torch.empty(
                    tw.shape, dtype=tw.dtype, device=tw.device
                )
                byte_view(new_tensor).copy_(
                    tw.cpu_buffer, non_blocking=stream is not None
                )
                tw.x = new_tensor

    def onload_epilogue(self) -> None:
        if not self.offloaded_tensors:
            return
        cur = current_stream()
        if self._onload_stream is not None and cur is not None:
            cur.wait_stream(self._onload_stream)
        pool = get_global_pool()
        for buf in self.cpu_buffers:
            pool.pin_put(buf)
        # Reset per-microbatch state so repeated use of the same group
        # would fail loudly rather than silently double-onload.
        self.cpu_buffers = []
        self.offloaded_tensors = []
        self.copy_buckets = []


__all__ = ["ActivationGroup", "CopyTaskGroup"]
