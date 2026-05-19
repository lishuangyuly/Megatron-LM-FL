"""Saved-tensor eligibility filter.

A single entry point — :func:`is_tensor_eligible` — applies five
short-circuit rules in order:

1. The candidate must be a :class:`torch.Tensor` living on the accelerator
   and not an :class:`torch.nn.Parameter`.
2. Leaf tensors that participate in autograd (``requires_grad=True``) are
   skipped — they are model parameters or activations re-used elsewhere.
3. Unless ``cfg.non_contiguous`` is set, views / non-contiguous storage /
   non-zero ``storage_offset`` are skipped.  Some fused kernels (e.g.
   FlashAttention) require the original saved-tensor layout in backward.
4. Tensors smaller than ``cfg.min_bytes`` are not worth the copy round
   trip and are skipped.
5. The ``(N, 1, 1, K)`` 4-D heuristic — common for broadcast/mask
   tensors in attention — is skipped.

The filter is pure and intentionally has no side effects.  In commit 4 it
will be called by the saved-tensor ``pack`` hook; for now (commit 3) it is
unit-tested directly.
"""

from typing import Any

import torch


def _nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def is_tensor_eligible(tensor: Any, cfg) -> bool:
    """Return True iff ``tensor`` should be offloaded under ``cfg``.

    ``cfg`` is a :class:`FlOffloadConfig` (the type is intentionally Duck-
    typed here so unit tests can pass a ``SimpleNamespace``).
    """
    # Rule 1: must be a real CUDA tensor, not a Parameter.
    if not isinstance(tensor, torch.Tensor):
        return False
    if tensor.device.type != "cuda":
        return False
    if isinstance(tensor, torch.nn.Parameter):
        return False

    # Rule 2: skip leaf-but-requires-grad (model parameters wrapped as
    # plain tensors, or activations that are reused as graph leaves).
    if tensor.is_leaf and tensor.requires_grad:
        return False

    # Rule 3: contiguity / view check.
    if not getattr(cfg, "non_contiguous", False):
        if not tensor.is_contiguous():
            return False
        if tensor._base is not None:
            return False
        if tensor.storage_offset() != 0:
            return False

    # Rule 4: minimum byte threshold.
    if _nbytes(tensor) < getattr(cfg, "min_bytes", 0):
        return False

    # Rule 5: skip the (N, 1, 1, K) broadcast/mask 4-D heuristic.
    if tensor.dim() == 4 and tensor.shape[1] == 1 and tensor.shape[2] == 1:
        return False

    return True


__all__ = ["is_tensor_eligible"]
