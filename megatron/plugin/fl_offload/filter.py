"""Saved-tensor eligibility filter — **legacy** (saved_tensors_hooks path).

``is_tensor_eligible`` was the gate for the commit-7 collection channel,
which installed ``saved_tensors_hooks`` and had to *blindly* judge every
tensor autograd saved (hence the conservative shape/contiguity/leaf
heuristics).  Commit 7.2 replaced that channel with explicit
``pack_hook(t, op_name=...)`` calls inside patched autograd Functions,
which use their own minimal gate (``hooks._should_collect``): a producer
that names a tensor has already decided it is an activation worth
offloading, so the leaf/contiguity/rope heuristics here would only cause
false rejections (e.g. MoE ``torch.split`` inputmats are leaf +
requires_grad + sometimes non-contiguous, all legitimate).

This function is retained only for the dormant blind-collection path and
its unit tests; the live offload path does not call it.

Five short-circuit rules in order:

1. Must be a :class:`torch.Tensor` on the accelerator, not a Parameter.
2. Leaf tensors with ``requires_grad=True`` are skipped.
3. Unless ``cfg.non_contiguous`` is set, views / non-contiguous storage /
   non-zero ``storage_offset`` are skipped.
4. Tensors smaller than ``cfg.min_bytes`` are skipped.
5. The ``(N, 1, 1, K)`` 4-D broadcast/mask heuristic is skipped.
"""

from typing import Any

import torch


def _nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def is_tensor_eligible(tensor: Any, cfg) -> bool:
    """**Legacy** blind-collection gate (see module docstring).

    Not called by the commit-7.2 explicit-pack path; retained for the
    dormant ``saved_tensors_hooks`` channel and its tests.

    ``cfg`` is a :class:`FlOffloadConfig` (duck-typed so unit tests can
    pass a ``SimpleNamespace``).
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
