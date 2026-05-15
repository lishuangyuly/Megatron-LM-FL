"""Saved-tensor eligibility filter (commit 3 placeholder).

Will host ``is_tensor_eligible(tensor, cfg)`` implementing the five short-
circuit rules:

1. reject anything that is not a CUDA ``torch.Tensor`` or is an ``nn.Parameter``;
2. reject leaf tensors that require grad;
3. reject views / non-contiguous storage unless ``cfg.non_contiguous`` is set;
4. reject tensors smaller than ``cfg.min_bytes``;
5. reject the ``dim==4 and shape[1]==shape[2]==1`` heuristic.

Empty in commit 1.
"""
