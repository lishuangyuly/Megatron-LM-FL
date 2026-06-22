"""Explicit-pack patch for TE ``_GroupedLinear`` (commit 7.2).

MoE expert GEMM — the largest activation source under EP.  We divert the
per-gemm ``inputmats`` from ``save_for_backward`` to ``pack_hook`` in
forward and recover them via ``unpack_hook`` in backward, leaving
weights/biases on the native save path (so fused-wgrad stays intact).

Save layout (TE 2.10.0, rev 3c34bb9a), each group N=num_gemms wide::

    prepare_for_saving(*inputmats, *weights_fp8, *weights, *biases)
    backward: saved[:N]=inputmats  [N:2N]=weights_fp8(→weights)
              [2N:3N]=weights(→origin_weights)  [3N:4N]=biases

After diverting inputmats the save list loses its first N slots, so the
three remaining slices shift by -N (verified against dcu's
``offload_offset = -N``).  Anchors are guarded in ``_patch_util`` — a TE
source change that moves them raises rather than mis-slicing silently.
"""

from __future__ import annotations

from megatron.plugin.fl_offload.te_patch._patch_util import (
    FunctionPatch,
    _Substitution,
)


_FORWARD_ANCHOR = """\
        tensors_to_save, tensor_objects = prepare_for_saving(
            *inputmats,
            *weights_fp8,
            *weights,
            *biases,
        )"""

_FORWARD_REPLACEMENT = """\
        from megatron.plugin.fl_offload.hooks import pack_hook as _fl_pack
        from megatron.plugin.fl_offload.te_patch.grouped_linear import _fl_audit_forward
        ctx.fl_tensor_packs = [_fl_pack(t, op_name="GroupedLinear") for t in inputmats]
        _fl_audit_forward(inputmats, weights_fp8, weights, biases)
        tensors_to_save, tensor_objects = prepare_for_saving(
            *weights_fp8,
            *weights,
            *biases,
        )"""

_BACKWARD_ANCHOR = """\
        inputmats = saved_tensors[:N]
        weights = saved_tensors[N : 2 * N]
        origin_weights = saved_tensors[2 * N : 3 * N]
        biases = saved_tensors[3 * N : 4 * N]"""

_BACKWARD_REPLACEMENT = """\
        from megatron.plugin.fl_offload.hooks import unpack_hook as _fl_unpack
        inputmats = [_fl_unpack(p) for p in ctx.fl_tensor_packs]
        weights = saved_tensors[0 : N]
        origin_weights = saved_tensors[N : 2 * N]
        biases = saved_tensors[2 * N : 3 * N]"""


def build_patch() -> FunctionPatch:
    """Return the :class:`FunctionPatch` for ``_GroupedLinear`` (lazy import)."""
    from transformer_engine.pytorch.module.grouped_linear import _GroupedLinear

    return FunctionPatch(
        function_cls=_GroupedLinear,
        forward_subs=[_Substitution(_FORWARD_ANCHOR, _FORWARD_REPLACEMENT)],
        backward_subs=[_Substitution(_BACKWARD_ANCHOR, _BACKWARD_REPLACEMENT)],
    )


# --- audit (FL_OFFLOAD_AUDIT=1) ------------------------------------- #
# Proves, per GroupedLinear.forward call, that the ONLY activation it
# saves is ``inputmats`` (all of which we pack), and that
# weights_fp8/weights/biases are parameters we deliberately leave on the
# native save path.  Confirms "all GroupedLinear activations offloaded".
_audit_calls = 0
_AUDIT_LIMIT = 24


def _bytes(seq) -> int:
    total = 0
    for t in seq:
        if t is not None and hasattr(t, "numel"):
            try:
                total += t.numel() * t.element_size()
            except Exception:
                pass
    return total


def _fl_audit_forward(inputmats, weights_fp8, weights, biases) -> None:
    global _audit_calls
    import os

    if os.environ.get("FL_OFFLOAD_AUDIT") != "1" or _audit_calls >= _AUDIT_LIMIT:
        return
    _audit_calls += 1
    from megatron.plugin.fl_offload.observability import _rank_tag

    n = len(inputmats)
    act_b = _bytes(inputmats)              # activations — should all offload
    param_b = _bytes(weights) + _bytes(biases)  # parameters — never offload
    nonempty = sum(
        1 for t in inputmats
        if t is not None and hasattr(t, "numel") and t.numel() > 0
    )
    print(
        f"[fl-offload][AUDIT][{_rank_tag()}] GroupedLinear num_gemms={n} "
        f"inputmats_nonempty={nonempty} act_bytes={act_b/(1<<20):.2f}MiB "
        f"param_bytes={param_b/(1<<20):.2f}MiB "
        f"(activations packed; params on native save)",
        flush=True,
    )


__all__ = ["build_patch"]
