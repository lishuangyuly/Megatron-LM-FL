"""Explicit activation pack/unpack for fl-offload (commit 7.2, dcu-style).

Collection is driven by **patched autograd Functions** calling these
helpers directly — *not* by ``torch.autograd.graph.saved_tensors_hooks``.
The earlier hooks-based approach (commit 7) is gone: it could not keep
weight Parameters intact (torch rewraps the unpack result, breaking TE's
fused-wgrad protocol) and its shape filter missed >95% of activations.

The two public entry points:

* :func:`pack_hook` / :func:`unpack_hook` — called from a patched TE /
  swiglu Function.  ``forward`` hands ``pack_hook`` the activation it is
  dropping from ``save_for_backward`` plus an ``op_name``; ``backward``
  calls ``unpack_hook`` to get it back (reloaded if it was offloaded).
  Weights never pass through here, so they keep their Parameter identity
  and attributes.
* :func:`record` — a context manager that, for one microbatch's forward,
  binds the module-level collection list so those ``pack_hook`` calls
  land in this microbatch's :class:`~megatron.plugin.fl_offload.group.ActivationGroup`,
  registered under the caller-supplied ``key`` on exit.

Nesting is safe: each ``record`` saves the outer collection on entry
and restores it on exit, so an inner ``record`` does not leak into the
outer group.

This module deliberately knows nothing about Megatron schedules — the
microbatch façade in :mod:`runtime` is what stitches forward / backward
contexts together.
"""

import contextlib
from typing import Any, Dict, List, Optional

import torch

from megatron.plugin.fl_offload.config import get_config
from megatron.plugin.fl_offload.group import ActivationGroup
from megatron.plugin.fl_offload.runtime import (
    TensorPack,
    TensorWrap,
    register_group,
)


# Module-level "current collection".  ``None`` means "no record() is
# active right now; pack_hook is a transparent wrapper without offload
# bookkeeping".  Single-threaded by construction (Megatron training is
# single-threaded per process).
_OFFLOAD_TENSORS: Optional[List[TensorWrap]] = None


def pack_hook(x: Any, op_name: Optional[str] = None) -> Optional[TensorPack]:
    """Explicitly pack a saved activation for offload (commit 7.2).

    Called **directly from a patched autograd Function's forward**, not
    from ``saved_tensors_hooks`` — the caller hands us the tensor it is
    about to drop from ``save_for_backward`` along with the ``op_name``
    identifying the producing op (e.g. ``"GroupedLinear"`` / ``"swiglu"``).

    Collection happens iff :func:`_should_collect` passes (a record is
    active, op_name allowed, plain non-Parameter tensor, >= min_bytes).
    A quantized wrapper (Float8Tensor etc.) is still wrapped + returned
    so backward's ``unpack_hook`` gets it back unchanged, but is not
    offloaded (fp8 activation offload is a separate unsolved problem;
    see plan 7.2 fp8 边界).

    Always returns a :class:`TensorPack` (even when not collected) so the
    patched backward can uniformly call ``unpack_hook``.
    """
    if x is None:
        return None
    tw = TensorWrap(x)
    collected = _OFFLOAD_TENSORS is not None and _should_collect(x, op_name)
    if collected:
        _OFFLOAD_TENSORS.append(tw)
    _reconcile_pack(x, op_name, collected)
    _debug_pack(x, op_name, collected)
    return TensorPack(tw, op_name=op_name)


# ---------------------------------------------------------------------- #
# Collection-completeness reconciliation (FL_OFFLOAD_RECON=1)            #
# ---------------------------------------------------------------------- #
# Per op_name tally proving every packed activation is later unpacked,
# and that every *skipped* tensor was skipped for a legitimate reason
# (empty expert / quantized / too-small / not-cuda / no-record).  A run
# is "complete" iff, per op_name, pack_total == unpack_total and the
# skip breakdown sums to (pack_total - collected).
class _ReconState:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        # op_name -> dict(counts)
        self.by_op: Dict[str, Dict[str, int]] = {}

    def _op(self, op_name) -> Dict[str, int]:
        key = op_name or "<none>"
        return self.by_op.setdefault(
            key,
            dict(pack=0, collected=0, unpack=0,
                 skip_empty=0, skip_quant=0, skip_small=0,
                 skip_notcuda=0, skip_norecord=0, skip_param=0, skip_op=0),
        )


_RECON = _ReconState()


def _recon_enabled() -> bool:
    import os

    return os.environ.get("FL_OFFLOAD_RECON") == "1"


def _reconcile_pack(x: Any, op_name, collected: bool) -> None:
    if not _recon_enabled():
        return
    rec = _RECON._op(op_name)
    rec["pack"] += 1
    if collected:
        rec["collected"] += 1
        return
    # Classify the skip reason (mirrors _should_collect order).
    if _OFFLOAD_TENSORS is None:
        rec["skip_norecord"] += 1
    elif not isinstance(x, torch.Tensor):
        rec["skip_notcuda"] += 1
    elif x.device.type != "cuda":
        rec["skip_notcuda"] += 1
    elif isinstance(x, torch.nn.Parameter):
        rec["skip_param"] += 1
    elif not _is_plain_tensor(x):
        rec["skip_quant"] += 1
    else:
        cfg = get_config()
        allow = getattr(cfg, "offload_modules", None) or []
        if allow and op_name not in allow:
            rec["skip_op"] += 1
        elif x.numel() == 0:
            rec["skip_empty"] += 1
        else:
            rec["skip_small"] += 1


def _reconcile_unpack(op_name) -> None:
    if not _recon_enabled():
        return
    _RECON._op(op_name)["unpack"] += 1


def report_reconciliation() -> None:
    """Print the per-op pack/unpack reconciliation (FL_OFFLOAD_RECON=1).

    Called once per training step from the schedule wrapper.  Completeness
    holds when, for every op_name: ``pack == unpack`` (every packed
    activation was recovered in backward) and
    ``collected + Σskip_* == pack`` (every tensor accounted for).
    """
    if not _recon_enabled() or not _RECON.by_op:
        return
    from megatron.plugin.fl_offload.observability import _rank_tag

    tag = _rank_tag()
    for op, r in sorted(_RECON.by_op.items()):
        skips = (r["skip_empty"] + r["skip_quant"] + r["skip_small"]
                 + r["skip_notcuda"] + r["skip_norecord"]
                 + r["skip_param"] + r["skip_op"])
        balanced = "OK" if r["pack"] == r["unpack"] else "MISMATCH"
        accounted = "OK" if r["collected"] + skips == r["pack"] else "LEAK"
        print(
            f"[fl-offload][RECON][{tag}] op={op} pack={r['pack']} "
            f"unpack={r['unpack']}({balanced}) collected={r['collected']} "
            f"skip[empty={r['skip_empty']} quant={r['skip_quant']} "
            f"small={r['skip_small']} notcuda={r['skip_notcuda']} "
            f"norecord={r['skip_norecord']} param={r['skip_param']} "
            f"op={r['skip_op']}] accounted={accounted}",
            flush=True,
        )
    _RECON.reset()


_debug_pack_printed = 0
_DEBUG_PACK_LIMIT = 40


def _debug_pack(x: Any, op_name, collected: bool) -> None:
    """One-shot-ish trace of pack_hook decisions (FL_OFFLOAD_DEBUG_PACK=1)."""
    global _debug_pack_printed
    import os

    if os.environ.get("FL_OFFLOAD_DEBUG_PACK") != "1":
        return
    if _debug_pack_printed >= _DEBUG_PACK_LIMIT:
        return
    _debug_pack_printed += 1
    cfg = get_config()
    in_record = _OFFLOAD_TENSORS is not None
    is_t = isinstance(x, torch.Tensor)
    nbytes = (x.numel() * x.element_size()) if is_t else 0
    min_bytes = getattr(cfg, "min_bytes", 0)
    leaf_rg = is_t and x.is_leaf and x.requires_grad
    reason = "collected" if collected else (
        "no-record" if not in_record else
        "not-tensor" if not is_t else
        "not-plain" if not _is_plain_tensor(x) else
        "op-not-allowed" if (getattr(cfg, "offload_modules", None)
                             and op_name not in cfg.offload_modules) else
        "leaf-requires-grad" if leaf_rg else
        f"under-min-bytes({nbytes}<{min_bytes})" if nbytes < min_bytes else
        "filter-reject-other"
    )
    dev = getattr(x, "device", None)
    shp = tuple(getattr(x, "shape", ()) or ())
    contig = x.is_contiguous() if is_t else None
    print(
        f"[fl-offload][PACK] op={op_name} in_record={in_record} "
        f"type={type(x).__name__} dev={dev} shape={shp} nbytes={nbytes} "
        f"contig={contig} -> {reason}",
        flush=True,
    )


def _is_plain_tensor(x: Any) -> bool:
    """True for an ordinary ``torch.Tensor`` (not a quantized subclass).

    fp8 / quantized activations are intentionally excluded — their
    storage layout carries scale_inv / transpose metadata that a flat
    byte-copy offload would corrupt, and we do not yet reconstruct them
    on reload (plan 7.2 fp8 边界).  We detect "plain" by exact type or
    the absence of TE quantizer attributes.
    """
    if not isinstance(x, torch.Tensor):
        return False
    # TE quantized tensors carry these; plain bf16/fp32 tensors do not.
    for attr in ("_quantizer", "_fp8_dtype", "_scale_inv", "_data"):
        if hasattr(x, attr):
            return False
    return True


def _should_collect(x: Any, op_name: Optional[str]) -> bool:
    """Eligibility gate for the explicit-pack path (commit 7.2, dcu-style).

    Independent of the legacy ``filter.is_tensor_eligible`` — a producer
    that calls ``pack_hook(t, op_name=...)`` has already decided ``t`` is
    an activation worth offloading, so we only apply the gates that still
    matter:

    * ``op_name`` is in the configured allowlist (empty = all enabled);
    * ``x`` is a plain CUDA ``torch.Tensor`` — CPU tensors (offload is a
      no-op) and quantized wrappers (Float8Tensor etc., fp8 offload
      unsolved, plan 7.2) are skipped;
    * ``x`` is not a Parameter (weights must never offload — a safety
      net; patches never pack weights);
    * ``x`` meets ``min_bytes`` and is non-empty (a 0-token MoE expert
      yields an empty tensor — nothing to offload, and it would trip the
      pinned pool's ``num_bytes > 0`` guard).

    Notably **not** applied: the leaf/requires_grad, contiguity/view, and
    rope-shape heuristics — those were blind-collection guards and would
    wrongly drop legitimate explicit activations (e.g. MoE ``torch.split``
    inputmats, which are leaf + requires_grad + sometimes non-contiguous).
    ``ActivationGroup.offload_prologue`` makes non-contiguous tensors
    contiguous before the D2H copy, so view-ness is not a correctness
    issue.
    """
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "cuda":
        return False
    if isinstance(x, torch.nn.Parameter):
        return False
    if not _is_plain_tensor(x):
        return False
    cfg = get_config()
    allow = getattr(cfg, "offload_modules", None) or []
    if allow and op_name not in allow:
        return False
    nbytes = x.numel() * x.element_size()
    if nbytes == 0:
        # Empty tensors (e.g. a MoE expert that received 0 routed tokens)
        # have nothing to offload and would trip the pinned-pool's
        # ``num_bytes > 0`` guard.
        return False
    if nbytes < getattr(cfg, "min_bytes", 0):
        return False
    return True


def unpack_hook(tensor_pack: Optional[TensorPack]) -> Optional[torch.Tensor]:
    """Return the (possibly onloaded) tensor backing ``tensor_pack``.

    Called explicitly from a patched Function's backward.  When the
    activation was offloaded, ``TensorPack.get`` (via the runtime) yields
    the reloaded GPU tensor; otherwise it returns the original.
    """
    if tensor_pack is None:
        return None
    _reconcile_unpack(getattr(tensor_pack, "op_name", None))
    return tensor_pack.get()


@contextlib.contextmanager
def record(key: Any, group_num: int = 1):
    """Open a collection window for one microbatch's offload group.

    Unlike commit 7 this does **not** install ``saved_tensors_hooks`` —
    collection is driven by patched autograd Functions calling
    :func:`pack_hook` directly during their forward (commit 7.2, dcu
    style).  ``record`` merely binds the module-level ``_OFFLOAD_TENSORS``
    list for the duration of the ``with`` so those ``pack_hook`` calls
    append into this microbatch's group, then registers the group under
    ``key`` on exit.

    Always registers a group — even an empty one — so the backward path
    can rely on ``get_group(key)`` returning something predictable.
    """
    global _OFFLOAD_TENSORS
    previous = _OFFLOAD_TENSORS
    _OFFLOAD_TENSORS = []
    try:
        yield
    finally:
        collected = _OFFLOAD_TENSORS
        _OFFLOAD_TENSORS = previous
        register_group(
            key,
            ActivationGroup(collected, key, max(1, int(group_num))),
        )


def current_collection() -> Optional[List[TensorWrap]]:
    """Return the active record's collection (None when no record is active).

    Provided as a testing aid; production code does not need it.
    """
    return _OFFLOAD_TENSORS


def _reset_state_for_tests() -> None:
    """Force ``_OFFLOAD_TENSORS`` back to ``None``. Tests only."""
    global _OFFLOAD_TENSORS
    _OFFLOAD_TENSORS = None


__all__ = [
    "current_collection",
    "pack_hook",
    "record",
    "unpack_hook",
]
