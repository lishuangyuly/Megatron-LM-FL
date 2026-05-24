"""``torch.autograd.graph.saved_tensors_hooks`` adapter for fl-offload.

The two public entry points are:

* :func:`pack_hook` / :func:`unpack_hook` — the actual pair installed
  on the autograd graph.  ``pack_hook`` wraps every saved tensor in a
  :class:`~megatron.plugin.fl_offload.runtime.TensorPack` and, when a
  :func:`record` is active, also appends a :class:`TensorWrap` to that
  record's collection so the surrounding context can offload it.
* :func:`record` — a context manager that, for one microbatch's
  forward, installs the hooks, collects eligible tensors, and at exit
  builds an :class:`~megatron.plugin.fl_offload.group.ActivationGroup`
  registered under the caller-supplied ``key``.

Nesting is safe: each ``record`` saves the outer collection on entry
and restores it on exit, so an inner ``record`` does not leak into the
outer group.

This module deliberately knows nothing about Megatron schedules — the
microbatch façade in :mod:`runtime` is what stitches forward / backward
contexts together.
"""

import contextlib
from typing import Any, List, Optional

import torch

from megatron.plugin.fl_offload.config import get_config
from megatron.plugin.fl_offload.filter import is_tensor_eligible
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
    """Wrap ``x`` so the autograd graph holds a :class:`TensorPack`.

    When a record() is active and ``x`` passes the eligibility filter,
    a fresh :class:`TensorWrap` is also appended to that record's
    collection — the surrounding context will later offload it.

    Returns ``None`` when ``x`` is ``None`` so callers do not need to
    special-case the "no tensor saved" path.
    """
    if x is None:
        return None
    tw = TensorWrap(x)
    if _OFFLOAD_TENSORS is not None and is_tensor_eligible(x, get_config()):
        _OFFLOAD_TENSORS.append(tw)
    return TensorPack(tw, op_name=op_name)


def unpack_hook(tensor_pack: Optional[TensorPack]) -> Optional[torch.Tensor]:
    """Return the (possibly onloaded) tensor backing ``tensor_pack``."""
    if tensor_pack is None:
        return None
    return tensor_pack.get()


@contextlib.contextmanager
def record(key: Any, group_num: int = 1):
    """Capture every saved tensor inside ``with`` under one group.

    On exit the collected :class:`TensorWrap` list is wrapped into an
    :class:`ActivationGroup` and registered under ``key`` via
    :func:`register_group`, so the matching backward path can find and
    onload it.

    Always registers a group — even an empty one — so the backward path
    can rely on ``get_group(key)`` returning something predictable.  An
    empty :class:`ActivationGroup` degrades to a no-op inside
    :class:`OffloadAsync` / :class:`OnloadAsync`.
    """
    global _OFFLOAD_TENSORS
    previous = _OFFLOAD_TENSORS
    _OFFLOAD_TENSORS = []
    try:
        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
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
