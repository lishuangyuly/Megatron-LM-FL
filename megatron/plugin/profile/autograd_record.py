"""
Backward-pass record_function via custom autograd Functions.

Forward and backward graphs run in opposite order. To put a
``record_function`` range around a segment of the backward pass, we plant
two ``autograd.Function`` nodes in the forward graph: one at the forward
input side (becomes the *exit* of the backward range) and one at the
forward output side (becomes the *entry* of the backward range). They
share a single-element list ``rf_holder`` so the entry node can hand the
active context manager to the exit node.

See profile_docs/design.md §2.3 for the full rationale.
"""

from typing import Callable, List, Tuple

import torch
from torch.profiler import record_function

from .core import _current_device_name, is_profile_enabled


def _anchor_kernel():
    """Emit a tiny fill_(0) on the current stream to pin a CPU range to GPU.

    Mirrors the forward-side gpu_anchor: a near-zero-cost compute kernel that
    PyTorch profiler can hang gpu_user_annotation onto. Use inside autograd
    backward methods where the actual work is NCCL / async comm that doesn't
    show up on the user's stream track.
    """
    torch.empty(1, device=_current_device_name()).fill_(0)


class BWDRecordStart(torch.autograd.Function):
    """Plant at the forward INPUT side. Backward = exit the range."""

    @staticmethod
    def forward(ctx, *args):
        *tensors, rf_holder = args
        ctx.rf_holder = rf_holder
        if len(tensors) == 1:
            return tensors[0].view_as(tensors[0])
        return tuple(t.view_as(t) for t in tensors)

    @staticmethod
    def backward(ctx, *grad_outputs):
        rf = ctx.rf_holder[0]
        if rf is not None:
            rf.__exit__(None, None, None)
            ctx.rf_holder[0] = None
        # one grad per forward input tensor, plus None for rf_holder
        return (*grad_outputs, None)


class BWDRecordEnd(torch.autograd.Function):
    """Plant at the forward OUTPUT side. Backward = enter the range."""

    @staticmethod
    def forward(ctx, *args):
        *tensors, name, rf_holder = args
        ctx.name = name
        ctx.rf_holder = rf_holder
        if len(tensors) == 1:
            return tensors[0].view_as(tensors[0])
        return tuple(t.view_as(t) for t in tensors)

    @staticmethod
    def backward(ctx, *grad_outputs):
        rf = record_function(ctx.name)
        rf.__enter__()
        ctx.rf_holder[0] = rf
        # one grad per forward input tensor, plus None for name and rf_holder
        return (*grad_outputs, None, None)


def _identity_start(*tensors):
    if len(tensors) == 1:
        return tensors[0]
    return tensors


def _identity_end(*args):
    *tensors, _name = args
    if len(tensors) == 1:
        return tensors[0]
    return tuple(tensors)


def bwd_record_pair() -> Tuple[Callable, Callable]:
    """Return ``(start, end)`` for a backward record_function pair.

    Usage::

        start, end = bwd_record_pair()
        x = start(x)                     # plant exit-of-backward node
        # ... forward computation ...
        y = end(y, "mcfl: phase=backward&func=...")   # plant entry-of-backward node

    When disabled both ``start`` and ``end`` are identity callables, so no
    extra autograd nodes enter the graph and there is no view overhead.
    The ``rf_holder`` is captured in closure; the caller never sees it.
    """
    rf_holder: List = [None]

    if not is_profile_enabled():
        return _identity_start, _identity_end

    def start(*tensors):
        return BWDRecordStart.apply(*tensors, rf_holder)

    def end(*args):
        *tensors, name = args
        return BWDRecordEnd.apply(*tensors, name, rf_holder)

    return start, end


class BWDRecordStartAnchored(torch.autograd.Function):
    """Like ``BWDRecordStart``, plus a GPU anchor kernel right before exit.

    Use when the surrounded backward segment is comm-heavy (NCCL all-to-all
    etc.) and PyTorch profiler's gpu_user_annotation cannot reach the actual
    NCCL stream. The anchor's fill_(0) lands on the autograd-engine-current
    stream and gives the profiler a kernel to attach the annotation to.
    """

    @staticmethod
    def forward(ctx, *args):
        *tensors, rf_holder = args
        ctx.rf_holder = rf_holder
        if len(tensors) == 1:
            return tensors[0].view_as(tensors[0])
        return tuple(t.view_as(t) for t in tensors)

    @staticmethod
    def backward(ctx, *grad_outputs):
        _anchor_kernel()
        rf = ctx.rf_holder[0]
        if rf is not None:
            rf.__exit__(None, None, None)
            ctx.rf_holder[0] = None
        return (*grad_outputs, None)


class BWDRecordEndAnchored(torch.autograd.Function):
    """Like ``BWDRecordEnd``, plus a GPU anchor kernel right after entry."""

    @staticmethod
    def forward(ctx, *args):
        *tensors, name, rf_holder = args
        ctx.name = name
        ctx.rf_holder = rf_holder
        if len(tensors) == 1:
            return tensors[0].view_as(tensors[0])
        return tuple(t.view_as(t) for t in tensors)

    @staticmethod
    def backward(ctx, *grad_outputs):
        rf = record_function(ctx.name)
        rf.__enter__()
        _anchor_kernel()
        ctx.rf_holder[0] = rf
        return (*grad_outputs, None, None)


def bwd_record_pair_with_anchor() -> Tuple[Callable, Callable]:
    """Same shape as ``bwd_record_pair`` but emits a GPU anchor kernel at
    both backward enter and exit.

    Pick this over ``bwd_record_pair`` when the wrapped segment's backward
    runs primarily on a stream the profiler doesn't project user_annotation
    onto (e.g. NCCL comm stream for moe dispatch / combine).
    """
    rf_holder: List = [None]

    if not is_profile_enabled():
        return _identity_start, _identity_end

    def start(*tensors):
        return BWDRecordStartAnchored.apply(*tensors, rf_holder)

    def end(*args):
        *tensors, name = args
        return BWDRecordEndAnchored.apply(*tensors, name, rf_holder)

    return start, end
