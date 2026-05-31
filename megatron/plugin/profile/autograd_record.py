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

from .core import is_profile_enabled


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
