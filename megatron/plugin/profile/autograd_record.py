"""Backward record_function ranges planted by paired autograd nodes."""

from typing import Callable, List, Tuple

import torch
from torch.profiler import record_function

from .core import _current_device_name, is_profile_enabled


def _anchor_kernel():
    torch.empty(1, device=_current_device_name()).fill_(0)


class BWDRecordStart(torch.autograd.Function):
    """Plant at the forward input side; backward exits the range."""

    @staticmethod
    def forward(ctx, *args):
        *tensors, rf_holder = args
        ctx.rf_holder = rf_holder
        if len(tensors) == 1:
            return tensors[0].view_as(tensors[0])
        return tuple(tensor.view_as(tensor) for tensor in tensors)

    @staticmethod
    def backward(ctx, *grad_outputs):
        record = ctx.rf_holder[0]
        if record is not None:
            record.__exit__(None, None, None)
            ctx.rf_holder[0] = None
        return (*grad_outputs, None)


class BWDRecordEnd(torch.autograd.Function):
    """Plant at the forward output side; backward enters the range."""

    @staticmethod
    def forward(ctx, *args):
        *tensors, name, rf_holder = args
        ctx.name = name
        ctx.rf_holder = rf_holder
        if len(tensors) == 1:
            return tensors[0].view_as(tensors[0])
        return tuple(tensor.view_as(tensor) for tensor in tensors)

    @staticmethod
    def backward(ctx, *grad_outputs):
        record = record_function(ctx.name)
        record.__enter__()
        ctx.rf_holder[0] = record
        return (*grad_outputs, None, None)


def _identity_start(*tensors):
    return tensors[0] if len(tensors) == 1 else tensors


def _identity_end(*args):
    *tensors, _name = args
    return tensors[0] if len(tensors) == 1 else tuple(tensors)


def bwd_record_pair() -> Tuple[Callable, Callable]:
    """Return forward identities that delimit a backward trace range."""
    holder: List = [None]
    if not is_profile_enabled():
        return _identity_start, _identity_end

    def start(*tensors):
        return BWDRecordStart.apply(*tensors, holder)

    def end(*args):
        *tensors, name = args
        return BWDRecordEnd.apply(*tensors, name, holder)

    return start, end


class BWDRecordStartAnchored(torch.autograd.Function):
    """Backward range exit with a current-stream GPU anchor."""

    @staticmethod
    def forward(ctx, *args):
        *tensors, rf_holder = args
        ctx.rf_holder = rf_holder
        if len(tensors) == 1:
            return tensors[0].view_as(tensors[0])
        return tuple(tensor.view_as(tensor) for tensor in tensors)

    @staticmethod
    def backward(ctx, *grad_outputs):
        _anchor_kernel()
        record = ctx.rf_holder[0]
        if record is not None:
            record.__exit__(None, None, None)
            ctx.rf_holder[0] = None
        return (*grad_outputs, None)


class BWDRecordEndAnchored(torch.autograd.Function):
    """Backward range entry with a current-stream GPU anchor."""

    @staticmethod
    def forward(ctx, *args):
        *tensors, name, rf_holder = args
        ctx.name = name
        ctx.rf_holder = rf_holder
        if len(tensors) == 1:
            return tensors[0].view_as(tensors[0])
        return tuple(tensor.view_as(tensor) for tensor in tensors)

    @staticmethod
    def backward(ctx, *grad_outputs):
        record = record_function(ctx.name)
        record.__enter__()
        _anchor_kernel()
        ctx.rf_holder[0] = record
        return (*grad_outputs, None, None)


def bwd_record_pair_with_anchor() -> Tuple[Callable, Callable]:
    """Return a backward trace pair with GPU anchors at both boundaries."""
    holder: List = [None]
    if not is_profile_enabled():
        return _identity_start, _identity_end

    def start(*tensors):
        return BWDRecordStartAnchored.apply(*tensors, holder)

    def end(*args):
        *tensors, name = args
        return BWDRecordEndAnchored.apply(*tensors, name, holder)

    return start, end
