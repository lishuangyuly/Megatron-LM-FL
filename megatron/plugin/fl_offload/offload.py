"""FL activation offload runtime, ported with only platform API adaptation.

The data model and lifecycle intentionally follow
``fl_megatron/core/pipeline_parallel/offload.py``: explicit operator
pack/unpack, one activation group per microbatch, fixed byte budget, staged
D2H/H2D issue, and shared CPU/GPU byte buffers.
"""

import contextlib
import math
from collections import defaultdict

import torch

from megatron.plugin.profile import semantic_record


_GROUPS = {}
_OFFLOAD_TENSORS = None
_PINNED_BUFFER_POOL = defaultdict(list)
_GPU_BUFFER_POOL = {}
_MEMCPY_STREAMS = {}
_PRINTED_CAPTURE_SUMMARY = False
_WARNED_INCOMPLETE_OFFLOAD = False
_WARNED_INCOMPLETE_ONLOAD = False
_NEXT_SEQUENCE_ID = 0

_SCHEDULE_LOCATIONS = (
    "after_combine_bwd",
    "after_dispatch_fwd",
    "after_dispatch_bwd",
    "after_combine_fwd",
)


def _args():
    from megatron.training import get_args

    return get_args()


def enabled():
    return bool(getattr(_args(), "fl_patch_te", False))


def get_memcpy_stream(key):
    if key not in ("offload", "onload"):
        raise ValueError(f"unsupported memcpy stream: {key}")
    if key not in _MEMCPY_STREAMS:
        _MEMCPY_STREAMS[key] = torch.cuda.Stream()
    return _MEMCPY_STREAMS[key]


def get_cpu_buffer(num_bytes, dtype=torch.uint8):
    pool = _PINNED_BUFFER_POOL[(num_bytes, dtype)]
    for index in range(len(pool) - 1, -1, -1):
        buffer, ready_event = pool[index]
        if ready_event is None or ready_event.query():
            pool.pop(index)
            return buffer
    return torch.empty(num_bytes, dtype=dtype, pin_memory=True)


def recycle_cpu_buffer(buffer, stream=None):
    ready_event = None
    if stream is not None and buffer.numel():
        ready_event = torch.cuda.Event()
        ready_event.record(stream)
    _PINNED_BUFFER_POOL[(buffer.numel(), buffer.dtype)].append((buffer, ready_event))


def get_persistent_gpu_buffer(key, size):
    current = _GPU_BUFFER_POOL.get(key)
    if current is None or current.numel() < size:
        current = torch.empty(size, dtype=torch.uint8, device="cuda")
        _GPU_BUFFER_POOL[key] = current
    return current[:size]


def _copy_record(group, func, phase, stage_id=None):
    fields = {
        "func": func,
        "phase": phase,
        "sequence_id": group.sequence_id,
    }
    if isinstance(group.key, tuple) and len(group.key) == 3:
        fields.update(
            activation_group_id=group.key[0],
            model_chunk_id=group.key[1],
            microbatch_id=group.key[2],
        )
    else:
        fields["activation_key"] = str(group.key).replace("&", "%26").replace(" ", "")
    if stage_id is not None:
        fields["stage_id"] = stage_id
    return semantic_record(**fields)


class TensorWrap:
    """Mutable tensor slot retained by an operator autograd context."""

    def __init__(self, tensor):
        self.x = tensor
        self.shape = tensor.shape
        self.dtype = tensor.dtype
        self.device = tensor.device


class TensorPack:
    def __init__(self, tensor_wrap, op_name=None):
        self.tensor_wrap = tensor_wrap
        self.op_name = op_name

    def get(self):
        return self.tensor_wrap.x


class CopyTaskGroup:
    """Split a byte interval into the same fixed-size stages as FL."""

    def __init__(self, total_size, group_num):
        self.group_num = group_num
        self.group_size = total_size // group_num
        self.groups = [[] for _ in range(group_num)]
        self.group_id = 0
        self.group_fill = 0

    def add_tensor(self, begin, end, tensor):
        tensor = tensor.view(-1)
        while begin < end:
            room = self.group_size - self.group_fill
            amount = min(room, end - begin)
            if amount:
                self.groups[self.group_id].append((begin, begin + amount, tensor[:amount]))
                begin += amount
                tensor = tensor[amount:]
                self.group_fill += amount
            if self.group_fill == self.group_size and begin < end:
                self.group_id += 1
                self.group_fill = 0


class ActivationGroup:
    """All explicitly packed activations owned by one microbatch."""

    def __init__(self, tensors, key, group_num):
        global _NEXT_SEQUENCE_ID
        self.key = key
        self.sequence_id = _NEXT_SEQUENCE_ID
        _NEXT_SEQUENCE_ID += 1
        self.tensors = sorted(tensors, key=lambda t: (not t.x.is_contiguous(), -t.x.numel()))
        self.group_num = group_num
        self.copy_groups = None
        self.mapping = []
        self.partial_remainders = {}
        self.state = "captured"

    def offload_prologue(self):
        if self.state != "captured":
            raise RuntimeError(
                f"FL offload group {self.key} cannot enter offload from state {self.state}"
            )
        global _PRINTED_CAPTURE_SUMMARY
        top = 0
        contiguous_ranges = {}
        for tensor in self.tensors:
            duplicate = False
            if tensor.x.is_contiguous():
                range_key = (
                    tensor.x.device,
                    tensor.x.data_ptr(),
                    tensor.x.numel() * tensor.x.element_size(),
                )
                previous = contiguous_ranges.get(range_key)
                if previous is not None:
                    begin, end, _duplicate = self.mapping[previous]
                    self.mapping.append((begin, end, True))
                    duplicate = True
                else:
                    contiguous_ranges[range_key] = len(self.mapping)
            if duplicate:
                continue
            size = tensor.x.numel() * tensor.x.element_size()
            self.mapping.append((top, top + size, False))
            top += size

        budget_mib = int(getattr(_args(), "fl_per_batch_offload_size", 0))
        if budget_mib < 0:
            raise ValueError("--fl-per-batch-offload-size must be non-negative")
        budget = budget_mib * (1 << 20)
        offload_size = budget if top >= budget else 0
        self.offload_size = offload_size
        if not _PRINTED_CAPTURE_SUMMARY:
            print(
                "[FL offload] "
                f"captured={top / (1 << 20):.2f} MiB, "
                f"budget={budget_mib} MiB, selected={offload_size / (1 << 20):.2f} MiB",
                flush=True,
            )
            _PRINTED_CAPTURE_SUMMARY = True
        if offload_size == 0:
            self.copy_groups = [[] for _ in range(self.group_num)]
            self.buffer_cpu = get_cpu_buffer(0)
            self.state = "offload_ready"
            return
        if offload_size % self.group_num:
            raise ValueError("offload bytes must be divisible by activation offload stages")

        tasks = CopyTaskGroup(offload_size, self.group_num)
        for tensor, (begin, end, duplicate) in zip(self.tensors, self.mapping):
            if begin >= offload_size:
                continue
            if duplicate:
                continue
            if not tensor.x.is_contiguous():
                tensor.x = tensor.x.contiguous()
            byte_tensor = tensor.x.view(torch.uint8).reshape(-1)
            selected = min(end, offload_size) - begin
            tasks.add_tensor(begin, begin + selected, byte_tensor[:selected])
            if selected < byte_tensor.numel():
                self.partial_remainders[tensor] = byte_tensor[selected:].clone()

        self.offload_size = offload_size
        self.copy_groups = tasks.groups
        self.buffer_cpu = get_cpu_buffer(offload_size)
        self.state = "offload_ready"

    def offload_issue(self, group_id):
        if self.state not in {"offload_ready", "offloading"}:
            raise RuntimeError(
                f"FL offload group {self.key} cannot issue D2H from state {self.state}"
            )
        self.state = "offloading"
        stream = get_memcpy_stream("offload")
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for begin, end, source in self.copy_groups[group_id]:
                self.buffer_cpu[begin:end].copy_(source, non_blocking=True)
                source.record_stream(stream)

    def offload_epilogue(self):
        if self.state not in {"offload_ready", "offloading"}:
            raise RuntimeError(
                f"FL offload group {self.key} cannot finish D2H from state {self.state}"
            )
        torch.cuda.current_stream().wait_stream(get_memcpy_stream("offload"))
        for tensor, (begin, _end, _duplicate) in zip(self.tensors, self.mapping):
            if begin < getattr(self, "offload_size", 0):
                tensor.x = None
        self.copy_groups = [[] for _ in range(self.group_num)]
        self.state = "offloaded"

    def onload_prologue(self):
        if self.state != "offloaded":
            raise RuntimeError(
                f"FL offload group {self.key} cannot enter reload from state {self.state}"
            )
        self.onload_stream = get_memcpy_stream("onload")
        self.onload_buffer = get_persistent_gpu_buffer("onload", self.buffer_cpu.numel())
        self.onload_stream.wait_stream(torch.cuda.current_stream())
        self.onload_stage_size = self.buffer_cpu.numel() // self.group_num
        self.state = "reload_ready"

    def onload_issue(self, group_id):
        if self.state not in {"reload_ready", "reloading"}:
            raise RuntimeError(
                f"FL offload group {self.key} cannot issue H2D from state {self.state}"
            )
        self.state = "reloading"
        begin = group_id * self.onload_stage_size
        end = begin + self.onload_stage_size
        with torch.cuda.stream(self.onload_stream):
            self.onload_buffer[begin:end].copy_(self.buffer_cpu[begin:end], non_blocking=True)

    def onload_epilogue(self):
        if self.state not in {"reload_ready", "reloading"}:
            raise RuntimeError(
                f"FL offload group {self.key} cannot finish H2D from state {self.state}"
            )
        torch.cuda.current_stream().wait_stream(self.onload_stream)
        recycle_cpu_buffer(self.buffer_cpu, self.onload_stream)
        restored_ranges = {}
        for tensor, (begin, end, duplicate) in zip(self.tensors, self.mapping):
            if duplicate and begin < self.onload_buffer.numel():
                restored_bytes = restored_ranges[(begin, end)]
                tensor.x = restored_bytes.view(tensor.dtype).view(tensor.shape)
                continue
            if end <= self.onload_buffer.numel():
                tensor.x = self.onload_buffer[begin:end].view(tensor.dtype).view(tensor.shape).clone()
            elif begin < self.onload_buffer.numel():
                restored = torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)
                restored_bytes = restored.view(torch.uint8).reshape(-1)
                prefix_size = self.onload_buffer.numel() - begin
                restored_bytes[:prefix_size].copy_(self.onload_buffer[begin:])
                restored_bytes[prefix_size:].copy_(self.partial_remainders.pop(tensor))
                tensor.x = restored
            if begin < self.onload_buffer.numel():
                restored_ranges[(begin, end)] = tensor.x.view(torch.uint8).reshape(-1)
        self.partial_remainders.clear()
        self.state = "reloaded"


def pack_hook(tensor, op_name=None):
    global _OFFLOAD_TENSORS
    if tensor is None:
        return None
    wrapped = TensorWrap(tensor)
    modules = getattr(_args(), "fl_offload_modules", []) or []
    min_bytes = int(getattr(_args(), "fl_min_offloaded_tensor_size", 1 << 20))
    is_rope_frequency_buffer = (
        tensor.dim() == 4 and tensor.shape[1] == 1 and tensor.shape[2] == 1
    )
    eligible = (
        enabled()
        and _OFFLOAD_TENSORS is not None
        and op_name in modules
        and not isinstance(tensor, torch.nn.Parameter)
        and not is_rope_frequency_buffer
        and tensor.numel() * tensor.element_size() >= min_bytes
    )
    if eligible:
        _OFFLOAD_TENSORS.append(wrapped)
    return TensorPack(wrapped, op_name)


def unpack_hook(tensor_pack):
    if tensor_pack is None:
        return None
    tensor = tensor_pack.get()
    if tensor is None:
        raise RuntimeError("FL activation was requested before its reload completed")
    return tensor


def get_offload_nstages():
    stages = int(getattr(_args(), "fl_activation_offload_stages", 1))
    if stages <= 0:
        raise ValueError("--fl-activation-offload-stages must be positive")
    return stages


@contextlib.contextmanager
def record(key, group_num=None):
    global _OFFLOAD_TENSORS
    if not enabled() or not torch.is_grad_enabled():
        yield
        return
    if group_num is None:
        group_num = get_offload_nstages()
    if group_num <= 0:
        raise ValueError("FL activation offload group count must be positive")
    if key in _GROUPS:
        raise RuntimeError(f"FL activation group key {key} is still active")
    previous = _OFFLOAD_TENSORS
    _OFFLOAD_TENSORS = []
    try:
        yield
        if key in _GROUPS:
            raise RuntimeError(f"FL activation group key {key} is still active")
        _GROUPS[key] = ActivationGroup(_OFFLOAD_TENSORS, key, group_num)
    finally:
        _OFFLOAD_TENSORS = previous


class OffloadAsync:
    def __init__(self, key, group_num=None):
        self.group_num = get_offload_nstages() if group_num is None else group_num
        if self.group_num <= 0:
            raise ValueError("FL activation offload group count must be positive")
        if key not in _GROUPS:
            raise RuntimeError(f"FL activation group {key} was not captured")
        self.group = _GROUPS[key]
        if self.group_num != self.group.group_num:
            raise ValueError(
                f"FL activation group {key} was captured with {self.group.group_num} stages, "
                f"but offload requested {self.group_num}"
            )
        self.issued_group = 0

    def __enter__(self):
        with _copy_record(self.group, "fl_offload", "prologue"):
            self.group.offload_prologue()
        return self

    def issue(self, group_id):
        if not 0 <= group_id < self.group_num:
            raise ValueError(
                f"FL offload stage {group_id} is outside [0, {self.group_num})"
            )
        while self.issued_group <= group_id and self.issued_group < self.group_num:
            with _copy_record(
                self.group, "fl_offload", "issue", stage_id=self.issued_group
            ):
                self.group.offload_issue(self.issued_group)
            self.issued_group += 1

    def __exit__(self, *_exc):
        global _WARNED_INCOMPLETE_OFFLOAD
        if self.issued_group < self.group_num and not _WARNED_INCOMPLETE_OFFLOAD:
            print(
                "[FL offload] warning: staged D2H was incomplete; issuing remaining groups at exit",
                flush=True,
            )
            _WARNED_INCOMPLETE_OFFLOAD = True
        self.issue(self.group_num - 1)
        with _copy_record(self.group, "fl_offload", "epilogue"):
            self.group.offload_epilogue()


class OnloadAsync:
    def __init__(self, key, group_num=None):
        self.key = key
        self.group_num = get_offload_nstages() if group_num is None else group_num
        if self.group_num <= 0:
            raise ValueError("FL activation reload group count must be positive")
        if key not in _GROUPS:
            raise RuntimeError(f"FL activation group {key} is unavailable for reload")
        self.group = _GROUPS[key]
        if self.group_num != self.group.group_num:
            raise ValueError(
                f"FL activation group {key} was captured with {self.group.group_num} stages, "
                f"but reload requested {self.group_num}"
            )
        self.issued_group = 0

    def __enter__(self):
        with _copy_record(self.group, "fl_reload", "prologue"):
            self.group.onload_prologue()
        return self

    def issue(self, group_id):
        if not 0 <= group_id < self.group_num:
            raise ValueError(
                f"FL reload stage {group_id} is outside [0, {self.group_num})"
            )
        while self.issued_group <= group_id and self.issued_group < self.group_num:
            with _copy_record(
                self.group, "fl_reload", "issue", stage_id=self.issued_group
            ):
                self.group.onload_issue(self.issued_group)
            self.issued_group += 1

    def __exit__(self, *_exc):
        global _WARNED_INCOMPLETE_ONLOAD
        if self.issued_group < self.group_num and not _WARNED_INCOMPLETE_ONLOAD:
            print(
                "[FL offload] warning: staged H2D was incomplete; issuing remaining groups at exit",
                flush=True,
            )
            _WARNED_INCOMPLETE_ONLOAD = True
        self.issue(self.group_num - 1)
        with _copy_record(self.group, "fl_reload", "epilogue"):
            self.group.onload_epilogue()
        del _GROUPS[self.key]


offload_ctx = None
reload_ctx = None


def issue_loads(stage):
    if not enabled():
        return
    assignment = getattr(_args(), "fl_activation_offload_stages_assignment", None)
    if not assignment:
        assignment = [get_offload_nstages() - 1]
    group_id = assignment[stage % len(assignment)]
    if not 0 <= group_id < get_offload_nstages():
        raise ValueError(
            f"FL offload stage assignment {group_id} is outside "
            f"[0, {get_offload_nstages()})"
        )
    with semantic_record(
        func="fl_issue_loads",
        schedule_stage=stage,
        stage_id=group_id,
        location=_SCHEDULE_LOCATIONS[stage % len(_SCHEDULE_LOCATIONS)],
    ):
        if reload_ctx is not None:
            reload_ctx.issue(group_id)
        if offload_ctx is not None:
            offload_ctx.issue(group_id)


def assert_runtime_idle():
    active = []
    if _OFFLOAD_TENSORS is not None:
        active.append("capture")
    if offload_ctx is not None:
        active.append("offload_ctx")
    if reload_ctx is not None:
        active.append("reload_ctx")
    if _GROUPS:
        active.append(f"groups={list(_GROUPS)}")
    if active:
        raise RuntimeError(
            "FL activation offload state crossed the training-step boundary: "
            + ", ".join(active)
        )


def reset_for_tests():
    global _OFFLOAD_TENSORS, _PRINTED_CAPTURE_SUMMARY
    global _WARNED_INCOMPLETE_OFFLOAD, _WARNED_INCOMPLETE_ONLOAD
    global offload_ctx, reload_ctx, _NEXT_SEQUENCE_ID
    _GROUPS.clear()
    _OFFLOAD_TENSORS = None
    _PRINTED_CAPTURE_SUMMARY = False
    _WARNED_INCOMPLETE_OFFLOAD = False
    _WARNED_INCOMPLETE_ONLOAD = False
    offload_ctx = None
    reload_ctx = None
    _NEXT_SEQUENCE_ID = 0
