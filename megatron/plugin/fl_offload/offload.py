"""FL activation offload runtime, ported with only platform API adaptation.

The data model and lifecycle intentionally follow
``fl_megatron/core/pipeline_parallel/offload.py``: explicit operator
pack/unpack, one activation group per microbatch, fixed byte budget, staged
D2H/H2D issue, and shared CPU/GPU byte buffers.
"""

import contextlib
import math
from collections import defaultdict
from contextvars import ContextVar

import torch

from megatron.plugin.profile import semantic_record


_GROUPS = {}
_OFFLOAD_TENSORS = None
_ATTENTION_OUTPUT_CANDIDATES = None
_PINNED_BUFFER_POOL = defaultdict(list)
_GPU_BUFFER_POOL = {}
_MEMCPY_STREAMS = {}
_PRINTED_CAPTURE_SUMMARIES = set()
_WARNED_INCOMPLETE_OFFLOAD = False
_WARNED_INCOMPLETE_ONLOAD = False
_WARNED_CONSERVATIVE_RELEASE = False
_NEXT_SEQUENCE_ID = 0
_TENSOR_SCOPE = ContextVar("fl_offload_tensor_scope", default=None)

_SCHEDULE_LOCATIONS = (
    "after_combine_bwd",
    "after_dispatch_fwd",
    "after_dispatch_bwd",
    "after_combine_fwd",
)

# These paths explicitly replace every custom-autograd saved reference to the
# copied tensor and have GPU round-trip coverage for active storage release.
# New scopes stay conservative until their external alias lifetimes have been
# verified independently on the target TE/backend version.
_ACTIVE_RELEASE_MODULES = {
    "LayerNormLinear",
    "GroupedLinear",
    "swiglu",
    "FlashAttention",
    "MTP",
}


def _supports_active_release(tensor):
    if tensor.op_name in _ACTIVE_RELEASE_MODULES:
        return True
    # Both probability-sized tensors are internal to TE's decomposed unfused
    # attention. Their softmax/context-BMM save sites are fully covered by the
    # narrow saved_tensors_hooks context, so no unpatched backward consumer can
    # retain their original storage.
    return (
        tensor.op_name == "UnfusedAttention"
        and tensor.tensor_name == "attention_probs"
    )


def _args():
    from megatron.training import get_args

    return get_args()


def enabled():
    return bool(getattr(_args(), "fl_patch_te", False))


def mtp_offload_enabled(config):
    """Whether the final MTP-containing model chunk must be recorded."""
    return (
        enabled()
        and bool(getattr(config, "mtp_num_layers", 0))
        and "MTP" in (getattr(_args(), "fl_offload_modules", []) or [])
    )


def group_available_for_reload(key):
    """Return whether a captured group has completed D2H and can start H2D."""
    group = _GROUPS.get(key)
    return group is not None and group.state == "offloaded"


@contextlib.contextmanager
def tensor_scope(op_name, tensor_name):
    """Identify a narrow operator region for the patched TE autograd function."""
    if op_name is None:
        yield
        return
    token = _TENSOR_SCOPE.set((op_name, tensor_name))
    try:
        yield
    finally:
        _TENSOR_SCOPE.reset(token)


def current_tensor_scope(default_op_name=None, default_tensor_name=None):
    scope = _TENSOR_SCOPE.get()
    if scope is not None:
        op_name, _tensor_name = scope
        if op_name in (getattr(_args(), "fl_offload_modules", []) or []):
            return scope
    return default_op_name, default_tensor_name


def maybe_pack_scoped_tensor(tensor):
    scope = _TENSOR_SCOPE.get()
    if scope is None or _OFFLOAD_TENSORS is None:
        return None
    op_name, tensor_name = scope
    if op_name not in (getattr(_args(), "fl_offload_modules", []) or []):
        return None
    return pack_hook(tensor, op_name=op_name, tensor_name=tensor_name)


def _same_storage(left, right):
    return (
        isinstance(left, torch.Tensor)
        and isinstance(right, torch.Tensor)
        and left.device == right.device
        and left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
    )


@contextlib.contextmanager
def unfused_attention_saved_tensors(query, key, value):
    """Capture tensors saved by TE's decomposed unfused attention operators.

    Unlike the TE and FlashAttention custom autograd functions, unfused
    attention is composed from PyTorch bmm, softmax, and dropout operators.
    A narrow saved-tensor context is therefore the only version-independent
    point at which their actual backward inputs can be replaced.
    """
    modules = getattr(_args(), "fl_offload_modules", []) or []
    if (
        not enabled()
        or _OFFLOAD_TENSORS is None
        or "UnfusedAttention" not in modules
    ):
        yield
        return

    references = (("q", query), ("k", key), ("v", value))

    def scoped_pack(tensor):
        tensor_name = None
        for name, reference in references:
            if _same_storage(tensor, reference):
                tensor_name = name
                break
        if tensor_name is None:
            if tensor.is_floating_point() and tensor.dim() >= 3:
                tensor_name = "attention_probs"
            else:
                return tensor
        return pack_hook(
            tensor,
            op_name="UnfusedAttention",
            tensor_name=tensor_name,
        )

    def scoped_unpack(tensor_pack):
        if isinstance(tensor_pack, TensorPack):
            return unpack_hook(tensor_pack)
        return tensor_pack

    with torch.autograd.graph.saved_tensors_hooks(scoped_pack, scoped_unpack):
        yield


def maybe_pack_unfused_attention_output(output):
    if not isinstance(output, torch.Tensor):
        return None
    modules = getattr(_args(), "fl_offload_modules", []) or []
    if _ATTENTION_OUTPUT_CANDIDATES is None or "UnfusedAttention" not in modules:
        return None
    # Unfused attention itself does not save O for backward.  Record only a
    # storage identity here; the following projection must confirm that its
    # actual saved input is this O tensor before it becomes eligible for active
    # release.  Packing O here would leave an unowned TensorPack and could
    # resize storage still referenced by an unpatched projection backward.
    _ATTENTION_OUTPUT_CANDIDATES.append(
        (
            output.device,
            output.data_ptr(),
            output.numel() * output.element_size(),
        )
    )
    return None


def get_memcpy_stream(key):
    if key not in ("offload", "onload"):
        raise ValueError(f"unsupported memcpy stream: {key}")
    if bool(getattr(_args(), "fl_use_comm_stream", False)):
        from megatron.core.pipeline_parallel.utils import get_comm_stream

        stream = get_comm_stream()
        if stream is None:
            raise RuntimeError(
                "--fl-use-comm-stream requires the combined schedule communication "
                "stream to be initialized"
            )
        return stream
    # Serialize D2H and H2D on one dedicated copy stream. This preserves their
    # issue order and avoids competing FL transfers unless the combined
    # communication stream is explicitly requested above.
    stream_key = "copy"
    if stream_key not in _MEMCPY_STREAMS:
        _MEMCPY_STREAMS[stream_key] = torch.cuda.Stream()
    return _MEMCPY_STREAMS[stream_key]


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

    def __init__(self, tensor, op_name=None, tensor_name=None):
        self.x = tensor
        self.shape = tensor.shape
        self.dtype = tensor.dtype
        self.device = tensor.device
        self.op_name = op_name
        self.tensor_name = tensor_name


class TensorPack:
    def __init__(self, tensor_wrap, op_name=None, tensor_name=None):
        self.tensor_wrap = tensor_wrap
        self.op_name = op_name
        self.tensor_name = tensor_name

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
        self.active_release_permissions = {}
        self.conservative_release_modules = set()
        self.state = "captured"

    def offload_prologue(self):
        if self.state != "captured":
            raise RuntimeError(
                f"FL offload group {self.key} cannot enter offload from state {self.state}"
            )
        global _PRINTED_CAPTURE_SUMMARIES
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
        # MTP is normally present only in the final PP/VPP chunk. Print each
        # distinct module composition once so an earlier decoder-only group
        # does not hide the final MTP capture summary.
        if top:
            captured_by_module = defaultdict(int)
            selected_by_module = defaultdict(int)
            for tensor, (begin, end, duplicate) in zip(self.tensors, self.mapping):
                if duplicate:
                    continue
                module = tensor.op_name or "unknown"
                if tensor.op_name in ("FlashAttention", "UnfusedAttention") and tensor.tensor_name:
                    module = f"{module}.{tensor.tensor_name}"
                captured_by_module[module] += end - begin
                selected_by_module[module] += max(0, min(end, offload_size) - begin)
            signature = tuple(sorted(captured_by_module))
            if signature not in _PRINTED_CAPTURE_SUMMARIES:
                module_summary = ", ".join(
                    f"{module}(captured={captured_by_module[module] / (1 << 20):.2f} MiB,"
                    f"selected={selected_by_module[module] / (1 << 20):.2f} MiB)"
                    for module in signature
                )
                print(
                    "[FL offload] "
                    f"captured={top / (1 << 20):.2f} MiB, "
                    f"budget={budget_mib} MiB, selected={offload_size / (1 << 20):.2f} MiB, "
                    f"modules=[{module_summary}]",
                    flush=True,
                )
                _PRINTED_CAPTURE_SUMMARIES.add(signature)
        if offload_size == 0:
            self.copy_groups = [[] for _ in range(self.group_num)]
            self.buffer_cpu = get_cpu_buffer(0)
            self.state = "offload_ready"
            return
        if offload_size % self.group_num:
            raise ValueError("offload bytes must be divisible by activation offload stages")

        logical_release_permissions = {}
        for tensor, (begin, end, _duplicate) in zip(self.tensors, self.mapping):
            if begin >= offload_size:
                continue
            logical_range = (begin, end)
            module_is_verified = _supports_active_release(tensor)
            logical_release_permissions[logical_range] = (
                logical_release_permissions.get(logical_range, True)
                and module_is_verified
            )
            if not module_is_verified:
                self.conservative_release_modules.add(tensor.op_name or "unknown")

        tasks = CopyTaskGroup(offload_size, self.group_num)
        for tensor, (begin, end, duplicate) in zip(self.tensors, self.mapping):
            if begin >= offload_size:
                continue
            if duplicate:
                continue
            logical_bytes = tensor.x.numel() * tensor.x.element_size()
            storage = tensor.x.untyped_storage()
            owns_exact_storage = (
                tensor.x.is_contiguous()
                and tensor.x.storage_offset() == 0
                and storage.nbytes() == logical_bytes
            )
            # Flash Q/K/V are commonly contiguous views into a packed QKV
            # allocation. Isolate such views before the active-release path
            # resizes copied source storages to zero.
            isolated_copy = not owns_exact_storage
            if isolated_copy:
                tensor.x = tensor.x.clone(memory_format=torch.contiguous_format)
            byte_tensor = tensor.x.view(torch.uint8).reshape(-1)
            storage = tensor.x.untyped_storage()
            storage_key = (tensor.x.device, storage.data_ptr())
            can_force_release = isolated_copy or logical_release_permissions[(begin, end)]
            self.active_release_permissions[storage_key] = (
                self.active_release_permissions.get(storage_key, True)
                and can_force_release
            )
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
        global _WARNED_CONSERVATIVE_RELEASE
        torch.cuda.current_stream().wait_stream(get_memcpy_stream("offload"))
        copied_storages = {}
        for copy_group in self.copy_groups:
            for _begin, _end, source in copy_group:
                storage = source.untyped_storage()
                copied_storages[(source.device, storage.data_ptr())] = storage
        for tensor, (begin, _end, _duplicate) in zip(self.tensors, self.mapping):
            if begin < getattr(self, "offload_size", 0):
                tensor.x = None
        self.copy_groups = [[] for _ in range(self.group_num)]
        # Match DCU's active-release behavior. The D2H stream is complete and
        # partial-tensor remainders were cloned in the prologue, so these source
        # storages are no longer needed by the backward graph.
        for storage_key, storage in copied_storages.items():
            if self.active_release_permissions.get(storage_key, False):
                storage.resize_(0)
        if self.conservative_release_modules and not _WARNED_CONSERVATIVE_RELEASE:
            print(
                "[FL offload] conservative source release for alias-sensitive modules: "
                + ", ".join(sorted(self.conservative_release_modules)),
                flush=True,
            )
            _WARNED_CONSERVATIVE_RELEASE = True
        self.state = "offloaded"

    def onload_prologue(self):
        if self.state != "offloaded":
            raise RuntimeError(
                f"FL offload group {self.key} cannot enter reload from state {self.state}"
            )
        self.onload_stream = get_memcpy_stream("onload")
        self.onload_buffer = get_persistent_gpu_buffer(
            "onload", self.buffer_cpu.numel()
        )
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
        self.buffer_cpu = None
        restored_ranges = {}
        for tensor, (begin, end, duplicate) in zip(self.tensors, self.mapping):
            if duplicate and begin < self.onload_buffer.numel():
                restored_bytes = restored_ranges[(begin, end)]
                tensor.x = restored_bytes.view(tensor.dtype).view(tensor.shape)
                continue
            if end <= self.onload_buffer.numel():
                tensor.x = (
                    self.onload_buffer[begin:end]
                    .view(tensor.dtype)
                    .view(tensor.shape)
                    .clone()
                )
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


def pack_hook(tensor, op_name=None, tensor_name=None):
    global _OFFLOAD_TENSORS
    if tensor is None:
        return None
    from .saved_tensor_profile import record_explicit_tensor

    record_explicit_tensor(tensor)
    wrapped = TensorWrap(tensor, op_name=op_name, tensor_name=tensor_name)
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
    return TensorPack(wrapped, op_name, tensor_name)


def maybe_pack_attention_projection(tensor):
    """Pack the projection input only when it is a captured attention output.

    Transformer Engine may materialize a distinct contiguous projection input
    after FlashAttention. The standalone memory model counts one O tensor, so
    this helper deliberately avoids capturing that additional allocation.
    """
    if (
        tensor is None
        or not isinstance(tensor, torch.Tensor)
        or _OFFLOAD_TENSORS is None
        or not tensor.is_contiguous()
    ):
        return None
    target_range = (
        tensor.device,
        tensor.data_ptr(),
        tensor.numel() * tensor.element_size(),
    )
    for wrapped in reversed(_OFFLOAD_TENSORS):
        candidate = wrapped.x
        if (
            wrapped.op_name == "FlashAttention"
            and wrapped.tensor_name == "output"
            and candidate is not None
            and candidate.is_contiguous()
            and (
                candidate.device,
                candidate.data_ptr(),
                candidate.numel() * candidate.element_size(),
            )
            == target_range
        ):
            return pack_hook(
                tensor,
                op_name=wrapped.op_name,
                tensor_name="attention_projection_input",
            )
    if _ATTENTION_OUTPUT_CANDIDATES is not None:
        for index in range(len(_ATTENTION_OUTPUT_CANDIDATES) - 1, -1, -1):
            if _ATTENTION_OUTPUT_CANDIDATES[index] == target_range:
                _ATTENTION_OUTPUT_CANDIDATES.pop(index)
                return pack_hook(
                    tensor,
                    op_name="UnfusedAttention",
                    tensor_name="output",
                )
    return None


def maybe_pack_linear_input(tensor):
    packed = maybe_pack_attention_projection(tensor)
    if packed is not None:
        return packed
    return maybe_pack_scoped_tensor(tensor)


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
    global _OFFLOAD_TENSORS, _ATTENTION_OUTPUT_CANDIDATES
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
    previous_attention_outputs = _ATTENTION_OUTPUT_CANDIDATES
    _OFFLOAD_TENSORS = []
    _ATTENTION_OUTPUT_CANDIDATES = []
    try:
        yield
        if key in _GROUPS:
            raise RuntimeError(f"FL activation group key {key} is still active")
        _GROUPS[key] = ActivationGroup(_OFFLOAD_TENSORS, key, group_num)
    finally:
        _OFFLOAD_TENSORS = previous
        _ATTENTION_OUTPUT_CANDIDATES = previous_attention_outputs


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
        if not -1 <= group_id < self.group_num:
            raise ValueError(
                f"FL offload stage {group_id} is outside [-1, {self.group_num})"
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
        if not -1 <= group_id < self.group_num:
            raise ValueError(
                f"FL reload stage {group_id} is outside [-1, {self.group_num})"
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
    if not -1 <= group_id < get_offload_nstages():
        raise ValueError(
            f"FL offload stage assignment {group_id} is outside "
            f"[-1, {get_offload_nstages()})"
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
    if _ATTENTION_OUTPUT_CANDIDATES is not None:
        active.append("attention_output_candidates")
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
    global _OFFLOAD_TENSORS, _ATTENTION_OUTPUT_CANDIDATES
    global _WARNED_INCOMPLETE_OFFLOAD, _WARNED_INCOMPLETE_ONLOAD
    global _WARNED_CONSERVATIVE_RELEASE
    global offload_ctx, reload_ctx, _NEXT_SEQUENCE_ID
    _GROUPS.clear()
    _OFFLOAD_TENSORS = None
    _ATTENTION_OUTPUT_CANDIDATES = None
    _PRINTED_CAPTURE_SUMMARIES.clear()
    _WARNED_INCOMPLETE_OFFLOAD = False
    _WARNED_INCOMPLETE_ONLOAD = False
    _WARNED_CONSERVATIVE_RELEASE = False
    offload_ctx = None
    reload_ctx = None
    _NEXT_SEQUENCE_ID = 0
