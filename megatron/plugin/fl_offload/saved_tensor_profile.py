"""Observe tensors saved for backward inside narrow FL semantic scopes."""

import contextlib
import contextvars
import json
from collections import defaultdict

import torch


SUPPORTED_SCOPES = (
    "qkv_linear",
    "core_attn",
    "attn_proj",
    "expert_fc1",
    "moe_act",
    "expert_fc2",
)

_CURRENT_COLLECTOR = contextvars.ContextVar("fl_saved_tensor_collector", default=None)
_INVOCATIONS = defaultdict(int)
_REPORT_COUNTS = defaultdict(int)
_STORAGE_SCOPES = defaultdict(set)
_REPORTS = []


def _args():
    from megatron.training import get_args

    return get_args()


def enabled():
    return bool(getattr(_args(), "fl_saved_tensor_profile", False))


def _selected_scopes():
    configured = getattr(_args(), "fl_saved_tensor_profile_scopes", None) or []
    scopes = tuple(configured) if configured else SUPPORTED_SCOPES
    unknown = sorted(set(scopes) - set(SUPPORTED_SCOPES))
    if unknown:
        raise ValueError(
            "unsupported --fl-saved-tensor-profile-scopes values: " + ", ".join(unknown)
        )
    return scopes


def _validate_runtime():
    if bool(getattr(_args(), "gradient_accumulation_fusion", False)):
        raise RuntimeError(
            "--fl-saved-tensor-profile currently requires "
            "--no-gradient-accumulation-fusion because PyTorch saved-tensor hooks "
            "do not preserve Parameter identity"
        )


def _rank():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _tensor_type(tensor):
    cls = type(tensor)
    return f"{cls.__module__}.{cls.__qualname__}"


def _storage_metadata(tensor):
    try:
        storage = tensor.untyped_storage()
        storage_ptr = storage.data_ptr()
        storage_nbytes = storage.nbytes()
        storage_key = (str(tensor.device), storage_ptr, storage_nbytes)
    except (AttributeError, NotImplementedError, RuntimeError, TypeError):
        storage_ptr = None
        storage_nbytes = None
        storage_key = None
    try:
        data_ptr = tensor.data_ptr()
    except (AttributeError, NotImplementedError, RuntimeError, TypeError):
        data_ptr = None
    return storage_key, storage_ptr, storage_nbytes, data_ptr


def _tensor_metadata(tensor, source):
    storage_key, storage_ptr, storage_nbytes, data_ptr = _storage_metadata(tensor)
    logical_bytes = tensor.numel() * tensor.element_size()
    is_parameter = isinstance(tensor, torch.nn.Parameter)
    metadata = {
        "source": source,
        "tensor_type": _tensor_type(tensor),
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "logical_bytes": logical_bytes,
        "data_ptr": data_ptr,
        "storage_ptr": storage_ptr,
        "storage_nbytes": storage_nbytes,
        "storage_offset": tensor.storage_offset(),
        "contiguous": tensor.is_contiguous(),
        "parameter": is_parameter,
        "requires_grad": tensor.requires_grad,
        "leaf": tensor.is_leaf,
        "_storage_key": storage_key,
    }
    return metadata


class _ScopeCollector:
    def __init__(self, scope, invocation):
        self.scope = scope
        self.invocation = invocation
        self.tensors = []

    def record(self, tensor, source):
        if isinstance(tensor, torch.Tensor):
            self.tensors.append(_tensor_metadata(tensor, source))

    def pack(self, tensor):
        self.record(tensor, "autograd")
        # Returning the original object avoids data copies. PyTorch still restores
        # Parameters as plain Tensors through this API, so fused wgrad is rejected
        # by _validate_runtime before the hook is installed.
        return tensor

    @staticmethod
    def unpack(packed):
        return packed

    def report(self):
        max_reports = int(getattr(_args(), "fl_saved_tensor_profile_max_reports", 1))
        if max_reports < 0:
            raise ValueError("--fl-saved-tensor-profile-max-reports must be non-negative")
        if max_reports and _REPORT_COUNTS[self.scope] >= max_reports:
            return

        logical_bytes = sum(item["logical_bytes"] for item in self.tensors)
        parameter_bytes = sum(
            item["logical_bytes"] for item in self.tensors if item["parameter"]
        )
        storage_sizes = {}
        activation_storage_sizes = {}
        cross_scope_storage_count = 0
        public_tensors = []
        for item in self.tensors:
            storage_key = item["_storage_key"]
            if storage_key is not None:
                storage_sizes[storage_key] = item["storage_nbytes"]
                if not item["parameter"]:
                    activation_storage_sizes[storage_key] = item["storage_nbytes"]
                shared_scopes = sorted(_STORAGE_SCOPES[storage_key] - {self.scope})
                if shared_scopes:
                    cross_scope_storage_count += 1
                _STORAGE_SCOPES[storage_key].add(self.scope)
            else:
                shared_scopes = []
            public_item = {key: value for key, value in item.items() if key != "_storage_key"}
            public_item["shared_with_scopes"] = shared_scopes
            public_tensors.append(public_item)

        report = {
            "rank": _rank(),
            "scope": self.scope,
            "invocation": self.invocation,
            "saved_tensors": len(self.tensors),
            "autograd_saved": sum(item["source"] == "autograd" for item in self.tensors),
            "explicit_saved": sum(item["source"] == "explicit" for item in self.tensors),
            "logical_bytes": logical_bytes,
            "activation_logical_bytes": logical_bytes - parameter_bytes,
            "parameter_logical_bytes": parameter_bytes,
            "unique_storage_bytes": sum(storage_sizes.values()),
            "unique_activation_storage_bytes": sum(activation_storage_sizes.values()),
            "cross_scope_storage_tensors": cross_scope_storage_count,
            "tensors": public_tensors,
        }
        _REPORTS.append(report)
        _REPORT_COUNTS[self.scope] += 1
        print("[FL saved-tensor-profile] " + json.dumps(report, sort_keys=True), flush=True)


@contextlib.contextmanager
def saved_tensor_scope(scope):
    """Observe saved tensors created while executing one semantic operator scope."""
    if not enabled() or not torch.is_grad_enabled() or scope not in _selected_scopes():
        yield
        return
    _validate_runtime()

    invocation = _INVOCATIONS[scope]
    _INVOCATIONS[scope] += 1
    collector = _ScopeCollector(scope, invocation)
    token = _CURRENT_COLLECTOR.set(collector)
    try:
        with torch.autograd.graph.saved_tensors_hooks(collector.pack, collector.unpack):
            yield
    finally:
        _CURRENT_COLLECTOR.reset(token)
        collector.report()


def record_explicit_tensor(tensor):
    """Include tensors handled by the existing explicit pack path in the active scope."""
    collector = _CURRENT_COLLECTOR.get()
    if collector is not None:
        collector.record(tensor, "explicit")


def reset_for_tests():
    _INVOCATIONS.clear()
    _REPORT_COUNTS.clear()
    _STORAGE_SCOPES.clear()
    _REPORTS.clear()
