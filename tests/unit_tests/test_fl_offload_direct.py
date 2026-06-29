"""Functional closure tests for the direct FL activation-offload port."""

import gc
from types import SimpleNamespace
import weakref

import pytest
import torch

from megatron.plugin.fl_offload import offload


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _runtime_args(enabled=True):
    return SimpleNamespace(
        fl_patch_te=enabled,
        fl_offload_modules=["swiglu"],
        fl_min_offloaded_tensor_size=0,
        fl_per_batch_offload_size=1,
        fl_activation_offload_stages=4,
        fl_activation_offload_stages_assignment=[0, 1, 2, 3],
    )


def test_one_mib_four_stage_round_trip(monkeypatch):
    args = _runtime_args()
    monkeypatch.setattr(offload, "_args", lambda: args)
    offload.reset_for_tests()

    source = torch.arange(1 << 19, dtype=torch.float16, device="cuda")
    expected = source.clone()
    key = (0, 0, 0)
    with offload.record(key, group_num=4):
        packed = offload.pack_hook(source, op_name="swiglu")

    with offload.OffloadAsync(key, group_num=4) as context:
        for stage in range(4):
            context.issue(stage)
    assert packed.get() is None

    with offload.OnloadAsync(key, group_num=4) as context:
        for stage in range(4):
            context.issue(stage)
    torch.testing.assert_close(packed.get(), expected, rtol=0, atol=0)


def test_partial_tensor_round_trip(monkeypatch):
    args = _runtime_args()
    monkeypatch.setattr(offload, "_args", lambda: args)
    offload.reset_for_tests()

    source = torch.arange(3 << 18, dtype=torch.float16, device="cuda")
    expected = source.clone()
    key = (0, 0, 1)
    with offload.record(key, group_num=4):
        packed = offload.pack_hook(source, op_name="swiglu")

    with offload.OffloadAsync(key, group_num=4) as context:
        context.issue(3)
    with offload.OnloadAsync(key, group_num=4) as context:
        context.issue(3)

    torch.testing.assert_close(packed.get(), expected, rtol=0, atol=0)


def test_offload_drops_copy_task_tensor_references(monkeypatch):
    args = _runtime_args()
    monkeypatch.setattr(offload, "_args", lambda: args)
    offload.reset_for_tests()

    source = torch.ones(1 << 19, dtype=torch.float16, device="cuda")
    torch.cuda.synchronize()
    allocated_with_source = torch.cuda.memory_allocated()
    source_ref = weakref.ref(source)
    key = (0, 0, 2)
    with offload.record(key, group_num=4):
        packed = offload.pack_hook(source, op_name="swiglu")
    del source

    with offload.OffloadAsync(key, group_num=4) as context:
        context.issue(3)
    torch.cuda.synchronize()
    gc.collect()
    assert source_ref() is None
    assert packed.get() is None
    assert torch.cuda.memory_allocated() <= allocated_with_source - (1 << 20)

    with offload.OnloadAsync(key, group_num=4) as context:
        context.issue(3)


def test_duplicate_contiguous_storage_is_copied_once(monkeypatch):
    args = _runtime_args()
    monkeypatch.setattr(offload, "_args", lambda: args)
    offload.reset_for_tests()

    source = torch.arange(1 << 19, dtype=torch.float16, device="cuda")
    expected = source.clone()
    source_ref = weakref.ref(source)
    key = (0, 0, 8)
    with offload.record(key, group_num=4):
        first = offload.pack_hook(source, op_name="swiglu")
        second = offload.pack_hook(source, op_name="swiglu")
    del source

    with offload.OffloadAsync(key, group_num=4) as context:
        context.issue(3)
    torch.cuda.synchronize()
    gc.collect()
    assert source_ref() is None

    with offload.OnloadAsync(key, group_num=4) as context:
        context.issue(3)
    torch.testing.assert_close(first.get(), expected, rtol=0, atol=0)
    torch.testing.assert_close(second.get(), expected, rtol=0, atol=0)
    assert first.get().untyped_storage().data_ptr() == second.get().untyped_storage().data_ptr()


def test_no_grad_record_does_not_create_activation_group(monkeypatch):
    args = _runtime_args()
    monkeypatch.setattr(offload, "_args", lambda: args)
    offload.reset_for_tests()

    source = torch.ones(1 << 19, dtype=torch.float16, device="cuda")
    key = (0, 0, 3)
    with torch.no_grad(), offload.record(key, group_num=4):
        packed = offload.pack_hook(source, op_name="swiglu")

    assert key not in offload._GROUPS
    assert packed.get() is source


def test_duplicate_live_group_key_is_rejected(monkeypatch):
    args = _runtime_args()
    monkeypatch.setattr(offload, "_args", lambda: args)
    offload.reset_for_tests()
    key = (0, 0, 5)

    with offload.record(key, group_num=4):
        pass
    with pytest.raises(RuntimeError, match="still active"):
        with offload.record(key, group_num=4):
            pass


def test_premature_unpack_and_reload_are_rejected(monkeypatch):
    args = _runtime_args()
    monkeypatch.setattr(offload, "_args", lambda: args)
    offload.reset_for_tests()
    source = torch.ones(1 << 19, dtype=torch.float16, device="cuda")
    key = (0, 0, 6)

    with offload.record(key, group_num=4):
        packed = offload.pack_hook(source, op_name="swiglu")
    with pytest.raises(RuntimeError, match="cannot enter reload"):
        offload.OnloadAsync(key, group_num=4).__enter__()

    with offload.OffloadAsync(key, group_num=4) as context:
        context.issue(3)
    with pytest.raises(RuntimeError, match="before its reload completed"):
        offload.unpack_hook(packed)
    with offload.OnloadAsync(key, group_num=4) as context:
        context.issue(3)


def test_stage_count_and_issue_range_are_validated(monkeypatch):
    args = _runtime_args()
    monkeypatch.setattr(offload, "_args", lambda: args)
    offload.reset_for_tests()
    key = (0, 0, 7)

    with offload.record(key, group_num=4):
        pass
    with pytest.raises(ValueError, match="captured with 4 stages"):
        offload.OffloadAsync(key, group_num=2)
    context = offload.OffloadAsync(key, group_num=4)
    with pytest.raises(ValueError, match="outside"):
        context.issue(4)
    with context:
        context.issue(3)
    with offload.OnloadAsync(key, group_num=4) as context:
        context.issue(3)


def test_issue_loads_reuses_four_stage_assignment_across_layers(monkeypatch):
    args = _runtime_args()
    monkeypatch.setattr(offload, "_args", lambda: args)
    offload.reset_for_tests()

    class IssueRecorder:
        def __init__(self):
            self.groups = []

        def issue(self, group_id):
            self.groups.append(group_id)

    offload.offload_ctx = IssueRecorder()
    offload.reload_ctx = IssueRecorder()
    offload.issue_loads(stage=5)

    assert offload.offload_ctx.groups == [1]
    assert offload.reload_ctx.groups == [1]


def test_weighted_swiglu_backward_matches_disabled_baseline(monkeypatch):
    from megatron.core.fusions.fused_bias_swiglu import WeightedSwiGLUFunction

    args = _runtime_args(enabled=False)
    monkeypatch.setattr(offload, "_args", lambda: args)
    x0 = torch.randn(1024, 512, dtype=torch.float16, device="cuda", requires_grad=True)
    w0 = torch.randn(1024, 1, dtype=torch.float16, device="cuda", requires_grad=True)
    WeightedSwiGLUFunction.apply(x0, w0, False).sum().backward()
    baseline_x_grad = x0.grad.clone()
    baseline_w_grad = w0.grad.clone()

    args.fl_patch_te = True
    x1 = x0.detach().clone().requires_grad_(True)
    w1 = w0.detach().clone().requires_grad_(True)
    key = (0, 0, 4)
    with offload.record(key, group_num=4):
        loss = WeightedSwiGLUFunction.apply(x1, w1, False).sum()
    with offload.OffloadAsync(key, group_num=4) as context:
        context.issue(3)
    with offload.OnloadAsync(key, group_num=4) as context:
        context.issue(3)
    loss.backward()

    torch.testing.assert_close(x1.grad, baseline_x_grad, rtol=0, atol=0)
    torch.testing.assert_close(w1.grad, baseline_w_grad, rtol=0, atol=0)
