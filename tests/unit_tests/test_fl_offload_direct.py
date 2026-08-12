"""Functional closure tests for the direct FL activation-offload port."""

import gc
from types import SimpleNamespace
import weakref

import pytest
import torch

from megatron.plugin.fl_offload import offload


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _runtime_args(enabled=True, modules=None, budget_mib=1, min_tensor_bytes=0):
    return SimpleNamespace(
        fl_patch_te=enabled,
        fl_offload_modules=modules or ["swiglu"],
        fl_min_offloaded_tensor_size=min_tensor_bytes,
        fl_per_batch_offload_size=budget_mib,
        fl_activation_offload_stages=4,
        fl_activation_offload_stages_assignment=[0, 1, 2, 3],
        fl_use_comm_stream=False,
    )


def test_memcpy_stream_can_use_combined_communication_stream(monkeypatch):
    args = _runtime_args()
    args.fl_use_comm_stream = True
    monkeypatch.setattr(offload, "_args", lambda: args)
    from megatron.core.pipeline_parallel import utils

    communication_stream = object()
    monkeypatch.setattr(utils, "get_comm_stream", lambda: communication_stream)

    assert offload.get_memcpy_stream("offload") is communication_stream
    assert offload.get_memcpy_stream("onload") is communication_stream


def test_offload_and_reload_share_dedicated_copy_stream(monkeypatch):
    args = _runtime_args()
    monkeypatch.setattr(offload, "_args", lambda: args)
    copy_stream = object()
    monkeypatch.setattr(torch.cuda, "Stream", lambda: copy_stream)
    streams = {}
    monkeypatch.setattr(offload, "_MEMCPY_STREAMS", streams)

    assert offload.get_memcpy_stream("offload") is copy_stream
    assert offload.get_memcpy_stream("onload") is copy_stream
    assert streams == {"copy": copy_stream}


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


def test_reload_clones_from_persistent_landing_buffer(monkeypatch):
    args = _runtime_args()
    monkeypatch.setattr(offload, "_args", lambda: args)
    offload.reset_for_tests()

    source = torch.arange(1 << 19, dtype=torch.float16, device="cuda")
    key = (0, 0, 9)
    with offload.record(key, group_num=4):
        packed = offload.pack_hook(source, op_name="swiglu")

    with offload.OffloadAsync(key, group_num=4) as context:
        context.issue(3)

    context = offload.OnloadAsync(key, group_num=4)
    context.__enter__()
    landing_data_ptr = context.group.onload_buffer.data_ptr()
    context.issue(3)
    context.__exit__(None, None, None)

    assert packed.get().data_ptr() != landing_data_ptr
    assert offload._GPU_BUFFER_POOL["onload"].data_ptr() == landing_data_ptr


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


def test_shared_storage_view_is_isolated_before_active_release(monkeypatch):
    args = _runtime_args()
    monkeypatch.setattr(offload, "_args", lambda: args)
    offload.reset_for_tests()

    base = torch.arange(1 << 20, dtype=torch.float16, device="cuda")
    source = base[: 1 << 19]
    expected = source.clone()
    base_storage = base.untyped_storage()
    key = (0, 0, 10)
    with offload.record(key, group_num=4):
        packed = offload.pack_hook(source, op_name="swiglu")

    with offload.OffloadAsync(key, group_num=4) as context:
        context.issue(3)
    assert base_storage.nbytes() == 2 << 20

    with offload.OnloadAsync(key, group_num=4) as context:
        context.issue(3)
    torch.testing.assert_close(packed.get(), expected, rtol=0, atol=0)


def test_alias_sensitive_q_does_not_force_release_source_storage(monkeypatch):
    args = _runtime_args(modules=["UnfusedAttention"])
    monkeypatch.setattr(offload, "_args", lambda: args)
    offload.reset_for_tests()

    source = torch.arange(1 << 19, dtype=torch.float16, device="cuda")
    expected = source.clone()
    source_storage = source.untyped_storage()
    key = (0, 0, 15)
    with offload.record(key, group_num=4):
        packed = offload.pack_hook(
            source,
            op_name="UnfusedAttention",
            tensor_name="q",
        )

    with offload.OffloadAsync(key, group_num=4) as context:
        context.issue(3)
    assert packed.get() is None
    assert source_storage.nbytes() == 1 << 20

    with offload.OnloadAsync(key, group_num=4) as context:
        context.issue(3)
    torch.testing.assert_close(packed.get(), expected, rtol=0, atol=0)


def test_unfused_attention_probability_storage_is_actively_released(monkeypatch):
    args = _runtime_args(modules=["UnfusedAttention"])
    monkeypatch.setattr(offload, "_args", lambda: args)
    offload.reset_for_tests()

    source = torch.arange(1 << 19, dtype=torch.float16, device="cuda")
    expected = source.clone()
    source_storage = source.untyped_storage()
    key = (0, 0, 16)
    with offload.record(key, group_num=4):
        packed = offload.pack_hook(
            source,
            op_name="UnfusedAttention",
            tensor_name="attention_probs",
        )

    with offload.OffloadAsync(key, group_num=4) as context:
        context.issue(3)
    assert packed.get() is None
    assert source_storage.nbytes() == 0

    with offload.OnloadAsync(key, group_num=4) as context:
        context.issue(3)
    torch.testing.assert_close(packed.get(), expected, rtol=0, atol=0)


def test_mtp_projection_input_storage_is_actively_released(monkeypatch):
    args = _runtime_args(modules=["MTP"])
    monkeypatch.setattr(offload, "_args", lambda: args)
    offload.reset_for_tests()

    source = torch.arange(1 << 19, dtype=torch.float16, device="cuda")
    expected = source.clone()
    source_storage = source.untyped_storage()
    key = (0, 0, 17)
    with offload.record(key, group_num=4):
        with offload.tensor_scope("MTP", "eh_proj_input"):
            packed = offload.maybe_pack_linear_input(source)

    with offload.OffloadAsync(key, group_num=4) as context:
        context.issue(3)
    assert packed.get() is None
    assert source_storage.nbytes() == 0

    with offload.OnloadAsync(key, group_num=4) as context:
        context.issue(3)
    torch.testing.assert_close(packed.get(), expected, rtol=0, atol=0)


def test_flash_attention_qkvo_lse_round_trip_matches_baseline(monkeypatch):
    flash_attn = pytest.importorskip("flash_attn")
    args = _runtime_args(
        modules=["FlashAttention"], budget_mib=33, min_tensor_bytes=0
    )
    monkeypatch.setattr(offload, "_args", lambda: args)
    offload.reset_for_tests()

    shape = (1, 4096, 64, 16)
    inputs = [
        torch.randn(shape, dtype=torch.float16, device="cuda", requires_grad=True)
        for _ in range(3)
    ]
    baseline_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in inputs]
    baseline_output = flash_attn.flash_attn_func(
        *baseline_inputs, dropout_p=0.0, causal=True
    )
    baseline_loss = baseline_output.float().square().mean()
    baseline_loss.backward()
    expected_grads = [tensor.grad.clone() for tensor in baseline_inputs]

    from megatron.plugin.fl_offload.te_patch import apply_te_patches, restore_te_patches

    apply_te_patches()
    key = (0, 0, 11)
    try:
        with offload.record(key, group_num=4):
            output = flash_attn.flash_attn_func(
                *inputs, dropout_p=0.0, causal=True
            )
            loss = output.clone().float().square().mean()
        del output

        group = offload._GROUPS[key]
        assert sum(
            tensor.x.numel() * tensor.x.element_size() for tensor in group.tensors
        ) == 33 << 20
        assert {tensor.tensor_name for tensor in group.tensors} == {
            "q",
            "k",
            "v",
            "output",
            "softmax_lse",
        }

        with offload.OffloadAsync(key, group_num=4) as context:
            context.issue(3)
        with offload.OnloadAsync(key, group_num=4) as context:
            context.issue(3)
        loss.backward()
    finally:
        restore_te_patches()

    for actual, expected in zip(inputs, expected_grads):
        torch.testing.assert_close(actual.grad, expected, rtol=2e-3, atol=2e-3)


def test_offload_drops_copy_task_tensor_references(monkeypatch):
    args = _runtime_args()
    monkeypatch.setattr(offload, "_args", lambda: args)
    offload.reset_for_tests()

    source = torch.ones(1 << 19, dtype=torch.float16, device="cuda")
    source_storage = source.untyped_storage()
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
    assert source_storage.nbytes() == 0
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


def test_issue_loads_supports_dcu_skip_and_six_stage_assignment(monkeypatch):
    args = _runtime_args()
    args.fl_activation_offload_stages = 6
    args.fl_activation_offload_stages_assignment = [-1, 0, 1, 2, 2, 3, 4, 5]
    monkeypatch.setattr(offload, "_args", lambda: args)
    offload.reset_for_tests()

    class IssueGroup:
        def __init__(self):
            self.sequence_id = 0
            self.key = (0, 0, 0)
            self.offloaded = []
            self.reloaded = []

        def offload_issue(self, group_id):
            self.offloaded.append(group_id)

        def onload_issue(self, group_id):
            self.reloaded.append(group_id)

    group = IssueGroup()
    offload_ctx = object.__new__(offload.OffloadAsync)
    offload_ctx.group_num = 6
    offload_ctx.issued_group = 0
    offload_ctx.group = group
    reload_ctx = object.__new__(offload.OnloadAsync)
    reload_ctx.group_num = 6
    reload_ctx.issued_group = 0
    reload_ctx.group = group
    offload.offload_ctx = offload_ctx
    offload.reload_ctx = reload_ctx

    for schedule_stage in range(8):
        offload.issue_loads(schedule_stage)

    assert group.offloaded == [0, 1, 2, 3, 4, 5]
    assert group.reloaded == [0, 1, 2, 3, 4, 5]


def test_issue_loads_rejects_stage_below_dcu_skip_sentinel(monkeypatch):
    args = _runtime_args()
    args.fl_activation_offload_stages_assignment = [-2]
    monkeypatch.setattr(offload, "_args", lambda: args)

    with pytest.raises(ValueError, match=r"outside \[-1, 4\)"):
        offload.issue_loads(0)


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


def test_shared_expert_swiglu_backward_matches_disabled_baseline(monkeypatch):
    from megatron.core.fusions.fused_bias_swiglu import SwiGLUFunction

    args = _runtime_args(enabled=False, modules=["SharedExpert"])
    monkeypatch.setattr(offload, "_args", lambda: args)
    x0 = torch.randn(1024, 512, dtype=torch.float16, device="cuda", requires_grad=True)
    SwiGLUFunction.apply(x0, False, False).sum().backward()
    baseline_grad = x0.grad.clone()

    args.fl_patch_te = True
    x1 = x0.detach().clone().requires_grad_(True)
    key = (0, 0, 12)
    with offload.record(key, group_num=4):
        with offload.tensor_scope("SharedExpert", "swiglu_input"):
            loss = SwiGLUFunction.apply(x1, False, False).sum()
    group = offload._GROUPS[key]
    assert len(group.tensors) == 1
    assert group.tensors[0].tensor_name == "swiglu_input"

    with offload.OffloadAsync(key, group_num=4) as context:
        context.issue(3)
    with offload.OnloadAsync(key, group_num=4) as context:
        context.issue(3)
    loss.backward()

    torch.testing.assert_close(x1.grad, baseline_grad, rtol=0, atol=0)


def test_unfused_attention_probability_round_trip_matches_baseline(monkeypatch):
    args = _runtime_args(
        modules=["UnfusedAttention"], budget_mib=1, min_tensor_bytes=0
    )
    monkeypatch.setattr(offload, "_args", lambda: args)
    offload.reset_for_tests()

    inputs = [
        torch.randn(8, 256, 64, dtype=torch.float16, device="cuda", requires_grad=True)
        for _ in range(3)
    ]
    baseline_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in inputs]

    def attention(q, k, v):
        probabilities = torch.softmax(torch.bmm(q, k.transpose(1, 2)), dim=-1)
        return torch.bmm(probabilities, v)

    attention(*baseline_inputs).float().sum().backward()
    expected_grads = [tensor.grad.clone() for tensor in baseline_inputs]

    key = (0, 0, 14)
    with offload.record(key, group_num=4):
        with offload.unfused_attention_saved_tensors(*inputs):
            loss = attention(*inputs).float().sum()

    group = offload._GROUPS[key]
    assert "attention_probs" in {tensor.tensor_name for tensor in group.tensors}
    with offload.OffloadAsync(key, group_num=4) as context:
        for stage in range(4):
            context.issue(stage)
    with offload.OnloadAsync(key, group_num=4) as context:
        for stage in range(4):
            context.issue(stage)
    loss.backward()

    for actual, expected in zip(inputs, expected_grads):
        torch.testing.assert_close(actual.grad, expected, rtol=2e-3, atol=2e-3)
