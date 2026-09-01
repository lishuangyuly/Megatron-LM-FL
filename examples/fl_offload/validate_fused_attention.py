"""Validate FL FusedAttention offload against a native TE baseline."""

import os
from pathlib import Path
import sys
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from megatron.plugin.fl_offload import offload
from megatron.plugin.fl_offload.te_patch import apply_te_patches, restore_te_patches


def _runtime_args():
    return SimpleNamespace(
        fl_patch_te=True,
        fl_offload_modules=["Attention"],
        attention_backend="fused",
        fl_min_offloaded_tensor_size=0,
        fl_per_batch_offload_size=33,
        fl_activation_offload_stages=4,
        fl_activation_offload_stages_assignment=[0, 1, 2, 3],
        fl_use_comm_stream=False,
    )


def _make_inputs():
    shape = (4096, 1, 64, 16)
    return [
        torch.randn(shape, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        for _ in range(3)
    ]


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("FL FusedAttention validation requires a CUDA GPU")

    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "1"
    os.environ["NVTE_UNFUSED_ATTN"] = "0"

    import transformer_engine.pytorch as te

    original_args = offload._args
    offload._args = _runtime_args
    offload.reset_for_tests()
    attention = te.DotProductAttention(
        num_attention_heads=64,
        kv_channels=16,
        num_gqa_groups=64,
        attention_dropout=0.0,
        qkv_format="sbhd",
        attn_mask_type="causal",
    ).cuda().train()
    projection = te.Linear(
        1024,
        1024,
        bias=False,
        params_dtype=torch.bfloat16,
    ).cuda().train()

    inputs = _make_inputs()
    baseline_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in inputs]
    baseline_output = attention(*baseline_inputs, None, attn_mask_type="causal")
    baseline_projection = projection(baseline_output)
    baseline_loss = baseline_projection.float().square().mean()
    baseline_loss.backward()
    expected_grads = [tensor.grad.clone() for tensor in baseline_inputs]
    expected_weight_grad = projection.weight.grad.clone()
    projection.zero_grad(set_to_none=True)

    apply_te_patches()
    key = (0, 0, 0)
    try:
        with offload.record(key, group_num=4):
            output = attention(*inputs, None, attn_mask_type="causal")
            projected = projection(output)
            loss = projected.float().square().mean()
        del output, projected

        group = offload._GROUPS[key]
        captured_by_name = {}
        for tensor in group.tensors:
            captured_by_name[tensor.tensor_name] = (
                captured_by_name.get(tensor.tensor_name, 0)
                + tensor.x.numel() * tensor.x.element_size()
            )
        expected = {
            "q": 8 << 20,
            "k": 8 << 20,
            "v": 8 << 20,
            "output": 8 << 20,
            "softmax_stats": 1 << 20,
            "attention_projection_input": 8 << 20,
        }
        if captured_by_name != expected:
            readable = {name: size / (1 << 20) for name, size in captured_by_name.items()}
            raise AssertionError(f"unexpected FusedAttention capture: {readable}")

        output_wrap = next(
            tensor for tensor in group.tensors if tensor.tensor_name == "output"
        )
        projection_wrap = next(
            tensor
            for tensor in group.tensors
            if tensor.tensor_name == "attention_projection_input"
        )
        if output_wrap.x.data_ptr() != projection_wrap.x.data_ptr():
            raise AssertionError("attention projection did not reuse FusedAttention output")

        with offload.OffloadAsync(key, group_num=4) as context:
            context.issue(3)
        with offload.OnloadAsync(key, group_num=4) as context:
            context.issue(3)
        loss.backward()
        torch.cuda.synchronize()

        for actual, expected_grad in zip(inputs, expected_grads):
            torch.testing.assert_close(actual.grad, expected_grad, rtol=2e-3, atol=2e-3)
        torch.testing.assert_close(
            projection.weight.grad, expected_weight_grad, rtol=2e-3, atol=2e-3
        )
        print(
            "[FL FusedAttention check] PASSED: captured=33.00 MiB, "
            f"loss={loss.item():.8f}, Q/K/V/projection gradients match",
            flush=True,
        )
    finally:
        restore_te_patches()
        offload.reset_for_tests()
        offload._args = original_args


if __name__ == "__main__":
    main()
