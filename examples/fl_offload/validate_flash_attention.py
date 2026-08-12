"""Validate the first FL FlashAttention v2 offload implementation on one GPU."""

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
        fl_offload_modules=["FlashAttention"],
        fl_min_offloaded_tensor_size=0,
        fl_per_batch_offload_size=33,
        fl_activation_offload_stages=4,
        fl_activation_offload_stages_assignment=[0, 1, 2, 3],
        fl_use_comm_stream=False,
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("FL FlashAttention validation requires a CUDA GPU")
    import flash_attn

    original_args = offload._args
    offload._args = _runtime_args
    offload.reset_for_tests()

    # Q/K/V/O are 8 MiB each and LSE is 1 MiB: 33 MiB in total.
    shape = (1, 4096, 64, 16)
    inputs = [
        torch.randn(shape, dtype=torch.float16, device="cuda", requires_grad=True)
        for _ in range(3)
    ]
    baseline_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in inputs]
    baseline_output = flash_attn.flash_attn_func(
        *baseline_inputs, dropout_p=0.0, causal=True
    )
    baseline_loss = baseline_output.clone().float().square().mean()
    baseline_loss.backward()
    expected_grads = [tensor.grad.clone() for tensor in baseline_inputs]

    apply_te_patches()
    key = (0, 0, 0)
    try:
        with offload.record(key, group_num=4):
            output = flash_attn.flash_attn_func(
                *inputs, dropout_p=0.0, causal=True
            )
            loss = output.clone().float().square().mean()
        del output

        group = offload._GROUPS[key]
        captured = sum(
            tensor.x.numel() * tensor.x.element_size() for tensor in group.tensors
        )
        names = {tensor.tensor_name for tensor in group.tensors}
        expected_names = {"q", "k", "v", "output", "softmax_lse"}
        if captured != 33 << 20 or names != expected_names:
            raise AssertionError(
                f"captured={captured / (1 << 20):.2f} MiB names={sorted(names)}"
            )

        with offload.OffloadAsync(key, group_num=4) as context:
            context.issue(3)
        with offload.OnloadAsync(key, group_num=4) as context:
            context.issue(3)
        loss.backward()
        torch.cuda.synchronize()

        for actual, expected in zip(inputs, expected_grads):
            torch.testing.assert_close(actual.grad, expected, rtol=2e-3, atol=2e-3)
        print(
            "[FL FlashAttention check] PASSED: captured=33.00 MiB, "
            f"loss={loss.item():.8f}, Q/K/V gradients match",
            flush=True,
        )
    finally:
        restore_te_patches()
        offload.reset_for_tests()
        offload._args = original_args


if __name__ == "__main__":
    main()
