"""CPU checks for the TE FusedAttention saved-tensor tuple adapter."""

from types import SimpleNamespace

import torch

from megatron.plugin.fl_offload import offload


def test_fused_attention_pack_restores_tensor_layout(monkeypatch):
    args = SimpleNamespace(
        fl_patch_te=True,
        fl_offload_modules=["Attention"],
        attention_backend="fused",
        fl_min_offloaded_tensor_size=0,
    )
    monkeypatch.setattr(offload, "_args", lambda: args)
    monkeypatch.setattr(offload, "_OFFLOAD_TENSORS", [])

    fp8_like_tensor = torch.nn.Parameter(torch.randn(2, 3), requires_grad=False)
    qkvo = tuple(torch.randn(2, 3) for _ in range(4))
    softmax_stats = torch.randn(1, 2, 3, 1, dtype=torch.float32)
    rng_state = torch.tensor([1, 2], dtype=torch.int64)

    packed_fp8, packed_qkvo, packed_aux, packs = (
        offload.pack_fused_attention_saved_tensors(
            (fp8_like_tensor, None, None, None),
            qkvo,
            [softmax_stats, rng_state],
        )
    )

    assert packed_fp8[0] is fp8_like_tensor
    assert packed_fp8[1:] == (None, None, None)
    assert packed_qkvo == (None, None, None, None)
    assert packed_aux == [None, rng_state]
    assert [pack.tensor_name for _, _, pack in packs] == [
        "q",
        "k",
        "v",
        "output",
        "softmax_stats",
    ]

    restored_fp8, restored_qkvo, restored_aux = (
        offload.unpack_fused_attention_saved_tensors(
            packs,
            packed_fp8,
            packed_qkvo,
            packed_aux,
        )
    )

    assert restored_fp8[0] is fp8_like_tensor
    assert restored_fp8[1:] == (None, None, None)
    assert all(actual is expected for actual, expected in zip(restored_qkvo, qkvo))
    assert restored_aux[0] is softmax_stats
    assert restored_aux[1] is rng_state
