from types import SimpleNamespace

import pytest
import torch

from megatron.plugin.fl_offload import offload


def _args():
    return SimpleNamespace(
        fl_patch_te=True,
        fl_activation_offload_stages=4,
        fl_activation_offload_stages_assignment=[0, 1, 2, 3],
        fl_per_batch_offload_size=1,
        fl_offload_modules=[],
        fl_min_offloaded_tensor_size=0,
    )


def test_live_key_and_reload_order_are_guarded():
    original_args = offload._args
    offload._args = _args
    try:
        offload.reset_for_tests()
        key = (0, 0, 0)
        with offload.record(key, group_num=4):
            pass
        with pytest.raises(RuntimeError, match="crossed the training-step boundary"):
            offload.assert_runtime_idle()

        with pytest.raises(RuntimeError, match="still active"):
            with offload.record(key, group_num=4):
                pass
        with pytest.raises(RuntimeError, match="cannot enter reload"):
            offload.OnloadAsync(key, group_num=4).__enter__()
    finally:
        offload.reset_for_tests()
        offload._args = original_args


def test_premature_unpack_is_guarded():
    wrapped = offload.TensorWrap(torch.ones(1))
    wrapped.x = None
    with pytest.raises(RuntimeError, match="before its reload completed"):
        offload.unpack_hook(offload.TensorPack(wrapped))


def test_stage_count_and_range_are_guarded():
    original_args = offload._args
    offload._args = _args
    try:
        offload.reset_for_tests()
        key = (0, 0, 1)
        with offload.record(key, group_num=4):
            pass

        with pytest.raises(ValueError, match="captured with 4 stages"):
            offload.OffloadAsync(key, group_num=2)
        context = offload.OffloadAsync(key, group_num=4)
        with pytest.raises(ValueError, match="outside"):
            context.issue(4)
    finally:
        offload.reset_for_tests()
        offload._args = original_args


def test_rope_frequency_buffer_is_not_captured():
    args = _args()
    args.fl_offload_modules = ["LayerNormLinear"]
    original_args = offload._args
    offload._args = lambda: args
    try:
        offload.reset_for_tests()
        key = (0, 0, 2)
        rope_frequencies = torch.ones(8, 1, 1, 16)
        with offload.record(key, group_num=4):
            packed = offload.pack_hook(rope_frequencies, op_name="LayerNormLinear")

        assert offload._GROUPS[key].tensors == []
        assert offload.unpack_hook(packed) is rope_frequencies
    finally:
        offload.reset_for_tests()
        offload._args = original_args
