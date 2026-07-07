"""Argument compatibility tests for the direct FL offload port."""

import argparse

from megatron.training.arguments import add_megatron_arguments


def test_fl_offload_arguments_have_an_independent_namespace():
    parser = add_megatron_arguments(argparse.ArgumentParser(allow_abbrev=False))

    option_strings = [option for action in parser._actions for option in action.option_strings]
    assert option_strings.count("--offload-modules") == 1
    assert option_strings.count("--min-offloaded-tensor-size") == 1
    assert option_strings.count("--fl-offload-modules") == 1
    assert option_strings.count("--fl-min-offloaded-tensor-size") == 1
    assert option_strings.count("--profile-pp-semantics") == 1
    assert option_strings.count("--profile-dir") == 1

    args = parser.parse_args(
        [
            "--fl-patch-te",
            "--fl-offload-modules",
            "LayerNormLinear",
            "GroupedLinear",
            "--fl-min-offloaded-tensor-size",
            "1048576",
            "--fl-activation-offload-ratio",
            "1.0",
            "--fl-per-batch-offload-size",
            "1",
            "--profile-pp-semantics",
            "--profile-dir",
            "/tmp/fl-trace",
        ]
    )

    assert args.fl_patch_te is True
    assert args.fl_offload_modules == ["LayerNormLinear", "GroupedLinear"]
    assert args.fl_min_offloaded_tensor_size == 1048576
    assert args.fl_activation_offload_ratio == [1.0]
    assert args.fl_per_batch_offload_size == 1
    assert args.profile_pp_semantics is True
    assert args.profile_dir == "/tmp/fl-trace"
