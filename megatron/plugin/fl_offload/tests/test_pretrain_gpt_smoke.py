"""Smoke tests for the ``pretrain_gpt.py`` ↔ ``fl_offload.apply()`` wiring (commit 7).

CPU-only.  Exercises:

* ``apply()`` returns a callable provider that registers all 10 fl-offload
  flags (8 from Commit 2 + 2 from Commit 7).
* ``apply(None)`` (no user provider) yields a provider that just adds
  fl-offload flags.
* ``apply(user_provider)`` runs ``user_provider`` first so the user's
  flags are available before ours.
* ``validate_plugin_args`` refuses ``--fl-offload-enable`` with
  ``--cuda-graph-impl=local`` unless ``--fl-offload-allow-cuda-graph`` is
  also set.
* ``FlOffloadConfig.report_interval`` / ``allow_cuda_graph`` propagate
  through ``validate_plugin_args``.
"""

from __future__ import annotations

import argparse
import types
import unittest

from megatron.plugin.fl_offload._patch import MegatronPatchesManager
from megatron.plugin.fl_offload.apply import apply
from megatron.plugin.fl_offload.config import FlOffloadConfig, get_config, set_config
from megatron.plugin.fl_offload.validate import validate_plugin_args


_EXPECTED_FLAGS = {
    "--pipeline-schedule-backend",
    "--fl-offload-enable",
    "--fl-offload-min-bytes",
    "--fl-offload-non-contiguous",
    "--fl-offload-no-pin-memory",
    "--fl-offload-ratio",
    "--fl-offload-per-batch-size",
    "--fl-offload-stages",
    "--fl-offload-report-interval",
    "--fl-offload-allow-cuda-graph",
}


def _empty_args() -> argparse.Namespace:
    return argparse.Namespace(
        pipeline_schedule_backend="vanilla",
        fl_offload_enable=False,
        fl_offload_min_bytes=1 << 20,
        fl_offload_non_contiguous=False,
        fl_offload_pin_memory=True,
        fl_offload_ratio=1.0,
        fl_offload_per_batch_size=0.0,
        fl_offload_stages=1,
        fl_offload_report_interval=10,
        fl_offload_allow_cuda_graph=False,
        fine_grained_activation_offloading=False,
        cpu_offloading=False,
        cpu_offloading_num_layers=0,
        use_dualpipev=False,
        cuda_graph_impl="none",
        gradient_accumulation_fusion=False,
    )


class _BaseCase(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_cfg = get_config()
        set_config(FlOffloadConfig(enable=False))
        MegatronPatchesManager._reset_for_tests()

    def tearDown(self) -> None:
        MegatronPatchesManager._reset_for_tests()
        set_config(self._saved_cfg)


class TestApplyProviderChain(_BaseCase):
    def test_apply_with_none_user_provider_registers_all_flags(self) -> None:
        extra_args, _ = apply(extra_args_provider=None)
        parser = argparse.ArgumentParser()
        extra_args(parser)
        registered = set(parser._option_string_actions.keys())
        for flag in _EXPECTED_FLAGS:
            self.assertIn(flag, registered, f"missing flag: {flag}")

    def test_apply_runs_user_provider_first(self) -> None:
        sentinel = {"user_called": False}

        def user_provider(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
            sentinel["user_called"] = True
            parser.add_argument("--user-flag", default=None)
            return parser

        extra_args, _ = apply(extra_args_provider=user_provider)
        parser = argparse.ArgumentParser()
        extra_args(parser)
        self.assertTrue(sentinel["user_called"])
        self.assertIn("--user-flag", parser._option_string_actions)
        for flag in _EXPECTED_FLAGS:
            self.assertIn(flag, parser._option_string_actions)

    def test_apply_validator_is_none_when_no_user_validator(self) -> None:
        _, validator = apply()
        self.assertIsNone(validator)


class TestCudaGraphGuard(_BaseCase):
    def test_disabled_offload_ignores_cuda_graph(self) -> None:
        args = _empty_args()
        args.cuda_graph_impl = "local"
        args.fl_offload_enable = False
        # No raise.
        validate_plugin_args(args)

    def test_local_cuda_graph_with_enable_raises(self) -> None:
        args = _empty_args()
        args.fl_offload_enable = True
        args.cuda_graph_impl = "local"
        with self.assertRaises(AssertionError) as ctx:
            validate_plugin_args(args)
        self.assertIn("cuda-graph-impl", str(ctx.exception))
        self.assertIn("--fl-offload-allow-cuda-graph", str(ctx.exception))

    def test_allow_cuda_graph_unblocks(self) -> None:
        args = _empty_args()
        args.fl_offload_enable = True
        args.cuda_graph_impl = "local"
        args.fl_offload_allow_cuda_graph = True
        cfg = validate_plugin_args(args)
        self.assertTrue(cfg.allow_cuda_graph)
        self.assertTrue(cfg.enable)

    def test_none_impl_passes(self) -> None:
        args = _empty_args()
        args.fl_offload_enable = True
        args.cuda_graph_impl = "none"
        cfg = validate_plugin_args(args)
        self.assertTrue(cfg.enable)
        self.assertFalse(cfg.allow_cuda_graph)

    def test_missing_impl_attr_passes(self) -> None:
        args = _empty_args()
        args.fl_offload_enable = True
        # Drop the attribute entirely; should be treated as "none".
        del args.cuda_graph_impl
        cfg = validate_plugin_args(args)
        self.assertTrue(cfg.enable)


class TestWgradFusionGuard(_BaseCase):
    """fl-offload + gradient_accumulation_fusion must be refused.

    saved_tensors_hooks strip the Parameter wrapper off saved weights;
    TE's fused-wgrad protocol then returns ``wgrad=None`` and DDP's
    overlap_grad_reduce hook asserts.  Until the explicit-pack TE patch
    (Commit 7.2) lands, the combination fails fast at validate time.
    """

    def test_enable_with_fusion_raises(self) -> None:
        args = _empty_args()
        args.fl_offload_enable = True
        args.gradient_accumulation_fusion = True
        with self.assertRaises(AssertionError) as ctx:
            validate_plugin_args(args)
        self.assertIn("no-gradient-accumulation-fusion", str(ctx.exception))

    def test_enable_without_fusion_passes(self) -> None:
        args = _empty_args()
        args.fl_offload_enable = True
        args.gradient_accumulation_fusion = False
        cfg = validate_plugin_args(args)
        self.assertTrue(cfg.enable)

    def test_disabled_offload_ignores_fusion(self) -> None:
        args = _empty_args()
        args.fl_offload_enable = False
        args.gradient_accumulation_fusion = True
        # No raise.
        validate_plugin_args(args)

    def test_missing_fusion_attr_passes(self) -> None:
        args = _empty_args()
        args.fl_offload_enable = True
        del args.gradient_accumulation_fusion
        cfg = validate_plugin_args(args)
        self.assertTrue(cfg.enable)


class TestNewFieldsPropagate(_BaseCase):
    def test_report_interval_propagates(self) -> None:
        args = _empty_args()
        args.fl_offload_report_interval = 5
        cfg = validate_plugin_args(args)
        self.assertEqual(cfg.report_interval, 5)

    def test_default_report_interval_is_10(self) -> None:
        args = _empty_args()
        # default in _empty_args is 10
        cfg = validate_plugin_args(args)
        self.assertEqual(cfg.report_interval, 10)

    def test_negative_report_interval_rejected(self) -> None:
        args = _empty_args()
        args.fl_offload_report_interval = -1
        with self.assertRaises(AssertionError):
            validate_plugin_args(args)


if __name__ == "__main__":
    unittest.main()
