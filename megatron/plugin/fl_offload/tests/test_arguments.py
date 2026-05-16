"""Commit 2 unit tests: argparse extension + validate wrapper.

Covers:

* default values for all eight CLI flags,
* the legacy ``--schedule-method`` alias,
* ``chain_extra_args_provider`` with and without a caller-supplied provider,
* range / dependency / mutual-exclusion checks in ``validate_plugin_args``,
* the happy path where ``set_config`` is populated.

None of these tests need a GPU or a real Megatron training setup.
"""

import argparse
import os
import sys
import types
import unittest


# Make ``megatron/`` importable when the test file is run directly.
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
    ),
)


from megatron.plugin.fl_offload.arguments import (
    add_fl_offload_args,
    chain_extra_args_provider,
)
from megatron.plugin.fl_offload.config import (
    FlOffloadConfig,
    get_config,
    set_config,
)
from megatron.plugin.fl_offload.validate import (
    validate_args_wrapper,
    validate_plugin_args,
)


def _make_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(add_help=False)


def _parse(provider, argv):
    parser = _make_parser()
    provider(parser)
    return parser.parse_args(argv)


def _baseline_namespace(**overrides):
    """An ``args`` namespace that defaults all required attrs to safe values."""
    ns = types.SimpleNamespace(
        pipeline_schedule_backend="vanilla",
        fl_offload_enable=False,
        fl_offload_min_bytes=1 << 20,
        fl_offload_non_contiguous=False,
        fl_offload_pin_memory=True,
        fl_offload_ratio=1.0,
        fl_offload_per_batch_size=0.0,
        fl_offload_stages=1,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        fine_grained_activation_offloading=False,
        cpu_offloading=False,
        cpu_offloading_num_layers=0,
        use_dualpipev=False,
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


class TestAddArgs(unittest.TestCase):
    def test_all_defaults(self):
        parser = _make_parser()
        add_fl_offload_args(parser)
        args = parser.parse_args([])

        self.assertEqual(args.pipeline_schedule_backend, "vanilla")
        self.assertFalse(args.fl_offload_enable)
        self.assertEqual(args.fl_offload_min_bytes, 1 << 20)
        self.assertFalse(args.fl_offload_non_contiguous)
        self.assertTrue(args.fl_offload_pin_memory)
        self.assertEqual(args.fl_offload_ratio, 1.0)
        self.assertEqual(args.fl_offload_per_batch_size, 0.0)
        self.assertEqual(args.fl_offload_stages, 1)

    def test_legacy_alias(self):
        parser = _make_parser()
        add_fl_offload_args(parser)
        args = parser.parse_args(["--schedule-method", "interleaved_1f1b"])
        self.assertEqual(args.pipeline_schedule_backend, "interleaved_1f1b")

    def test_no_pin_memory_flag(self):
        parser = _make_parser()
        add_fl_offload_args(parser)
        args = parser.parse_args(["--fl-offload-no-pin-memory"])
        self.assertFalse(args.fl_offload_pin_memory)

    def test_idempotent_registration(self):
        """Calling ``add_fl_offload_args`` twice on the same parser is a no-op."""
        parser = _make_parser()
        add_fl_offload_args(parser)
        # Should not raise (argparse would otherwise complain about a
        # conflicting option).
        add_fl_offload_args(parser)
        args = parser.parse_args(["--fl-offload-enable"])
        self.assertTrue(args.fl_offload_enable)


class TestChainProvider(unittest.TestCase):
    def test_chain_without_caller(self):
        args = _parse(chain_extra_args_provider(None), ["--fl-offload-enable"])
        self.assertTrue(args.fl_offload_enable)

    def test_chain_with_caller_provider(self):
        def caller_provider(parser):
            parser.add_argument("--caller-flag", action="store_true")
            return parser

        args = _parse(
            chain_extra_args_provider(caller_provider),
            ["--caller-flag", "--fl-offload-stages", "4"],
        )
        self.assertTrue(args.caller_flag)
        self.assertEqual(args.fl_offload_stages, 4)

    def test_caller_provider_may_return_none(self):
        def caller_provider(parser):
            parser.add_argument("--quiet", action="store_true")
            # Intentionally don't return the parser.

        args = _parse(chain_extra_args_provider(caller_provider), ["--quiet"])
        self.assertTrue(args.quiet)
        # Plugin args still got registered.
        self.assertEqual(args.fl_offload_ratio, 1.0)


class TestValidateRanges(unittest.TestCase):
    def setUp(self):
        self._original_cfg = get_config()

    def tearDown(self):
        set_config(self._original_cfg)

    def test_invalid_backend(self):
        ns = _baseline_namespace(pipeline_schedule_backend="garbage")
        with self.assertRaises(AssertionError):
            validate_plugin_args(ns)

    def test_negative_min_bytes(self):
        ns = _baseline_namespace(fl_offload_min_bytes=-1)
        with self.assertRaises(AssertionError):
            validate_plugin_args(ns)

    def test_ratio_below_zero(self):
        ns = _baseline_namespace(fl_offload_ratio=-0.1)
        with self.assertRaises(AssertionError):
            validate_plugin_args(ns)

    def test_ratio_above_one(self):
        ns = _baseline_namespace(fl_offload_ratio=1.5)
        with self.assertRaises(AssertionError):
            validate_plugin_args(ns)

    def test_negative_per_batch_size(self):
        ns = _baseline_namespace(fl_offload_per_batch_size=-1)
        with self.assertRaises(AssertionError):
            validate_plugin_args(ns)

    def test_zero_stages(self):
        ns = _baseline_namespace(fl_offload_stages=0)
        with self.assertRaises(AssertionError):
            validate_plugin_args(ns)


class TestValidateBackendDependencies(unittest.TestCase):
    def setUp(self):
        self._original_cfg = get_config()

    def tearDown(self):
        set_config(self._original_cfg)

    def test_interleaved_requires_pp_gt_1(self):
        ns = _baseline_namespace(
            pipeline_schedule_backend="interleaved_1f1b",
            pipeline_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=2,
        )
        with self.assertRaises(AssertionError):
            validate_plugin_args(ns)

    def test_interleaved_requires_vpp(self):
        ns = _baseline_namespace(
            pipeline_schedule_backend="interleaved_1f1b",
            pipeline_model_parallel_size=2,
            virtual_pipeline_model_parallel_size=None,
        )
        with self.assertRaises(AssertionError):
            validate_plugin_args(ns)

    def test_interleaved_passes_when_pp_and_vpp_set(self):
        ns = _baseline_namespace(
            pipeline_schedule_backend="interleaved_1f1b",
            pipeline_model_parallel_size=2,
            virtual_pipeline_model_parallel_size=2,
            fl_offload_enable=True,
        )
        cfg = validate_plugin_args(ns)
        self.assertEqual(cfg.pipeline_schedule_backend, "interleaved_1f1b")
        self.assertTrue(cfg.enable)


class TestValidateMutualExclusion(unittest.TestCase):
    def setUp(self):
        self._original_cfg = get_config()

    def tearDown(self):
        set_config(self._original_cfg)

    def _enabled_ns(self, **overrides):
        return _baseline_namespace(fl_offload_enable=True, **overrides)

    def test_conflict_with_fine_grained(self):
        ns = self._enabled_ns(fine_grained_activation_offloading=True)
        with self.assertRaises(AssertionError) as ctx:
            validate_plugin_args(ns)
        self.assertIn("fine_grained_activation_offloading", str(ctx.exception))

    def test_conflict_with_cpu_offloading(self):
        ns = self._enabled_ns(cpu_offloading=True)
        with self.assertRaises(AssertionError) as ctx:
            validate_plugin_args(ns)
        self.assertIn("cpu_offloading", str(ctx.exception))

    def test_conflict_with_cpu_offloading_num_layers(self):
        ns = self._enabled_ns(cpu_offloading_num_layers=2)
        with self.assertRaises(AssertionError) as ctx:
            validate_plugin_args(ns)
        self.assertIn("cpu_offloading_num_layers>0", str(ctx.exception))

    def test_conflict_with_use_dualpipev(self):
        ns = self._enabled_ns(use_dualpipev=True)
        with self.assertRaises(AssertionError) as ctx:
            validate_plugin_args(ns)
        self.assertIn("use_dualpipev", str(ctx.exception))

    def test_conflict_reported_only_when_enabled(self):
        # All conflicts at once but enable=False → silently OK.
        ns = _baseline_namespace(
            fl_offload_enable=False,
            fine_grained_activation_offloading=True,
            cpu_offloading=True,
            cpu_offloading_num_layers=2,
            use_dualpipev=True,
        )
        cfg = validate_plugin_args(ns)
        self.assertFalse(cfg.enable)


class TestValidateHappyPath(unittest.TestCase):
    def setUp(self):
        self._original_cfg = get_config()

    def tearDown(self):
        set_config(self._original_cfg)

    def test_set_config_populated(self):
        ns = _baseline_namespace(
            pipeline_schedule_backend="interleaved_1f1b",
            pipeline_model_parallel_size=4,
            virtual_pipeline_model_parallel_size=2,
            fl_offload_enable=True,
            fl_offload_min_bytes=2048,
            fl_offload_non_contiguous=True,
            fl_offload_pin_memory=False,
            fl_offload_ratio=0.5,
            fl_offload_per_batch_size=128.0,
            fl_offload_stages=4,
        )
        cfg = validate_plugin_args(ns)

        self.assertIs(get_config(), cfg)
        self.assertTrue(cfg.enable)
        self.assertEqual(cfg.pipeline_schedule_backend, "interleaved_1f1b")
        self.assertEqual(cfg.min_bytes, 2048)
        self.assertTrue(cfg.non_contiguous)
        self.assertFalse(cfg.pin_memory)
        self.assertEqual(cfg.ratio, 0.5)
        self.assertEqual(cfg.per_batch_size, 128.0)
        self.assertEqual(cfg.stages, 4)

    def test_defaults_yield_disabled_config(self):
        ns = _baseline_namespace()
        cfg = validate_plugin_args(ns)
        self.assertFalse(cfg.enable)
        self.assertEqual(cfg.pipeline_schedule_backend, "vanilla")


class TestValidateArgsWrapper(unittest.TestCase):
    """Confirm the wrapper preserves the upstream contract."""

    def setUp(self):
        self._original_cfg = get_config()

    def tearDown(self):
        set_config(self._original_cfg)

    def test_inner_validator_runs_first(self):
        call_order = []

        def inner_validate(args, defaults):
            call_order.append("inner")
            # The upstream contract: inner returns the args.
            args.touched_by_inner = True
            return args

        wrapped = validate_args_wrapper(inner_validate)
        ns = _baseline_namespace()
        out = wrapped(ns)

        self.assertEqual(call_order, ["inner"])
        self.assertIs(out, ns)
        self.assertTrue(getattr(out, "touched_by_inner", False))
        # Plugin validator must have run too.
        self.assertFalse(get_config().enable)

    def test_plugin_check_runs_even_if_inner_passes(self):
        def inner_validate(args, defaults):
            return args

        wrapped = validate_args_wrapper(inner_validate)
        ns = _baseline_namespace(
            fl_offload_enable=True, cpu_offloading=True
        )
        with self.assertRaises(AssertionError):
            wrapped(ns)


class TestApplyEntryPoint(unittest.TestCase):
    """End-to-end check on the apply() public surface."""

    def setUp(self):
        self._original_cfg = get_config()

    def tearDown(self):
        set_config(self._original_cfg)

    def test_apply_returns_callable_provider(self):
        from megatron.plugin.fl_offload import apply

        provider, validator = apply()
        self.assertTrue(callable(provider))
        self.assertIsNone(validator)

        # Provider must register the plugin's flags.
        parser = _make_parser()
        provider(parser)
        args = parser.parse_args(["--fl-offload-enable", "--fl-offload-stages", "2"])
        self.assertTrue(args.fl_offload_enable)
        self.assertEqual(args.fl_offload_stages, 2)

    def test_apply_chains_user_validator(self):
        from megatron.plugin.fl_offload import apply

        calls = []

        def user_validator(args, defaults=None):
            calls.append("user")
            return args

        provider, validator = apply(validate_args=user_validator)
        self.assertTrue(callable(validator))

        ns = _baseline_namespace()
        validator(ns)
        self.assertEqual(calls, ["user"])
        self.assertFalse(get_config().enable)


if __name__ == "__main__":
    unittest.main()
