"""Commit 1 smoke tests.

These exist solely to confirm that the skeleton is import-clean and that the
default config / runtime are no-ops.  Later commits will add real unit
tests; this file is intentionally minimal.
"""

import contextlib
import os
import sys
import unittest


# Allow running this file directly from a clone without installing the
# package.  ``megatron/`` lives four levels above this file.
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
    ),
)


class TestSkeleton(unittest.TestCase):
    def test_package_imports_without_side_effects(self):
        # Importing the package and the apply entry point must not change any
        # interesting global state.  We assert by re-importing and checking
        # that the default config is still pristine.
        import megatron.plugin.fl_offload as pkg

        self.assertTrue(hasattr(pkg, "apply"))
        self.assertTrue(hasattr(pkg, "FlOffloadConfig"))
        self.assertTrue(hasattr(pkg, "get_config"))
        self.assertTrue(hasattr(pkg, "set_config"))

    def test_default_config_is_disabled(self):
        from megatron.plugin.fl_offload import FlOffloadConfig, get_config

        cfg = get_config()
        self.assertIsInstance(cfg, FlOffloadConfig)
        self.assertFalse(cfg.enable)
        self.assertEqual(cfg.pipeline_schedule_backend, "vanilla")
        self.assertEqual(cfg.stages, 1)
        self.assertEqual(cfg.ratio, 1.0)
        self.assertEqual(cfg.per_batch_size, 0)
        self.assertTrue(cfg.pin_memory)
        self.assertFalse(cfg.non_contiguous)
        self.assertEqual(cfg.report_interval, 0)
        self.assertFalse(cfg.allow_cuda_graph)

    def test_set_config_round_trip(self):
        from megatron.plugin.fl_offload import (
            FlOffloadConfig,
            get_config,
            set_config,
        )

        original = get_config()
        try:
            new_cfg = FlOffloadConfig(enable=True, stages=4, ratio=0.5)
            set_config(new_cfg)
            self.assertIs(get_config(), new_cfg)
            self.assertTrue(get_config().enable)
        finally:
            # Restore so the rest of the test suite sees a pristine singleton.
            set_config(original)

    def test_set_config_rejects_wrong_type(self):
        from megatron.plugin.fl_offload import set_config

        with self.assertRaises(TypeError):
            set_config({"enable": True})  # type: ignore[arg-type]

    def test_apply_returns_callable_provider_and_optional_validator(self):
        from megatron.plugin.fl_offload import apply

        # No validator passed: returns (chained_provider, None).
        provider, validator = apply()
        self.assertTrue(callable(provider))
        self.assertIsNone(validator)

        # With a validator: both slots are callables, neither is identity to
        # the inputs (both get wrapped).
        user_provider = lambda parser: parser  # noqa: E731
        user_validator = lambda args, defaults=None: args  # noqa: E731
        out_provider, out_validator = apply(user_provider, user_validator)
        self.assertTrue(callable(out_provider))
        self.assertTrue(callable(out_validator))
        self.assertIsNot(out_provider, user_provider)
        self.assertIsNot(out_validator, user_validator)

    def test_runtime_singleton_is_disabled(self):
        from megatron.plugin.fl_offload.runtime import (
            get_pipeline_offload_runtime,
        )

        runtime = get_pipeline_offload_runtime()
        self.assertFalse(runtime.enabled())

        # Both façade methods must return a context manager that does
        # nothing.  ``nullcontext`` is the cheapest such object.
        fwd = runtime.forward_microbatch(
            phase="test", virtual_microbatch_id=0, model_chunk_id=0
        )
        self.assertIsInstance(fwd, contextlib.AbstractContextManager)
        with fwd:
            pass

        bwd = runtime.backward_microbatch(
            phase="test", virtual_microbatch_id=0, model_chunk_id=0
        )
        self.assertIsInstance(bwd, contextlib.AbstractContextManager)
        with bwd:
            pass

    def test_offload_async_stubs(self):
        from megatron.plugin.fl_offload.runtime import (
            OffloadAsync,
            OnloadAsync,
        )

        with OffloadAsync(key=("test", 0), stages=4) as ctx:
            ctx.issue(0)
            ctx.issue(3)

        with OnloadAsync(key=("test", 0), stages=4) as ctx:
            ctx.issue(0)
            ctx.issue(3)


if __name__ == "__main__":
    unittest.main()
