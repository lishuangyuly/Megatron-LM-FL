"""Tests for the schedule selector + monkey-patch (commit 6).

Single-process, CPU-only.  We exercise the public surface of
``apply()`` against the real ``megatron.core.pipeline_parallel.schedules``
module (which imports cleanly without CUDA) and verify:

* ``enable=False`` → ``get_forward_backward_func(...)`` returns the
  upstream FL callable unchanged (identity check).
* ``enable=True`` + interleaved selection → returns a *wrapped*
  callable (not identity).
* ``enable=True`` + non-interleaved / no-pipelining selection →
  the wrapper raises ``NotImplementedError`` when invoked.
* ``MegatronPatchesManager`` walks ``sys.modules`` and updates
  ``from ... import get_forward_backward_func`` aliases to the wrapper.
* ``_assert_no_dualpipev()`` raises when the dualpipev getter is set.
"""

from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

from megatron.plugin.fl_offload._patch import MegatronPatchesManager
from megatron.plugin.fl_offload.apply import apply
from megatron.plugin.fl_offload.config import FlOffloadConfig, get_config, set_config
from megatron.plugin.fl_offload.selector import (
    _assert_no_dualpipev,
    get_forward_backward_func_wrapper,
)


def _disabled_config() -> FlOffloadConfig:
    return FlOffloadConfig(enable=False)


def _enabled_config() -> FlOffloadConfig:
    return FlOffloadConfig(
        enable=True,
        pipeline_schedule_backend="interleaved_1f1b",
        ratio=1.0,
    )


class _BaseCase(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_cfg = get_config()
        set_config(_disabled_config())
        MegatronPatchesManager._reset_for_tests()

    def tearDown(self) -> None:
        MegatronPatchesManager._reset_for_tests()
        set_config(self._saved_cfg)


class TestApplyIdempotent(_BaseCase):
    def test_apply_registers_once(self) -> None:
        apply()
        first_count = len(MegatronPatchesManager._patches)
        apply()  # second invocation
        self.assertEqual(len(MegatronPatchesManager._patches), first_count)
        self.assertTrue(MegatronPatchesManager._applied)


class TestEnableFalseIsIdentity(_BaseCase):
    def test_disabled_returns_original_interleaved(self) -> None:
        from megatron.core.pipeline_parallel import schedules as core_sched

        original_interleaved = core_sched.forward_backward_pipelining_with_interleaving
        apply()
        # apply() does not touch enable; it defaults to False from setUp.
        chosen = core_sched.get_forward_backward_func(pp_size=2, vp_size=2)
        self.assertIs(chosen, original_interleaved)

    def test_disabled_returns_original_no_pipelining(self) -> None:
        from megatron.core.pipeline_parallel import schedules as core_sched

        original_no_pp = core_sched.forward_backward_no_pipelining
        apply()
        chosen = core_sched.get_forward_backward_func(pp_size=1, vp_size=None)
        self.assertIs(chosen, original_no_pp)


class TestEnableTrueWrapsInterleaved(_BaseCase):
    def test_enabled_interleaved_returns_wrapper(self) -> None:
        from megatron.core.pipeline_parallel import schedules as core_sched

        original_interleaved = core_sched.forward_backward_pipelining_with_interleaving
        apply()
        set_config(_enabled_config())
        chosen = core_sched.get_forward_backward_func(pp_size=2, vp_size=2)
        self.assertIsNot(chosen, original_interleaved)
        # The wrapper should carry over functools.wraps metadata.
        self.assertEqual(
            chosen.__wrapped__,
            original_interleaved,
        )

    def test_enabled_no_pp_path_stub_raises(self) -> None:
        from megatron.core.pipeline_parallel import schedules as core_sched

        apply()
        set_config(_enabled_config())
        with self.assertRaises(NotImplementedError) as ctx:
            core_sched.get_forward_backward_func(pp_size=1, vp_size=None)
        self.assertIn("Commit 7", str(ctx.exception))

    def test_enabled_without_interleaving_path_stub_raises(self) -> None:
        from megatron.core.pipeline_parallel import schedules as core_sched

        apply()
        set_config(_enabled_config())
        with self.assertRaises(NotImplementedError) as ctx:
            core_sched.get_forward_backward_func(pp_size=2, vp_size=None)
        self.assertIn("Commit 7", str(ctx.exception))


class TestAliasReplacement(_BaseCase):
    def test_pre_apply_alias_is_synced(self) -> None:
        """``from schedules import get_forward_backward_func`` in a 3rd
        module before apply() should still see the wrapper afterwards."""
        from megatron.core.pipeline_parallel import schedules as core_sched

        # Create a fake module that imports the symbol *before* apply().
        fake = types.ModuleType("fl_offload_test_alias_module")
        fake.gfbf_alias = core_sched.get_forward_backward_func
        sys.modules["fl_offload_test_alias_module"] = fake
        try:
            original = core_sched.get_forward_backward_func
            apply()
            new = core_sched.get_forward_backward_func
            self.assertIsNot(new, original)
            # _replace_matching_aliases should have updated the fake module.
            self.assertIs(fake.gfbf_alias, new)
        finally:
            sys.modules.pop("fl_offload_test_alias_module", None)


class TestDualpipevGuard(_BaseCase):
    def test_assert_raises_when_dualpipev_size_set(self) -> None:
        from megatron.core import parallel_state as ps

        with mock.patch.object(
            ps,
            "get_dualpipev_pipeline_model_parallel_world_size",
            return_value=2,
            create=True,
        ):
            with self.assertRaises(AssertionError):
                _assert_no_dualpipev()

    def test_assert_passes_when_dualpipev_size_none(self) -> None:
        from megatron.core import parallel_state as ps

        with mock.patch.object(
            ps,
            "get_dualpipev_pipeline_model_parallel_world_size",
            return_value=None,
            create=True,
        ):
            # Should not raise.
            _assert_no_dualpipev()


class TestWrapperFactory(_BaseCase):
    def test_factory_independent_of_apply(self) -> None:
        """The factory is callable on a bare callable for unit-test use."""

        def fake_get_fbf(pp_size=None, vp_size=None):
            return "interleaved_sentinel"

        wrapped = get_forward_backward_func_wrapper(fake_get_fbf)
        # Disabled → identity passthrough at *call* time.
        set_config(_disabled_config())
        self.assertEqual(wrapped(pp_size=2, vp_size=2), "interleaved_sentinel")


if __name__ == "__main__":
    unittest.main()
